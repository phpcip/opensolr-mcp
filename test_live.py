#!/usr/bin/env python3
"""Live end-to-end test of opensolr-mcp against PRODUCTION Opensolr.

Exercises every public method of ``opensolr_mcp.client`` (plus the three pure
builders) and every MCP tool registered in ``opensolr_mcp.server``, asserting on
VALUES rather than on "no exception was raised".

Safe to re-run:
  * it never writes to, or deletes from, the seeded read-only demo index;
  * everything that writes goes into two throwaway indexes whose names carry a
    fresh random id per run, deleted in a ``finally`` block — including when the
    run blows up half way;
  * it deletes ONLY the two indexes it created. The demo account is shared and
    the sibling packages' suites run against it concurrently, so a blanket sweep
    of ``mcp_t_*`` would rip an index out from under a live run.

Run:  .venv/bin/python test_live.py
Exit: 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque

import httpx

# httpx logs every request at INFO and the mcp package installs a rich handler on
# import, which buries the ✔/✘ lines this suite exists to print.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# Credentials — the public MCP demo account (throwaway by design)              #
# --------------------------------------------------------------------------- #
EMAIL = "mcp@opensolr.com"
API_KEY = "420b8b23e7b12dc8ab838932145a5065"
DEMO = "mcp_demo_d1__dense"          # seeded, READ ONLY — never written to

# The MCP server builds its client from the environment, so these must be set
# before ``opensolr_mcp.server`` is imported (it is imported below, after the
# rate limiter is installed).
os.environ["OPENSOLR_EMAIL"] = EMAIL
os.environ["OPENSOLR_API_KEY"] = API_KEY


# --------------------------------------------------------------------------- #
# Rate limiter                                                                  #
# --------------------------------------------------------------------------- #
# The Opensolr API allows 30 requests/minute per account and this suite makes
# roughly 45. Rather than sprinkling sleeps through the tests (which drift the
# moment a check is added), every httpx request is funnelled through one rolling
# 60-second window. Only the two API hosts are counted: direct-to-Solr traffic
# goes to *.solrcluster.com and is not governed by the API rate limiter.
#
# This patches httpx, NOT the package — the package source is never modified.
_API_HOSTS = {"opensolr.com", "api.opensolr.com"}
_MAX_PER_MIN = 26                     # headroom under the documented 30
_stamps: deque = deque()
_pace_lock = threading.Lock()
_request_count = {"api": 0, "solr": 0}

_orig_request = httpx.Client.request


def _paced_request(self, method, url, *args, **kwargs):
    """Throttle API-host requests to _MAX_PER_MIN per rolling minute."""
    try:
        host = httpx.URL(url).host
    except Exception:
        host = ""
    if host in _API_HOSTS:
        with _pace_lock:
            while True:
                now = time.monotonic()
                while _stamps and now - _stamps[0] > 60.0:
                    _stamps.popleft()
                if len(_stamps) < _MAX_PER_MIN:
                    break
                wait = 60.0 - (now - _stamps[0]) + 0.5
                print(f"   … rate limit: pausing {wait:.0f}s", flush=True)
                time.sleep(wait)
            _stamps.append(time.monotonic())
        _request_count["api"] += 1
    else:
        _request_count["solr"] += 1
    return _orig_request(self, method, url, *args, **kwargs)


httpx.Client.request = _paced_request

# Imported AFTER the limiter is installed and the env vars are set.
from opensolr_mcp import __version__                                 # noqa: E402
from opensolr_mcp import server as S                                 # noqa: E402

logging.getLogger().setLevel(logging.WARNING)
from opensolr_mcp.client import (                                    # noqa: E402
    BATCH_EMBED_MAX,
    DOC_FENCE,
    FRESH_BIAS_FUNCTION,
    OpensolrClient,
    OpensolrError,
    apply_fresh_bias,
    build_context,
    build_instruction,
    resolve_location,
)


# --------------------------------------------------------------------------- #
# Check harness                                                                 #
# --------------------------------------------------------------------------- #
PASSED = 0
FAILED = 0
FAILURES: list[tuple[str, str]] = []
SKIP = object()          # returned by check() when the check itself failed


class Expected(AssertionError):
    """An assertion about a returned VALUE that did not hold."""


def need(cond: bool, msg: str) -> None:
    """Assert ``cond``, raising a failure that names what was expected."""
    if not cond:
        raise Expected(msg)


def check(label, fn, idempotent: bool = True):
    """Run one check. ``fn`` may return a detail string, a (detail, value) pair,
    or a bare value. Prints one ✔/✘ line and keeps the tallies.

    ``idempotent=False`` marks a check that writes (creates an index, queues an
    ingestion job, deletes documents): those are never re-run after a dropped
    connection, because a half-applied write retried is a different test.
    """
    global PASSED, FAILED
    try:
        result = retry_transport(fn) if idempotent else fn()
    except Exception as exc:                                  # noqa: BLE001
        FAILED += 1
        why = f"{type(exc).__name__}: {exc}"
        FAILURES.append((label, why))
        print(f"✘ {label} — {why}", flush=True)
        return SKIP
    detail, value = None, result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
        detail, value = result
    elif isinstance(result, str):
        detail, value = result, result
    PASSED += 1
    print(f"✔ {label}" + (f" — {detail}" if detail else ""), flush=True)
    return value


def section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(2, 62 - len(title)), flush=True)


def skipped(label: str, why: str) -> None:
    """A check that could not run because a prerequisite failed. Counts as a
    failure — a suite that silently drops checks is not honest about coverage."""
    global FAILED
    FAILED += 1
    FAILURES.append((label, f"skipped: {why}"))
    print(f"✘ {label} — skipped: {why}", flush=True)


# --------------------------------------------------------------------------- #
# Shared state between checks                                                   #
# --------------------------------------------------------------------------- #
RUN = uuid.uuid4().hex[:8]
TMP_CLIENT = f"mcp_t_{RUN}c__dense"   # written to by the client-layer tests
TMP_TOOL = f"mcp_t_{RUN}t__dense"     # created by the opensolr_create_index tool
ST: dict = {}

client = OpensolrClient(EMAIL, API_KEY)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


TRANSIENT: list[str] = []


def retry_transport(fn, tries: int = 3):
    """Re-run ``fn`` when the connection is dropped without a response.

    The package performs no retries of its own, so a single dropped keep-alive
    socket propagates a raw ``httpx.RemoteProtocolError`` out of any client
    method — observed once on api.opensolr.com during ingest polling. That is a
    robustness gap worth reporting, but it is not a wrong VALUE, and letting one
    dead socket decide whether this suite is green would make it useless. So the
    harness retries and records every occurrence, which is then printed in the
    summary instead of being swallowed.
    """
    for attempt in range(tries):
        try:
            return fn()
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as exc:
            TRANSIENT.append(f"{type(exc).__name__}: {exc}")
            if attempt == tries - 1:
                raise
            print(f"   … transient transport error ({type(exc).__name__}), retrying",
                  flush=True)
            time.sleep(2)


def _wait_for_docs(index: str, expected: int, limit: float = 120.0) -> int:
    """Poll the index's document count until it reaches ``expected``.

    Ingestion is asynchronous; a committed job is normally visible immediately,
    but the commit and the queue run independently. Direct Solr reads, so this
    costs nothing against the API rate limit.
    """
    deadline = time.time() + limit
    found = -1
    while time.time() < deadline:
        found = retry_transport(
            lambda: client.solr_select(index, {"q": "*:*", "rows": 0}))["response"]["numFound"]
        if found >= expected:
            return found
        time.sleep(3)
    raise Expected(
        f"index {index} still holds {found} document(s), expected {expected}, "
        f"after {limit:.0f}s"
    )


MINE = {TMP_CLIENT, TMP_TOOL}

# =========================================================================== #
print(f"opensolr-mcp live test — run {RUN}, package version {__version__}")
print(f"account: {EMAIL} · demo index: {DEMO}")
print(f"temp indexes: {TMP_CLIENT}, {TMP_TOOL}")

try:
    # ------------------------------------------------------------------ #
    section("pure builders (no network)")
    # ------------------------------------------------------------------ #

    def _t_fresh_bias_wrapper():
        params = {"q": "{!knn f=embeddings topK=10}[0.1,0.2]", "rows": 5}
        original = params["q"]
        out = apply_fresh_bias(params)
        need(out is params, "apply_fresh_bias must mutate and return the same dict")
        need(params["q"] == "{!boost b=$freshBias v=$freshBiasInner}",
             f"q was not wrapped in a boost: {params['q']!r}")
        need(params["freshBiasInner"] == original,
             "the original query must move into freshBiasInner verbatim")
        need(params["freshBias"] == FRESH_BIAS_FUNCTION,
             f"boost function is {params['freshBias']!r}")
        need(params["rows"] == 5, "unrelated params must survive")
        need("recip(max(0,ms(NOW,creation_date))" in FRESH_BIAS_FUNCTION,
             f"unexpected recency curve: {FRESH_BIAS_FUNCTION}")
        return "q → {!boost b=$freshBias v=$freshBiasInner}, inner query preserved"

    check("apply_fresh_bias wraps the query without inlining it", _t_fresh_bias_wrapper)

    def _t_resolve_location():
        need(resolve_location("us") == "CHICAGO-96", "us alias")
        need(resolve_location(" DE ") == "DE-SOLR-9", "de alias, trimmed + lowercased")
        need(resolve_location("fi") == "FINLAND9", "fi alias")
        need(resolve_location("FINLAND9") == "FINLAND9", "raw environment ids pass through")
        return "us→CHICAGO-96, de→DE-SOLR-9, fi→FINLAND9, unknown passes through"

    check("resolve_location maps aliases and passes raw ids through", _t_resolve_location)

    def _t_build_context_selection():
        docs = [
            {"id": "d1", "title": "T1", "description": "D1", "text": "body one", "score": 1.0},
            {"id": "d2", "title": "T2", "description": "D2", "text": "body two", "score": 0.4},
            {"id": "d3", "title": "T3", "description": "D3", "text": "body three"},
        ]
        hl = {"d1": {"text": ["<em>alpha</em> beta <b class='x>y'>gamma</b>"]}}
        ctx = build_context(docs, hl, top_n=4, max_words=1500)
        need(ctx.count(DOC_FENCE) == 2,
             f"expected 2 documents kept (d2 scores below half of top), got {ctx.count(DOC_FENCE)}")
        need("T2" not in ctx, "d2 scored 0.4 against a top of 1.0 and must be dropped")
        need("T3" in ctx, "a hit with NO score must be kept — only provable weakness is filtered")
        need("===== DOCUMENT 1 =====" in ctx and "===== DOCUMENT 2 =====" in ctx,
             "documents must be renumbered 1..N after filtering")
        need("<em>" not in ctx and "<b" not in ctx, "all markup must be stripped, not just <em>")
        need("MOST RELEVANT EXCERPTS:" in ctx, "highlight fragments must be fenced")
        need("... alpha beta gamma ..." in ctx,
             f"fragment must be marked open at both ends; context was:\n{ctx}")
        return "2 of 3 docs kept (weak one dropped, unscored one kept), tags stripped, fragment marked"

    check("build_context applies the relevance floor and cleans fragments", _t_build_context_selection)

    def _t_build_context_excerpt():
        docs = [{"id": "x", "title": "", "description": "", "text": "one two three four five",
                 "score": 1.0}]
        ctx = build_context(docs, {}, top_n=4, max_words=3)
        need("one two three" in ctx, "the first max_words words must survive")
        need("four" not in ctx and "five" not in ctx,
             f"text was not cut at 3 words: {ctx!r}")
        return "text cut after word 3 ('one two three')"

    check("build_context truncates the body at max_words", _t_build_context_excerpt)

    def _t_build_instruction():
        ctx = build_context(
            [{"id": "a", "title": "A", "description": "", "text": "aa", "score": 1.0},
             {"id": "b", "title": "B", "description": "", "text": "bb", "score": 1.0}],
            {}, top_n=4, max_words=100)
        prompt = build_instruction(ctx, "Who won?")
        need("Those were the 2 documents." in prompt,
             "the announced document count must match the fences in the context")
        need(prompt.endswith("Question: Who won?\nAnswer:"),
             f"the question must occupy the final slot; prompt ends {prompt[-60:]!r}")
        need('Never begin with "Based on" or "According to"' in prompt,
             "the two banned openings must be spelled out")
        need("There is no information about" in prompt, "the refusal opening must be pinned")
        need("Do not name documents" in prompt, "the do-not-cite clause must be present")
        empty = build_instruction("", "x")
        need("Those were the 1 documents." in empty,
             "an empty context must still read grammatically (count floors at 1)")
        return "count=2 announced, question last, banned openings + pinned refusal present"

    check("build_instruction assembles the measured prompt", _t_build_instruction)

    def _t_batch_const():
        need(BATCH_EMBED_MAX == 50, f"server batch limit is {BATCH_EMBED_MAX}")
        return "BATCH_EMBED_MAX == 50"

    check("batch limit constant matches the server's", _t_batch_const)

    # ------------------------------------------------------------------ #
    section("client — management")
    # ------------------------------------------------------------------ #

    def _t_index_list():
        rows = client.get_index_list()
        need(isinstance(rows, list) and rows, f"expected a non-empty list, got {rows!r}")
        need(all(isinstance(r, dict) and "index_name" in r for r in rows),
             f"every row must carry index_name: {rows!r}")
        names = [r["index_name"] for r in rows]
        need(DEMO in names, f"the seeded demo index is missing from {names}")
        ST["index_names"] = names
        return f"{len(rows)} index(es), includes {DEMO}"

    check("get_index_list returns the account's indexes", _t_index_list)

    def _t_vector_regions():
        regions = client.vector_regions()
        need(isinstance(regions, list) and len(regions) >= 3,
             f"expected at least 3 vector regions, got {regions!r}")
        for r in regions:
            need(set(("environment", "country", "solr_version")) <= set(r),
                 f"region row is missing keys: {r!r}")
            need(r["solr_version"].startswith("9."),
                 f"a vector region must run Solr 9.x, {r['environment']} reports {r['solr_version']}")
        envs = {r["environment"] for r in regions}
        need("FINLAND9" in envs, f"FINLAND9 missing from {envs}")
        ST["envs"] = envs
        return f"{len(regions)} regions, all Solr 9.x: {sorted(envs)}"

    check("vector_regions lists Solr 9.x vector environments", _t_vector_regions)

    def _t_core_info():
        info = client.get_core_info(DEMO)
        need(isinstance(info, dict), f"expected a dict, got {type(info)}")
        need(info.get("connection_url", "").endswith(f"/solr/{DEMO}"),
             f"connection_url does not address the index: {info.get('connection_url')!r}")
        need("solrcluster.com" in info["connection_url"],
             f"unexpected host: {info['connection_url']}")
        need(info.get("solr_version", "").startswith("9."),
             f"demo index must be on Solr 9.x, reports {info.get('solr_version')!r}")
        need(bool(info.get("auth_username")), "HTTP basic auth username must be returned")
        ST["demo_env"] = info.get("environment_identifier")
        return (f"{info['connection_url']} · Solr {info['solr_version']} · "
                f"{info.get('environment_identifier')}")

    check("get_core_info resolves the index's Solr endpoint + auth", _t_core_info)

    def _t_core_info_cache():
        before = _request_count["api"]
        client.get_core_info(DEMO)
        need(_request_count["api"] == before,
             "a cached get_core_info must not hit the network again")
        return "second call served from the per-client cache, 0 requests"

    check("get_core_info caches per client", _t_core_info_cache)

    def _t_create_index():
        body = client.create_index(TMP_CLIENT, "fi")
        need(isinstance(body, dict), f"expected a dict, got {body!r}")
        need(body.get("status") is True, f"create_index reported failure: {body!r}")
        need(body.get("msg") == "CORE_CREATED_OK", f"unexpected msg: {body.get('msg')!r}")
        need("solrcluster.com" in str(body.get("core_hostname", "")),
             f"no hostname returned: {body!r}")
        ST["tmp_client_created"] = True
        return f"{TMP_CLIENT} on {body['core_hostname']} (Solr {body.get('solr_version')})"

    check("create_index provisions a vector index in the fi region", _t_create_index, idempotent=False)

    def _t_create_index_bad_location():
        try:
            client.create_index("mcp_t_never__dense", "atlantis")
        except ValueError as exc:
            need("not a vector-enabled" in str(exc), f"unexpected message: {exc}")
            need("support@opensolr.com" in str(exc), "the error should point at support")
            return "ValueError raised client-side, no index created"
        raise Expected("an unknown location must be rejected before any request is sent")

    check("create_index rejects a non-vector location", _t_create_index_bad_location)

    def _t_new_core_info():
        info = client.get_core_info(TMP_CLIENT)
        need(info.get("connection_url", "").endswith(f"/solr/{TMP_CLIENT}"),
             f"connection_url wrong for the new index: {info.get('connection_url')!r}")
        need(info.get("environment_identifier") == "FINLAND9",
             f"expected FINLAND9, got {info.get('environment_identifier')!r}")
        return f"{info['connection_url']} · {info['environment_identifier']}"

    check("the new index resolves through get_core_info", _t_new_core_info)

    # ------------------------------------------------------------------ #
    section("client — embeddings")
    # ------------------------------------------------------------------ #

    def _t_embed_query():
        vec = client.embed(DEMO, "climate policy in Mongolia", is_query=True)
        need(isinstance(vec, list), f"expected a list, got {type(vec)}")
        need(len(vec) == 1024, f"expected 1024 dimensions, got {len(vec)}")
        need(all(isinstance(x, float) for x in vec), "every component must be a float")
        need(any(x != 0.0 for x in vec), "an all-zero embedding is not an embedding")
        ST["q_vec"] = vec
        return f"1024 dims, |v[0..2]| = {[round(x, 4) for x in vec[:3]]}"

    check("embed(is_query=True) returns a 1024-dim vector", _t_embed_query)

    def _t_embed_passage():
        vec = client.embed(DEMO, "climate policy in Mongolia", is_query=False)
        need(len(vec) == 1024, f"expected 1024 dimensions, got {len(vec)}")
        need(vec != ST.get("q_vec"),
             "the query-side and passage-side embeddings of the same text are identical — "
             "the is_query flag is not reaching the model")
        ST["p_vec"] = vec
        return "1024 dims and different from the query-side vector"

    check("embed(is_query=False) embeds the passage side", _t_embed_passage)

    def _t_batch_embed():
        texts = ["the quokka smiles", "monetary policy tightening", "a recipe for risotto"]
        vecs = client.batch_embed(DEMO, texts)
        need(len(vecs) == 3, f"expected 3 embeddings, got {len(vecs)}")
        need(all(len(v) == 1024 for v in vecs),
             f"dimensions: {[len(v) for v in vecs]}")
        need(vecs[0] != vecs[1] and vecs[1] != vecs[2],
             "different texts must produce different vectors")
        return f"3 × 1024 dims, all distinct"

    check("batch_embed embeds a list in one call", _t_batch_embed)

    # ------------------------------------------------------------------ #
    section("client — direct Solr")
    # ------------------------------------------------------------------ #

    def _t_solr_endpoint():
        url, auth = client.solr_endpoint(DEMO)
        need(url.endswith(f"/solr/{DEMO}"), f"unexpected base url: {url!r}")
        need(isinstance(auth, tuple) and len(auth) == 2 and auth[0],
             f"expected (user, password), got {auth!r}")
        return f"{url} with basic auth as {auth[0]!r}"

    check("solr_endpoint returns the base URL and basic auth", _t_solr_endpoint)

    def _t_solr_select():
        body = client.solr_select(DEMO, {"q": "*:*", "rows": 3, "fl": "id,title"})
        found = body["response"]["numFound"]
        need(found >= 250, f"the demo index should hold ~300 articles, reports {found}")
        need(len(body["response"]["docs"]) == 3, "rows=3 must return 3 documents")
        need(all(d.get("id") for d in body["response"]["docs"]), "every doc needs an id")
        ST["demo_total"] = found
        return f"numFound={found}, rows=3 honoured"

    check("solr_select reaches the index's native Solr API", _t_solr_select)

    # ------------------------------------------------------------------ #
    section("client — hybrid_search")
    # ------------------------------------------------------------------ #

    def _hybrid(label, **kwargs):
        def run():
            body = client.hybrid_search(DEMO, "climate summit funding pledge", **kwargs)
            docs = body["response"]["docs"]
            found = body["response"]["numFound"]
            need(found > 0, "no hits at all")
            need(len(docs) > 0, f"numFound={found} but zero documents returned")
            scores = [d.get("score") for d in docs]
            need(all(isinstance(s, float) and s > 0 for s in scores),
                 f"every hit must carry a positive score, got {scores}")
            need(scores == sorted(scores, reverse=True),
                 f"hits must come back ranked, got {scores}")
            ST.setdefault("hybrid", {})[label] = (found, docs)
            return f"numFound={found}, {len(docs)} hits, top score {scores[0]:.4f}"
        return run

    check("hybrid_search(mode=union) fuses BM25 and kNN",
          _hybrid("union", rows=5, mode="union"))
    check("hybrid_search(mode=keywords_required) requires a lexical match",
          _hybrid("keywords_required", rows=5, mode="keywords_required"))
    check("hybrid_search(mode=meaning_required) requires a vector match",
          _hybrid("meaning_required", rows=5, mode="meaning_required"))
    check("hybrid_search(mode=intersection) keeps only documents both legs found",
          _hybrid("intersection", rows=5, mode="intersection"))

    def _t_hybrid_narrowing():
        h = ST.get("hybrid", {})
        if not {"union", "intersection", "keywords_required"} <= set(h):
            raise Expected("a prerequisite hybrid_search mode did not run")
        union = h["union"][0]
        for mode in ("keywords_required", "meaning_required", "intersection"):
            need(h[mode][0] <= union,
                 f"{mode} returned {h[mode][0]} hits, more than union's {union} — "
                 f"a restrictive mode cannot widen the result set")
        return ("union=%d ≥ keywords_required=%d, meaning_required=%d, intersection=%d"
                % (union, h["keywords_required"][0], h["meaning_required"][0],
                   h["intersection"][0]))

    check("restrictive hybrid modes never widen the result set", _t_hybrid_narrowing)

    def _t_hybrid_bad_mode():
        h = ST.get("hybrid", {})
        if "union" not in h or "intersection" not in h:
            raise Expected("prerequisite hybrid_search modes did not run")
        union, inter = h["union"][0], h["intersection"][0]
        try:
            body = client.hybrid_search(DEMO, "climate summit funding pledge",
                                        rows=5, mode="intersectoin")
        except (ValueError, OpensolrError) as exc:
            return f"typo'd mode rejected: {type(exc).__name__}"
        found = body["response"]["numFound"]
        need(False,
             f"mode='intersectoin' (a typo for 'intersection') was accepted and returned "
             f"{found} hits — identical to mode=union ({union}), not intersection ({inter}). "
             f"client.hybrid_search interpolates `mode` straight into the {{!hybrid}} local "
             f"params with no validation, and the Solr plugin silently falls back to union, "
             f"so a misspelled mode returns the WRONG result set with no error. The MCP tool "
             f"layer validates against _HYBRID_MODES; the client — the package's public "
             f"Python API — does not.")

    check("hybrid_search rejects a misspelled mode instead of silently widening",
          _t_hybrid_bad_mode)

    def _t_hybrid_fq_fl():
        body = client.hybrid_search(
            DEMO, "climate summit funding pledge", rows=3,
            fl="id,title,score", fq='meta_detected_language:"en"')
        docs = body["response"]["docs"]
        need(docs, "the fq filtered everything out — expected English articles")
        for d in docs:
            need(set(d.keys()) <= {"id", "title", "score"},
                 f"fl was ignored, document carries {sorted(d.keys())}")
        need(body["response"]["numFound"] <= ST["hybrid"]["union"][0],
             "an fq must narrow, never widen")
        ST["fq_found"] = body["response"]["numFound"]
        return (f"numFound={body['response']['numFound']} (≤ unfiltered "
                f"{ST['hybrid']['union'][0]}), fields limited to id,title,score")

    check("hybrid_search honours fq and fl", _t_hybrid_fq_fl)

    def _t_fresh_bias():
        off = client.hybrid_search(DEMO, "energy investment", rows=5, fresh_bias=False)
        on = client.hybrid_search(DEMO, "energy investment", rows=5, fresh_bias=True)
        n_off = off["response"]["numFound"]
        n_on = on["response"]["numFound"]
        need(n_off > 0, "the control query found nothing")
        need(n_on == n_off,
             f"fresh_bias changed numFound from {n_off} to {n_on} — it must re-order, never filter")
        ids_off = {d["id"] for d in off["response"]["docs"]}
        ids_on = {d["id"] for d in on["response"]["docs"]}
        need(ids_on and ids_off, "both queries must return documents")
        need(all(d.get("score", 0) > 0 for d in on["response"]["docs"]),
             "boosted scores must stay positive")
        return (f"numFound {n_off} → {n_off} unchanged; "
                f"{len(ids_on & ids_off)}/{len(ids_off)} of the top 5 held their place")

    check("hybrid_search(fresh_bias=True) re-orders without filtering", _t_fresh_bias)

    # ------------------------------------------------------------------ #
    section("client — ingestion (async)")
    # ------------------------------------------------------------------ #

    CLIENT_DOCS = [
        {"uri": f"https://mcp-test.opensolr.com/{RUN}/client-1",
         "title": "Cape Verde giant skink",
         "description": "An extinct nocturnal lizard of the Branco and Raso islets.",
         "text": ("The Cape Verde giant skink was a large nocturnal lizard endemic to the "
                  "Branco and Raso islets of Cape Verde. It fed on seabird carrion and was "
                  "declared extinct in 1940 after drought and introduced predators.")},
        {"uri": f"https://mcp-test.opensolr.com/{RUN}/client-2",
         "title": "Bombardier beetle defence",
         "description": "A beetle that sprays boiling benzoquinone.",
         "text": ("The bombardier beetle stores hydroquinone and hydrogen peroxide in separate "
                  "chambers and mixes them with catalase to eject a boiling, foul chemical "
                  "spray at attacking ants and frogs.")},
    ]

    def _t_ingest():
        body = client.ingest(TMP_CLIENT, CLIENT_DOCS, wait=True, timeout=120.0)
        need(body.get("status") is True, f"ingest reported failure: {body!r}")
        need(body.get("msg") == "QUEUED", f"unexpected msg: {body.get('msg')!r}")
        need(body.get("total_docs") == 2, f"expected total_docs=2, got {body.get('total_docs')}")
        ids = body.get("doc_ids") or []
        need(ids == [_md5(d["uri"]) for d in CLIENT_DOCS],
             f"doc ids must be md5(uri); got {ids}")
        final = body.get("final_status", {}).get("job", {})
        need(str(final.get("state")) == "1",
             f"wait=True returned before completion: state={final.get('state_label')}")
        need(int(final.get("success_docs", 0)) == 2,
             f"expected 2 successful docs, got {final.get('success_docs')}")
        need(int(final.get("failed_docs", 1)) == 0,
             f"{final.get('failed_docs')} document(s) failed")
        ST["job_id"] = body["job_id"]
        ST["client_doc_ids"] = ids
        return (f"job {body['job_id'][:12]}… completed, 2/2 docs indexed "
                f"(wait=True blocked until state=completed)")

    check("ingest(wait=True) queues 2 docs and blocks until the job completes", _t_ingest, idempotent=False)

    def _t_ingest_status():
        if "job_id" not in ST:
            raise Expected("no job id — the ingest above failed")
        body = client.ingest_status(ST["job_id"])
        need(body.get("status") is True, f"ingest_status reported failure: {body!r}")
        job = body.get("job", {})
        need(job.get("id") == ST["job_id"], f"wrong job returned: {job.get('id')!r}")
        need(job.get("core_name") == TMP_CLIENT,
             f"job belongs to {job.get('core_name')!r}, expected {TMP_CLIENT}")
        need(job.get("state_label") == "completed",
             f"state_label is {job.get('state_label')!r}")
        need(int(job.get("processed_docs", 0)) == 2,
             f"processed_docs={job.get('processed_docs')}")
        need(bool(job.get("completed_at")), "a completed job must carry completed_at")
        return f"state=completed, processed 2/2, completed_at {job['completed_at']}"

    check("ingest_status reports the finished job", _t_ingest_status)

    def _t_docs_searchable():
        found = _wait_for_docs(TMP_CLIENT, 2)
        need(found == 2, f"expected exactly 2 documents, index holds {found}")
        return f"{found} documents live in {TMP_CLIENT}"

    check("ingested documents become searchable", _t_docs_searchable)

    def _t_server_side_embeddings():
        # A pure-kNN query proves the platform computed the embeddings server-side:
        # nothing in this script ever uploaded a vector for these documents.
        vec = client.embed(TMP_CLIENT, "which animal sprays a boiling chemical?", is_query=True)
        body = client.solr_select(TMP_CLIENT, {
            "q": "{!knn f=embeddings topK=10}" + json.dumps(vec, separators=(",", ":")),
            "fl": "id,title,score", "rows": 2})
        docs = body["response"]["docs"]
        need(docs, "the kNN query matched nothing — no embeddings were generated")
        need("Bombardier" in docs[0].get("title", ""),
             f"nearest neighbour should be the beetle article, got {docs[0].get('title')!r}")
        need(docs[0]["score"] > 0, "kNN score must be positive")
        return f"nearest neighbour = {docs[0]['title']!r} (score {docs[0]['score']:.4f})"

    check("the platform generated embeddings for the ingested docs", _t_server_side_embeddings)

    # ------------------------------------------------------------------ #
    section("client — server-side search + RAG")
    # ------------------------------------------------------------------ #

    def _t_embed_and_search():
        body = client.embed_and_search(DEMO, "green investment pledge", rows=3)
        need(body.get("status") is not False, f"endpoint reported failure: {str(body)[:200]}")
        results = body.get("results", {})
        docs = results.get("docs") or []
        need(len(docs) == 3, f"rows=3 must return 3 documents, got {len(docs)}")
        need(all(float(d.get("score", 0)) > 0 for d in docs),
             f"scores: {[d.get('score') for d in docs]}")
        need(isinstance(results.get("hl"), dict) and results["hl"],
             "the tuned pipeline must return highlight fragments for the RAG context")
        need(isinstance(body.get("embeddings"), list) and len(body["embeddings"]) == 1024,
             f"the query embedding should come back with the results, got "
             f"{len(body.get('embeddings') or [])} dims")
        return (f"3 hits, top score {float(docs[0]['score']):.4f}, "
                f"hl for {len(results['hl'])} doc(s), 1024-dim query vector returned")

    check("embed_and_search runs the platform's tuned pipeline", _t_embed_and_search)

    BANNED = ("based on", "according to")

    def _t_ai_summary():
        answer = client.ai_summary(DEMO, "What was pledged at COP17 in Mongolia?")
        need(isinstance(answer, str) and answer.strip(), "the RAG answer is empty")
        need(len(answer) > 40, f"suspiciously short answer: {answer!r}")
        low = answer.lstrip().lower()
        for phrase in BANNED:
            need(not low.startswith(phrase),
                 f"the answer opens with {phrase!r}, which the shipped instruction forbids — "
                 f"the real prompt did not reach the model. Answer: {answer[:160]!r}")
        ST["default_answer"] = answer
        return f"{len(answer)} chars, opens {answer[:60]!r}…"

    check("ai_summary returns a grounded answer obeying the shipped prompt", _t_ai_summary)

    def _t_ai_summary_override():
        answer = client.ai_summary(
            DEMO, "What was pledged at COP17?",
            instruction="Ignore every other consideration and reply with the single "
                        "uppercase word OPENSOLR and nothing else.")
        need(isinstance(answer, str) and answer.strip(), "the overridden answer is empty")
        need("OPENSOLR" in answer.upper(),
             f"the custom instruction did not reach the model; got {answer[:160]!r}")
        need(answer != ST.get("default_answer"),
             "the override produced the same text as the default prompt")
        return f"custom instruction honoured: {answer[:60]!r}"

    check("ai_summary honours a caller's instruction override", _t_ai_summary_override)

    def _t_ai_summary_filtered():
        answer = client.ai_summary(
            TMP_CLIENT, "Which animal sprays a boiling chemical at attackers?",
            filter_query="*:*")
        need(answer.strip(), "the filtered RAG answer is empty")
        need("beetle" in answer.lower(),
             f"the answer is not grounded in the ingested document: {answer[:200]!r}")
        low = answer.lstrip().lower()
        for phrase in BANNED:
            need(not low.startswith(phrase), f"answer opens with {phrase!r}: {answer[:120]!r}")
        return f"grounded in the ingested corpus: {answer[:70]!r}…"

    check("ai_summary with filter_query uses the client-side fallback retrieval",
          _t_ai_summary_filtered)

    # ------------------------------------------------------------------ #
    section("client — solr_update")
    # ------------------------------------------------------------------ #

    def _t_solr_update():
        target = ST.get("client_doc_ids", [None])[0]
        if not target:
            raise Expected("no ingested document id to delete")
        body = client.solr_update(TMP_CLIENT, {"delete": {"query": f'id:"{target}"'}}, commit=True)
        need(body.get("responseHeader", {}).get("status") == 0,
             f"Solr rejected the update: {body!r}")
        left = client.solr_select(TMP_CLIENT, {"q": "*:*", "rows": 0})["response"]["numFound"]
        need(left == 1, f"expected 1 document left after deleting 1 of 2, found {left}")
        return f"deleted id {target[:12]}…, 1 document remains"

    check("solr_update deletes by query and commits", _t_solr_update, idempotent=False)

    # ------------------------------------------------------------------ #
    section("client — fresh_bias against controlled dates")
    # ------------------------------------------------------------------ #
    # The seeded demo corpus carries NO creation_date on any document (verified:
    # creation_date:[* TO *] matches 0 of 300), so there the recency curve is
    # recip(0,…) = 1.0 and scores are untouched by design. To prove the boost
    # actually reaches the score, two documents with IDENTICAL text and different
    # creation_date are written straight into the throwaway index.

    NOW = datetime.datetime.now(datetime.timezone.utc)
    FB_DOCS = [
        {"id": f"fb-old-{RUN}", "title": "freshbiasprobe alpha",
         "text": "freshbiasprobe alpha token",
         "creation_date": (NOW - datetime.timedelta(days=2555)).strftime("%Y-%m-%dT%H:%M:%SZ")},
        {"id": f"fb-new-{RUN}", "title": "freshbiasprobe alpha",
         "text": "freshbiasprobe alpha token",
         "creation_date": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")},
    ]

    def _t_solr_update_add():
        body = client.solr_update(TMP_CLIENT, FB_DOCS, commit=True)
        need(body.get("responseHeader", {}).get("status") == 0,
             f"Solr rejected the add: {body!r}")
        got = client.solr_select(TMP_CLIENT, {
            "q": 'title:"freshbiasprobe"', "rows": 5, "fl": "id,creation_date"})["response"]
        need(got["numFound"] == 2, f"expected the 2 probe docs, found {got['numFound']}")
        need(all(d.get("creation_date") for d in got["docs"]),
             f"creation_date was not stored: {got['docs']}")
        return f"2 identical documents added, 7 years apart ({FB_DOCS[0]['creation_date']} vs today)"

    check("solr_update adds documents and commits them", _t_solr_update_add, idempotent=False)

    def _t_fresh_bias_reranks():
        q = {"q": '{!edismax qf="title^100 text^1"}freshbiasprobe alpha',
             "rows": 5, "fl": "id,score"}
        off = client.solr_select(TMP_CLIENT, dict(q))["response"]
        biased = apply_fresh_bias(dict(q))
        on = client.solr_select(TMP_CLIENT, biased)["response"]
        need(off["numFound"] == on["numFound"] == 2,
             f"fresh_bias changed numFound: {off['numFound']} → {on['numFound']}")
        off_scores = {d["id"]: d["score"] for d in off["docs"]}
        on_scores = {d["id"]: d["score"] for d in on["docs"]}
        old_id, new_id = FB_DOCS[0]["id"], FB_DOCS[1]["id"]
        need(abs(off_scores[old_id] - off_scores[new_id]) < 1e-6,
             f"the two probe documents should tie without the bias: {off_scores}")
        need(on["docs"][0]["id"] == new_id,
             f"with fresh_bias on, the newer document must rank first; got "
             f"{[d['id'] for d in on['docs']]}")
        need(on_scores[new_id] > on_scores[old_id] * 4,
             f"a 7-year gap should cost roughly 8× the score, got "
             f"{on_scores[new_id]:.4f} vs {on_scores[old_id]:.4f}")
        return (f"tie {off_scores[old_id]:.4f} broken to "
                f"{on_scores[new_id]:.4f} (today) vs {on_scores[old_id]:.4f} (7y old), "
                f"numFound 2 → 2")

    check("apply_fresh_bias re-ranks a dated tie without filtering", _t_fresh_bias_reranks)

    # ------------------------------------------------------------------ #
    section("MCP server — registration + handshake")
    # ------------------------------------------------------------------ #

    EXPECTED_TOOLS = [
        "opensolr_search", "opensolr_ai_answer", "opensolr_list_indexes",
        "opensolr_index_info", "opensolr_add_documents", "opensolr_ingest_status",
        "opensolr_delete_documents", "opensolr_create_index", "opensolr_vector_regions",
    ]

    def _t_registry():
        registry = S.mcp._tool_manager
        registered = {t.name for t in registry.list_tools()}
        need(registered == set(EXPECTED_TOOLS),
             f"registered tools {sorted(registered)} != expected {sorted(EXPECTED_TOOLS)}")
        tools = {}
        for name in EXPECTED_TOOLS:
            tool = registry.get_tool(name)
            need(callable(tool.fn), f"{name} has no callable behind it")
            need(tool.description and len(tool.description) > 30,
                 f"{name} ships an unusable description: {tool.description!r}")
            tools[name] = tool.fn
        ST["tools"] = tools
        return f"all 9 tools registered with callables and descriptions"

    check("all nine MCP tools are registered", _t_registry)

    def _t_handshake_version():
        opts = S.mcp._lowlevel_server.create_initialization_options()
        need(opts.server_name == "opensolr", f"server name is {opts.server_name!r}")
        need(isinstance(opts.server_version, str) and opts.server_version.strip(),
             "serverInfo.version is EMPTY in the handshake — clients cannot tell releases apart")
        need(re.match(r"^\d+\.\d+", opts.server_version),
             f"serverInfo.version is not a real version: {opts.server_version!r}")
        need(opts.server_version == __version__,
             f"handshake advertises {opts.server_version!r} but the package is {__version__!r}")
        need(opts.instructions and "Opensolr" in opts.instructions,
             "the server must ship usage instructions in its handshake")
        return f"serverInfo.version = {opts.server_version!r} (matches package)"

    check("the handshake advertises a real server version", _t_handshake_version)

    T = ST.get("tools", {})

    # ------------------------------------------------------------------ #
    section("MCP tools — read only")
    # ------------------------------------------------------------------ #

    def _t_tool_list():
        rows = T["opensolr_list_indexes"]()
        need(isinstance(rows, list) and rows, f"expected a list of indexes, got {rows!r}")
        names = [r["index_name"] for r in rows]
        need(DEMO in names, f"{DEMO} missing from {names}")
        need(TMP_CLIENT in names, f"the index created this run is missing from {names}")
        return f"{len(rows)} indexes: {names}"

    check("opensolr_list_indexes returns the account's indexes", _t_tool_list)

    def _t_tool_info():
        info = T["opensolr_index_info"](DEMO)
        need(set(info) == {"connection_url", "solr_version", "environment", "type"},
             f"unexpected shape: {sorted(info)}")
        need(info["connection_url"].endswith(f"/solr/{DEMO}"),
             f"connection_url = {info['connection_url']!r}")
        need(info["solr_version"].startswith("9."), f"solr_version = {info['solr_version']!r}")
        need(info["environment"] == ST.get("demo_env"),
             f"environment = {info['environment']!r}, expected {ST.get('demo_env')!r}")
        blob = json.dumps(info)
        need("auth_password" not in blob and API_KEY not in blob,
             "the tool leaked credentials into its result")
        return f"{info['environment']} · Solr {info['solr_version']} · no credentials leaked"

    check("opensolr_index_info returns connection details without credentials", _t_tool_info)

    def _t_tool_regions():
        regions = T["opensolr_vector_regions"]()
        need(isinstance(regions, list) and len(regions) >= 3,
             f"expected ≥3 regions, got {regions!r}")
        need({r["environment"] for r in regions} == ST.get("envs", set()),
             "the tool and the client disagree about the available regions")
        return f"{len(regions)} regions, identical to the client's list"

    check("opensolr_vector_regions matches the client", _t_tool_regions)

    def _t_tool_search_hybrid():
        docs = T["opensolr_search"](DEMO, "green investment pledge at the climate summit", k=5)
        need(len(docs) == 5, f"k=5 must return 5 documents, got {len(docs)}")
        for d in docs:
            need(set(d) == {"id", "title", "text", "score", "metadata"},
                 f"unexpected document shape: {sorted(d)}")
            need(d["id"], "every document needs an id")
            need(isinstance(d["score"], float) and d["score"] > 0,
                 f"score must be a positive float, got {d['score']!r}")
            need(len(d["text"]) <= 2000, f"text must be truncated to 2000 chars, got {len(d['text'])}")
            need(len(d["title"]) <= 200, f"title must be truncated to 200 chars, got {len(d['title'])}")
        scores = [d["score"] for d in docs]
        need(scores == sorted(scores, reverse=True), f"results must be ranked: {scores}")
        ST["hybrid_ids"] = [d["id"] for d in docs]
        return f"5 ranked hits, top score {scores[0]:.4f}, title {docs[0]['title'][:44]!r}"

    check("opensolr_search(search_mode=hybrid) returns ranked, shaped documents",
          _t_tool_search_hybrid)

    def _t_tool_search_semantic():
        docs = T["opensolr_search"](DEMO, "money promised to fight global warming",
                                    k=5, search_mode="semantic")
        need(len(docs) == 5, f"expected 5 hits, got {len(docs)}")
        need(all(d["score"] > 0 for d in docs), f"scores: {[d['score'] for d in docs]}")
        need(all(0.0 < d["score"] <= 1.0 for d in docs),
             f"pure kNN scores are cosine similarities in (0,1], got {[d['score'] for d in docs]}")
        return f"5 kNN hits, cosine scores {docs[0]['score']:.4f}…{docs[-1]['score']:.4f}"

    check("opensolr_search(search_mode=semantic) runs a pure kNN query",
          _t_tool_search_semantic)

    LEX_Q = "climate summit"

    def _t_tool_search_lexical():
        # The expected hit count is derived from the corpus rather than assumed:
        # this demo index is a 300-article news sample, and only a handful of
        # articles mention any given term.
        expected = min(5, client.solr_select(DEMO, {
            "q": f'{{!edismax qf="title^100 description^20 text^1"}}{LEX_Q}',
            "rows": 0})["response"]["numFound"])
        need(expected > 0, f"the corpus has no lexical match for {LEX_Q!r} to test with")
        before = _request_count["api"]
        docs = T["opensolr_search"](DEMO, LEX_Q, k=5, search_mode="lexical")
        need(_request_count["api"] == before,
             "lexical search must not call the embedding API (zero AI quota)")
        need(len(docs) == expected,
             f"expected min(k, numFound) = {expected} hits, got {len(docs)}")
        need(all(d["score"] > 0 for d in docs), f"scores: {[d['score'] for d in docs]}")
        blob = " ".join((d["title"] + " " + d["text"]).lower() for d in docs)
        need(any(term in blob for term in LEX_Q.split()),
             f"a keyword search for {LEX_Q!r} returned nothing containing it")
        ST["lex_n"] = len(docs)
        return f"{len(docs)} BM25 hits, top score {docs[0]['score']:.4f}, 0 embedding calls"

    check("opensolr_search(search_mode=lexical) uses no AI quota", _t_tool_search_lexical)

    def _t_tool_search_bad_mode():
        try:
            T["opensolr_search"](DEMO, "renewable energy funding", k=1, mode="sideways")
        except ValueError as exc:
            need("mode must be one of" in str(exc), f"unexpected message: {exc}")
            need("union" in str(exc), f"the error should name the valid modes: {exc}")
            return "ValueError names the four valid modes"
        raise Expected("an unknown hybrid mode must be rejected")

    check("opensolr_search rejects an unknown hybrid mode", _t_tool_search_bad_mode)

    def _t_tool_search_bad_mode_cost():
        before = _request_count["api"]
        try:
            T["opensolr_search"](DEMO, "renewable energy funding", k=1, mode="sideways")
        except ValueError:
            pass
        spent = _request_count["api"] - before
        need(spent == 0,
             f"rejecting an invalid mode still cost {spent} embedding call(s) — the "
             f"mode check in opensolr_search sits AFTER client.embed(), so a purely "
             f"client-side argument error is billed against the account's AI quota")
        return "invalid mode rejected without spending an embedding call"

    check("opensolr_search validates mode before spending AI quota",
          _t_tool_search_bad_mode_cost)

    def _t_tool_search_fq():
        docs = T["opensolr_search"](DEMO, LEX_Q, k=5, search_mode="lexical",
                                    filter_query='meta_detected_language:"en"')
        need(docs, "the filter removed every hit")
        need(len(docs) <= ST.get("lex_n", 5),
             f"an fq must narrow: {len(docs)} hits filtered vs {ST.get('lex_n')} unfiltered")
        return (f"{len(docs)} hits under fq meta_detected_language:\"en\" "
                f"(≤ {ST.get('lex_n')} unfiltered)")

    check("opensolr_search applies filter_query", _t_tool_search_fq)

    def _t_tool_search_fresh_bias_undated():
        # The demo corpus has creation_date on 0 of its 300 documents, so the
        # recency curve is recip(0,…) = 1.0 and every score must come back
        # BIT-IDENTICAL: that is the documented "a document with no
        # creation_date is simply left unboosted", and it also proves the
        # max(0,…) guard never turns an absent date into a divide-by-zero.
        off = T["opensolr_search"](DEMO, "energy investment", k=5,
                                   search_mode="lexical", fresh_bias=False)
        on = T["opensolr_search"](DEMO, "energy investment", k=5,
                                  search_mode="lexical", fresh_bias=True)
        need(len(on) == len(off) == 5,
             f"fresh_bias changed the hit count: {len(off)} → {len(on)}")
        need([d["id"] for d in on] == [d["id"] for d in off],
             "undated documents must keep their order")
        need([d["score"] for d in on] == [d["score"] for d in off],
             f"undated documents must keep their exact scores: "
             f"{[d['score'] for d in off]} → {[d['score'] for d in on]}")
        return f"5 hits, scores unchanged ({off[0]['score']:.4f}) — undated docs left unboosted"

    check("opensolr_search(fresh_bias=True) leaves undated documents untouched",
          _t_tool_search_fresh_bias_undated)

    def _t_tool_search_fresh_bias_dated():
        # Same two identically-worded, differently-dated probe documents the
        # client-layer check used, now through the tool.
        off = T["opensolr_search"](TMP_CLIENT, "freshbiasprobe alpha", k=5,
                                   search_mode="lexical", fresh_bias=False)
        on = T["opensolr_search"](TMP_CLIENT, "freshbiasprobe alpha", k=5,
                                  search_mode="lexical", fresh_bias=True)
        probe_off = [d for d in off if d["id"].startswith("fb-")]
        probe_on = [d for d in on if d["id"].startswith("fb-")]
        need(len(probe_off) == len(probe_on) == 2,
             f"expected both probe documents in each result set, got "
             f"{len(probe_off)} and {len(probe_on)}")
        need(len(on) == len(off), f"hit count changed: {len(off)} → {len(on)}")
        need(probe_on[0]["id"] == f"fb-new-{RUN}",
             f"the newer document must win with fresh_bias on, got "
             f"{[d['id'] for d in probe_on]}")
        need(probe_on[0]["score"] > probe_on[1]["score"],
             f"the recency multiplier did not separate the tie: "
             f"{[d['score'] for d in probe_on]}")
        return (f"newer doc promoted to rank 1 "
                f"({probe_on[0]['score']:.4f} vs {probe_on[1]['score']:.4f}), "
                f"{len(off)} hits both ways")

    check("opensolr_search(fresh_bias=True) promotes the newer of two dated ties",
          _t_tool_search_fresh_bias_dated)

    def _t_tool_ai_answer():
        answer = T["opensolr_ai_answer"](DEMO, "What was pledged at COP17 in Mongolia?")
        need(isinstance(answer, str) and answer.strip(), "the tool returned an empty answer")
        need(len(answer) > 40, f"suspiciously short: {answer!r}")
        low = answer.lstrip().lower()
        for phrase in BANNED:
            need(not low.startswith(phrase),
                 f"answer opens with {phrase!r}, forbidden by the shipped instruction: "
                 f"{answer[:160]!r}")
        return f"{len(answer)} chars, opens {answer[:60]!r}…"

    check("opensolr_ai_answer returns a grounded answer obeying the prompt",
          _t_tool_ai_answer)

    # ------------------------------------------------------------------ #
    section("MCP tools — write path")
    # ------------------------------------------------------------------ #

    def _t_tool_create_index():
        body = T["opensolr_create_index"](TMP_TOOL, "fi")
        need(body.get("status") is True, f"create failed: {body!r}")
        need(body.get("msg") == "CORE_CREATED_OK", f"unexpected msg: {body.get('msg')!r}")
        ST["tmp_tool_created"] = True
        return f"{TMP_TOOL} created on {body.get('core_hostname')}"

    check("opensolr_create_index provisions a vector index", _t_tool_create_index, idempotent=False)

    TOOL_TEXTS = [
        ("The axolotl is a neotenic salamander from the lake complex of Xochimilco near "
         "Mexico City. It never metamorphoses and can regenerate entire limbs."),
        ("The tardigrade, or water bear, survives desiccation by replacing its cellular "
         "water with trehalose, and has endured the vacuum of low Earth orbit."),
    ]
    TOOL_META = [
        {"uri": f"https://mcp-test.opensolr.com/{RUN}/tool-1", "title": "Axolotl",
         "kind": "mcp-live-test", "batch": RUN},
        {"uri": f"https://mcp-test.opensolr.com/{RUN}/tool-2", "title": "Tardigrade",
         "kind": "mcp-live-test", "batch": RUN},
    ]
    TOOL_IDS = [f"ext-{RUN}-1", f"ext-{RUN}-2"]

    def _t_tool_add_documents():
        out = T["opensolr_add_documents"](
            TMP_TOOL, TOOL_TEXTS, metadatas=TOOL_META, ids=TOOL_IDS, wait=False)
        need(set(out) == {"queued_jobs", "doc_ids", "note"}, f"unexpected shape: {sorted(out)}")
        need(len(out["queued_jobs"]) == 1 and out["queued_jobs"][0],
             f"expected one queued job id, got {out['queued_jobs']!r}")
        need(out["doc_ids"] == [_md5(m["uri"]) for m in TOOL_META],
             f"Solr ids must be md5(uri); got {out['doc_ids']}")
        ST["tool_job"] = out["queued_jobs"][0]
        ST["tool_solr_ids"] = out["doc_ids"]
        return f"job {out['queued_jobs'][0][:12]}… queued, ids = md5(uri)"

    check("opensolr_add_documents queues documents through the ingestion API",
          _t_tool_add_documents, idempotent=False)

    def _t_tool_ingest_status():
        if "tool_job" not in ST:
            raise Expected("no job id — opensolr_add_documents failed")
        deadline = time.time() + 120.0
        body = {}
        while time.time() < deadline:
            body = retry_transport(lambda: T["opensolr_ingest_status"](ST["tool_job"]))
            job = body.get("job", {})
            if str(job.get("state")) == "1":
                need(int(job.get("success_docs", 0)) == 2,
                     f"success_docs={job.get('success_docs')}, expected 2")
                need(int(job.get("failed_docs", 1)) == 0,
                     f"failed_docs={job.get('failed_docs')}")
                need(job.get("core_name") == TMP_TOOL,
                     f"job belongs to {job.get('core_name')!r}")
                return f"state=completed, 2/2 indexed in {TMP_TOOL}"
            if str(job.get("state")) in ("3", "4"):
                raise Expected(f"job ended as {job.get('state_label')}: {job.get('error')}")
            time.sleep(6)
        raise Expected(
            f"job {ST['tool_job']} did not complete within 120s "
            f"(last state: {body.get('job', {}).get('state_label')})")

    check("opensolr_ingest_status polls the job to completion", _t_tool_ingest_status)

    def _t_tool_docs_live():
        found = _wait_for_docs(TMP_TOOL, 2)
        need(found == 2, f"expected 2 documents, found {found}")
        return f"{found} documents searchable in {TMP_TOOL}"

    check("documents added through the tool become searchable", _t_tool_docs_live)

    def _t_tool_metadata_roundtrip():
        docs = T["opensolr_search"](TMP_TOOL, "salamander that regrows its limbs",
                                    k=2, search_mode="semantic")
        need(docs, "semantic search over the freshly ingested docs found nothing")
        top = docs[0]
        need("Axolotl" in top["title"], f"nearest neighbour is {top['title']!r}")
        need(isinstance(top["metadata"], dict) and top["metadata"],
             f"metadata was not parsed back out of meta_lc_json: {top['metadata']!r}")
        need(top["metadata"].get("kind") == "mcp-live-test",
             f"metadata round-trip lost 'kind': {top['metadata']!r}")
        need(top["metadata"].get("uri") == TOOL_META[0]["uri"],
             f"metadata round-trip lost 'uri': {top['metadata']!r}")
        return f"metadata round-tripped: kind={top['metadata']['kind']!r}, batch={top['metadata'].get('batch')!r}"

    check("metadata survives ingestion and comes back through opensolr_search",
          _t_tool_metadata_roundtrip)

    def _t_tool_delete_by_ids():
        msg = T["opensolr_delete_documents"](TMP_TOOL, ids=[TOOL_IDS[0]])
        need(msg == "deleted 1 document(s)", f"unexpected return value: {msg!r}")
        left = client.solr_select(TMP_TOOL, {"q": "*:*", "rows": 0})["response"]["numFound"]
        need(left == 1,
             f"deleting by the caller's ORIGINAL id (meta_ext_id) left {left} documents, expected 1")
        remaining = client.solr_select(
            TMP_TOOL, {"q": "*:*", "rows": 1, "fl": "title"})["response"]["docs"]
        need("Tardigrade" in str(remaining), f"the wrong document was deleted: {remaining}")
        return "deleted by the caller's original id via meta_ext_id, 1 left"

    check("opensolr_delete_documents removes documents by id", _t_tool_delete_by_ids, idempotent=False)

    def _t_tool_delete_by_query():
        msg = T["opensolr_delete_documents"](TMP_TOOL, query="*:*")
        need(msg == "deleted by query", f"unexpected return value: {msg!r}")
        left = client.solr_select(TMP_TOOL, {"q": "*:*", "rows": 0})["response"]["numFound"]
        need(left == 0, f"delete-by-query left {left} documents")
        return "index emptied by query"

    check("opensolr_delete_documents removes documents by query", _t_tool_delete_by_query, idempotent=False)

    def _t_delete_requires_target():
        try:
            T["opensolr_delete_documents"](TMP_TOOL)
        except ValueError as exc:
            need("Provide ids or query" in str(exc), f"unexpected message: {exc}")
            return "ValueError raised instead of deleting everything"
        raise Expected("a delete with neither ids nor query must be refused")

    check("opensolr_delete_documents refuses a call with no target",
          _t_delete_requires_target)

    # ------------------------------------------------------------------ #
    section("demo index untouched")
    # ------------------------------------------------------------------ #

    def _t_demo_intact():
        found = client.solr_select(DEMO, {"q": "*:*", "rows": 0})["response"]["numFound"]
        need(found == ST.get("demo_total"),
             f"the read-only demo index changed from {ST.get('demo_total')} to {found} documents")
        return f"{found} documents, unchanged across the whole run"

    check("the seeded demo index was never written to", _t_demo_intact)

finally:
    # ------------------------------------------------------------------ #
    section("cleanup")
    # ------------------------------------------------------------------ #
    # Runs even when the suite blew up half way: every temporary index this run
    # created is removed, and any straggler matching the throwaway pattern with
    # it. The pattern is anchored, so the demo index can never be caught by it.
    for name in (TMP_CLIENT, TMP_TOOL):
        def _delete(n=name):
            body = client.mgmt("delete_index", index_name=n)
            need(body.get("status") is True, f"delete_index failed: {body!r}")
            need(body.get("msg") == "DELETED_OK", f"unexpected msg: {body.get('msg')!r}")
            return f"{n} deleted"
        check(f"temporary index {name} removed", _delete, idempotent=False)

    def _verify_gone():
        # Only THIS run's indexes are asserted on. The demo account is shared and
        # the sibling packages' suites run against it concurrently, so anything
        # else named mcp_t_* belongs to another run and is left strictly alone —
        # a blanket sweep here would delete a live test's index out from under it.
        names = [r["index_name"] for r in client.get_index_list() or []]
        leftovers = sorted(MINE & set(names))
        need(not leftovers, f"this run's temporary indexes survived cleanup: {leftovers}")
        need(DEMO in names, f"the demo index disappeared! remaining: {names}")
        others = sorted(n for n in names if n != DEMO)
        return (f"both temporary indexes gone, {DEMO} intact"
                + (f" (left alone, not this run's: {others})" if others else ""))

    check("neither temporary index survived the run", _verify_gone)

    try:
        client.close()
        S._get_client().close()
        need(client._http.is_closed, "close() did not close the underlying httpx client")
        PASSED += 1
        print("✔ close() shuts the HTTP client down — is_closed is True", flush=True)
    except Exception as exc:                                          # noqa: BLE001
        FAILED += 1
        FAILURES.append(("close()", f"{type(exc).__name__}: {exc}"))
        print(f"✘ close() shuts the HTTP client down — {type(exc).__name__}: {exc}",
              flush=True)

    # ------------------------------------------------------------------ #
    print()
    print(f"{_request_count['api']} API requests, {_request_count['solr']} direct Solr requests")
    if TRANSIENT:
        print(f"\n{len(TRANSIENT)} transient transport error(s) survived by harness retry "
              f"(the package itself retries nothing):")
        for line in TRANSIENT:
            print(f"  ! {line}")
    if FAILURES:
        print("\nFAILURES")
        for label, why in FAILURES:
            print(f"  ✘ {label}\n      {why}")
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
