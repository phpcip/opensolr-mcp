"""Opensolr MCP server — managed Apache Solr with hybrid (BM25 + kNN) search
and server-side GPU embeddings, exposed as Model Context Protocol tools.

Credentials come from the environment:
    OPENSOLR_EMAIL    — Opensolr account email
    OPENSOLR_API_KEY  — Opensolr API key (Account > API in the control panel)

To try it without an account, use the shared public demo credentials
mcp@opensolr.com / 420b8b23e7b12dc8ab838932145a5065 — it comes with the index
mcp_demo_d1__dense (300 news articles), anything created there is deleted after
3 days, automatically, other people can change or delete what you create, and the
limits are per index and small on purpose: 200 MB bandwidth, 50 MB disk. For a
private index that persists: https://opensolr.com/register (free forever, no card).

Run: ``opensolr-mcp`` (stdio transport — for Claude Desktop, Cursor, etc.)
"""

from __future__ import annotations

import json
import base64
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from mcp.server import MCPServer

from . import __version__
from .client import (
    OpensolrClient,
    OpensolrError,
    apply_fresh_bias,
    apply_search_operators,
    parse_operators,
    resolve_location,
)

mcp = MCPServer(
    "opensolr",
    # Without this the handshake reports serverInfo.version as an empty string, so every
    # client and directory listing shows the server as version-less and cannot tell one
    # release from another (seen in the Glama build log, 2026-08-29).
    version=__version__,
    instructions=(
        "Managed Apache Solr search for the user's Opensolr account. "
        "Use opensolr_search for retrieval (hybrid keyword+semantic by default), "
        "opensolr_ai_answer for a grounded RAG answer, and the document tools "
        "to index or remove content. Embedding happens server-side — tools "
        "accept plain text, never vectors."
    ),
)

_client: Optional[OpensolrClient] = None


def _get_client() -> OpensolrClient:
    global _client
    if _client is None:
        email = os.environ.get("OPENSOLR_EMAIL", "")
        api_key = os.environ.get("OPENSOLR_API_KEY", "")
        if not (email and api_key):
            # A user who hits this is stuck inside their agent client with no way out,
            # so hand them working demo credentials right here instead of a signup link.
            raise OpensolrError(
                "Set OPENSOLR_EMAIL and OPENSOLR_API_KEY in the MCP server environment. "
                "To try it now: OPENSOLR_EMAIL=mcp@opensolr.com "
                "OPENSOLR_API_KEY=420b8b23e7b12dc8ab838932145a5065 — shared public demo "
                "account with the index mcp_demo_d1__dense (300 news articles); anything "
                "created there is deleted after 3 days, automatically. For a private index "
                "that persists: https://opensolr.com/register (free forever, no card)."
            )
        _client = OpensolrClient(email, api_key)
    return _client


_META_KEY_RE = re.compile(r"[^a-z0-9_]+")

_HYBRID_MODES = ("union", "keywords_required", "meaning_required", "intersection")


def _doc_out(solr_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a Solr doc for LLM consumption: id, title, text, score, metadata."""

    def _flat(v: Any) -> Any:
        if isinstance(v, list):
            return v[0] if len(v) == 1 else v
        return v

    metadata: Dict[str, Any] = {}
    raw = _flat(solr_doc.get("meta_lc_json"))
    if raw:
        try:
            metadata = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            metadata = {}
    text = _flat(solr_doc.get("text", "")) or ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text)
    return {
        "id": str(_flat(solr_doc.get("id", ""))),
        "title": str(_flat(solr_doc.get("title", "")) or "")[:200],
        "text": str(text)[:2000],
        "score": float(_flat(solr_doc.get("score", 0.0)) or 0.0),
        "metadata": metadata,
    }


@mcp.tool()
def opensolr_search(
    index: str,
    query: str,
    k: int = 5,
    search_mode: str = "hybrid",
    mode: str = "union",
    alpha: float = 0.5,
    filter_query: Optional[str] = None,
    fresh_bias: bool = False,
) -> List[Dict[str, Any]]:
    """Search an Opensolr index and return the k most relevant documents.

    search_mode: "hybrid" (default — BM25 + semantic kNN fused per document),
    "semantic" (pure kNN), or "lexical" (pure keyword edismax — no embedding
    call, zero AI quota, works on ANY index including non-vector ones).
    For hybrid: mode is union / keywords_required / meaning_required /
    intersection, alpha balances semantic (0) vs lexical (1).
    filter_query accepts a raw Solr fq expression, e.g. 'meta_category:"docs"'.
    fresh_bias biases the ranking toward recent documents by multiplying each
    score by a recency curve on creation_date. It re-orders and never filters:
    the hit count is unchanged, nothing old becomes unreachable, and a document
    with no creation_date is simply left unboosted. Works in all three search
    modes. Off by default — turn it on when newer should beat older on a tie.
    """
    client = _get_client()
    return _do_search(client, index, query, k, search_mode, mode, alpha, filter_query, fresh_bias)


def _do_search(
    client: "OpensolrClient",
    index: str,
    query: str,
    k: int,
    search_mode: str,
    mode: str,
    alpha: float,
    filter_query: Optional[str],
    fresh_bias: bool,
) -> List[Dict[str, Any]]:
    """Shared search body used by opensolr_search and opensolr_search_by_image."""
    params: Dict[str, Any] = {"rows": k, "fl": "*,score"}
    # Search operators (+word, -word, +"phrase", -"phrase") come out of the text once, for
    # every mode below. See parse_operators() for why they cannot stay inside the query on
    # any path that involves a vector — an agent that types "-Ruben" currently gets MORE
    # Ruben, not less, because the minus sign is just another token to the embedder.
    ops = parse_operators(query)
    # Nothing left once the operators are removed ("-Ruben" alone): the operators ARE the
    # query. Hand the whole string to edismax, which understands them natively, and emit no
    # filters.
    ops_only = ops["has_ops"] and len(ops["base"]) < 2
    lexical_text = query if (not ops["has_ops"] or ops_only) else ops["base"]

    if search_mode == "hybrid" and mode not in _HYBRID_MODES:
        raise ValueError(f"mode must be one of {_HYBRID_MODES}")
    if search_mode == "lexical" or ops_only:
        # Bound by reference (2026-09-09) instead of inlined: inlining meant a '}' in the
        # caller's text closed the local-param block, which is why braces AND quotes were
        # stripped first — and stripping the quotes silently broke every phrase query.
        params["uq"] = lexical_text
        params["q"] = '{!edismax qf="title^100 description^20 text^1" v=$uq}'
    else:
        # The embedder must never see an operator.
        vector = client.embed(
            index, ops["base"] if ops["has_ops"] else query, is_query=True
        )
        compact = json.dumps(vector, separators=(",", ":"))
        knn = f"{{!knn f=embeddings topK={max(k, 10)}}}{compact}"
        if search_mode == "hybrid":
            params["uq"] = lexical_text
            params["q"] = (
                f"{{!hybrid lexical=$lexicalRaw vector=$vectorQuery "
                f"mode={mode} alpha={alpha} topN={max(k, 10)}}}"
            )
            params["lexicalRaw"] = '{!edismax qf="title^100 text^1" v=$uq}'
            params["vectorQuery"] = knn
        else:
            params["q"] = knn
        # Operators become filters on both vector-bearing shapes. On the pure-kNN shape this
        # is the only thing that can honour them at all — that query has no edismax in it.
        apply_search_operators(params, ops)
    if fresh_bias:
        apply_fresh_bias(params)
    if filter_query:
        # Append: the operator filters and the Fresh Results Bias date clause are already in
        # params by now, and a plain assignment would drop both.
        existing = params.get("fq")
        if existing is None:
            params["fq"] = filter_query
        elif isinstance(existing, list):
            params["fq"] = existing + [filter_query]
        else:
            params["fq"] = [existing, filter_query]
    body = client.solr_select(index, params)
    return [_doc_out(d) for d in body["response"]["docs"]]


@mcp.tool()
def opensolr_search_by_image(
    index: str,
    image_path: str,
    k: int = 5,
    using: str = "auto",
    search_mode: str = "hybrid",
    mode: str = "union",
    alpha: float = 0.5,
    filter_query: Optional[str] = None,
    fresh_bias: bool = False,
) -> Dict[str, Any]:
    """Search an Opensolr index with a PHOTO instead of a text query.

    The image at image_path is read three ways by the Opensolr image engine —
    visual labels (what it depicts), OCR text (words printed on it), and any
    barcode / QR code — and turned into a text query that runs through the normal
    search. No image vector is stored; the picture simply becomes words.

    using selects which reading drives the search:
      "auto"    the engine's chosen text (OCR text when the picture is mostly text,
                otherwise the visual labels) — the default,
      "meaning" the visual labels (what the picture depicts),
      "text"    only the OCR text read off the picture (empty if none),
      "code"    the first barcode / QR code, matched as an exact keyword.

    search_mode / mode / alpha / fresh_bias / filter_query behave exactly as in
    opensolr_search. Returns {"read": {text, mode, labels, codes}, "results": [...]}
    so the caller sees both what the picture was read as and the matching documents.
    """
    client = _get_client()
    with open(image_path, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode("ascii")
    ans = client.image_to_text(index, image_b64, top_k=8)
    labels = [l["label"] for l in (ans.get("labels") or []) if isinstance(l, dict) and l.get("label")]
    codes = []
    for c in (ans.get("codes") or []):
        text = c.get("text") if isinstance(c, dict) else (c if isinstance(c, str) else None)
        if text:
            codes.append(text)
    read = {"text": (ans.get("text") or "").strip(), "mode": ans.get("mode") or "clip", "labels": labels, "codes": codes}

    if using == "meaning":
        q = ", ".join(labels)
    elif using == "text":
        q = read["text"] if read["mode"] == "ocr" else ""
    elif using == "code":
        q = codes[0] if codes else ""
        search_mode = "lexical"  # a code is an exact token
    elif using == "all":
        parts = list(labels)
        if read["mode"] == "ocr" and read["text"]:
            parts.append(read["text"])
        parts.extend(codes)
        q = ", ".join(p for p in parts if p)
    elif using == "auto":
        q = read["text"]
    else:
        raise ValueError("using must be one of 'auto', 'meaning', 'text', 'code', 'all'")

    results = _do_search(client, index, q, k, search_mode, mode, alpha, filter_query, fresh_bias) if q else []
    return {"read": read, "results": results}


@mcp.tool()
def opensolr_ai_answer(
    index: str,
    query: str,
    filter_query: Optional[str] = None,
    # 4 documents is the platform's measured context size (OpensolrClient.RAG_DOCS);
    # this used to pass 3, which quietly overrode the client default with a smaller one.
    rag_docs: int = 4,
    rag_words: int = 1500,
    instruction: Optional[str] = None,
    tuning: Optional[Dict[str, Any]] = None,
) -> str:
    """Ask a question and get a grounded RAG answer generated ONLY from the
    content already indexed in the given Opensolr index. Retrieval runs
    through the platform's tuned hybrid pipeline (the index's saved Search
    Tuning applies automatically) — the top rag_docs hits (first rag_words
    words of text each) become the LLM context, the same pipeline as the
    hosted search UI. filter_query optionally narrows retrieval with a raw
    Solr fq expression; instruction optionally replaces the default prompt
    (e.g. "Answer in German", "Extract a list of people"); tuning optionally
    overrides retrieval knobs per call. That list is the whole set, not a
    sample — an abbreviated one reads as everything that is supported, and
    freshness_boost was invisible to callers because of it: fw_title,
    fw_description, fw_uri, fw_text, fw_text_t, lexical_weight,
    vector_weight, vector_topk, search_mode (union / keywords_required /
    meaning_required / intersection), quality_boost, min_score,
    freshness_boost, fresh_bias, lexical_norm_k, mm (flexible / balanced /
    strict or raw Solr mm syntax). freshness_boost and fresh_bias are different
    knobs despite the names: the first is a hard window in DAYS that filters
    older documents out, the second only re-orders, multiplying each score by a
    recency curve on creation_date so recent documents win ties while nothing
    becomes unreachable."""
    return _get_client().ai_summary(
        index, query, filter_query=filter_query,
        rag_docs=rag_docs, rag_words=rag_words, instruction=instruction,
        tuning=tuning,
    )


@mcp.tool()
def opensolr_list_indexes() -> List[Dict[str, str]]:
    """List all search indexes in the connected Opensolr account."""
    return _get_client().get_index_list()


@mcp.tool()
def opensolr_index_info(index: str) -> Dict[str, Any]:
    """Get connection details for an index: Solr URL, version, environment.
    (Credentials are intentionally not returned.)"""
    info = _get_client().get_core_info(index)
    return {
        "connection_url": info.get("connection_url"),
        "solr_version": info.get("solr_version"),
        "environment": info.get("environment_identifier"),
        "type": info.get("type"),
    }


@mcp.tool()
def opensolr_add_documents(
    index: str,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    wait: bool = True,
) -> Dict[str, Any]:
    """Index plain-text documents via the Opensolr Data Ingestion API.

    Ingestion is ASYNC: embeddings, sentiment, and all derived fields are
    computed server-side and documents become searchable within about a
    minute (progress is visible in the Opensolr Control Panel). With
    wait=true (default) this blocks until the job completes. Metadata keys
    become filterable meta_* fields; metadata "uri" (a real URL) is used as
    the document identity — the Solr id is md5(uri). Returns job info and
    the resulting Solr document ids.
    """
    import hashlib
    from urllib.parse import quote

    client = _get_client()
    metadatas = metadatas or [{} for _ in texts]
    ids = ids or [str(uuid.uuid4()) for _ in texts]
    if not (len(texts) == len(metadatas) == len(ids)):
        raise ValueError("texts, metadatas and ids must have the same length")

    docs = []
    solr_ids = []
    for text, meta, doc_id in zip(texts, metadatas, ids):
        meta = dict(meta or {})
        uri = meta.get("uri") or meta.get("url")
        if not (isinstance(uri, str) and uri.startswith(("http://", "https://"))):
            uri = f"https://ingest.opensolr.com/{index}/{quote(str(doc_id), safe='')}"
        uri = uri.rstrip("/")
        text = text or " "
        doc: Dict[str, Any] = {
            "uri": uri,
            "title": str(meta.get("title") or text[:100] or uri)[:250],
            "description": str(meta.get("description") or text[:200]),
            "text": text,
            "meta_ext_id": str(doc_id),
            "meta_lc_json": json.dumps(meta, ensure_ascii=False),
        }
        if meta.get("rtf"):
            doc["rtf"] = True
        if meta.get("timestamp"):
            doc["timestamp"] = meta["timestamp"]
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)) and key not in ("rtf", "uri", "url"):
                doc[f"meta_{_META_KEY_RE.sub('_', str(key).lower()).strip('_')}"] = str(value)
        docs.append(doc)
        solr_ids.append(hashlib.md5(uri.encode()).hexdigest())

    results = []
    for i in range(0, len(docs), 50):
        results.append(client.ingest(index, docs[i : i + 50], wait=wait))
    return {
        "queued_jobs": [r.get("job_id") for r in results],
        "doc_ids": solr_ids,
        "note": "Ingestion is asynchronous; check opensolr_ingest_status or the Control Panel.",
    }


@mcp.tool()
def opensolr_ingest_status(job_id: str) -> Dict[str, Any]:
    """Status of a Data Ingestion job (state, processed/success/failed doc
    counts). Also visible in the Opensolr Control Panel."""
    return _get_client().ingest_status(job_id)


@mcp.tool()
def opensolr_delete_documents(
    index: str,
    ids: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> str:
    """Delete documents by ids (Solr ids or your original ids) or by a raw
    Solr query, e.g. 'meta_category:"drafts"' or '+id:"abc123"'."""
    client = _get_client()
    if query:
        client.solr_update(index, {"delete": {"query": query}})
        return "deleted by query"
    if not ids:
        raise ValueError("Provide ids or query")
    joined = " OR ".join('"' + str(i).replace('"', '') + '"' for i in ids)
    client.solr_update(index, {"delete": {"query": f"id:({joined}) OR meta_ext_id:({joined})"}})
    return f"deleted {len(ids)} document(s)"


@mcp.tool()
def opensolr_create_index(index: str, location: str = "us") -> Dict[str, Any]:
    """Create a new vector-enabled Opensolr index. location: us, de, fi, or
    any environment id from opensolr_vector_regions. Additional dedicated
    regions can be deployed on request (support@opensolr.com)."""
    return _get_client().create_index(index, resolve_location(location))


@mcp.tool()
def opensolr_vector_regions() -> List[Dict[str, str]]:
    """List the vector-enabled Opensolr environments currently available
    (Solr 9.x with dense vectors and the hybrid query parser)."""
    return _get_client().vector_regions()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
