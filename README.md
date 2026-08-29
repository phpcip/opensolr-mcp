# opensolr-mcp

mcp-name: com.opensolr/opensolr-mcp

MCP (Model Context Protocol) server for [Opensolr](https://opensolr.com) —
gives any AI agent **managed Apache Solr search** as tools: hybrid
(BM25 + kNN) retrieval, server-side GPU embeddings, document indexing, and
grounded RAG answers.

**See it live (real news index, hybrid + AI answer):** https://search.opensolr.com/news__dense?q=how+am+I+supposed+to+save+money%3F

No embedding model to configure. No vector database to run. One API key.

## Tools

| Tool | What it does |
|---|---|
| `opensolr_search` | Hybrid (keyword + semantic) or pure semantic search, with Solr filters |
| `opensolr_ai_answer` | Grounded RAG answer: top hybrid hits become the LLM context — same pipeline as the hosted search UI |
| `opensolr_add_documents` | Index plain text + metadata (embedded server-side) |
| `opensolr_delete_documents` | Remove documents by id |
| `opensolr_list_indexes` / `opensolr_index_info` | Inspect the account's indexes |
| `opensolr_create_index` | Provision a vector-enabled index (`us`, `de`, `fi`) |
| `opensolr_vector_regions` | Live list of vector-enabled regions |

## Setup

Get a free Opensolr account (15-day trial, no card) at
[opensolr.com/register](https://opensolr.com/register) and copy your API key
from **Account**.

> **Trying it from a directory listing?** Some catalogues show this server with a
> shared demo credential, so every tool works immediately without signing up —
> create an index, push documents into it, search, ask. It is a genuinely open
> demo account: **everyone browsing that page shares it**, anything you create
> there is visible to them, they can change or delete it, and you can do the same
> to theirs. Indexes on it are wiped after **3 days**.
>
> That is fine for a five-minute look and useless for anything else. The moment
> you want an index that is yours, is private, and stays put, get your own key —
> [free trial, no card](https://opensolr.com/register), or see
> [pricing](https://opensolr.com/pricing) — and put your own `OPENSOLR_EMAIL` and
> `OPENSOLR_API_KEY` in the config below.

### Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "opensolr": {
      "command": "uvx",
      "args": ["opensolr-mcp"],
      "env": {
        "OPENSOLR_EMAIL": "you@example.com",
        "OPENSOLR_API_KEY": "YOUR_OPENSOLR_API_KEY"
      }
    }
  }
}
```

### Cursor / Windsurf / any MCP client

Same shape — stdio transport, command `uvx opensolr-mcp` (or
`pipx run opensolr-mcp`), with `OPENSOLR_EMAIL` and `OPENSOLR_API_KEY` in env.

## Example agent session

> **You:** Index our FAQ answers, then find everything about refunds.
>
> The agent calls `opensolr_add_documents(index="faq__dense", texts=[...])`,
> then `opensolr_search(index="faq__dense", query="refund policy", hybrid=True)`
> — BM25 catches the exact word "refund", kNN catches "giving customers their
> money back", and the scores fuse per document.

## Notes

- Vector-enabled indexes run on Opensolr's Solr 9.x environments — currently
  `us` (Chicago), `de` (Germany), `fi` (Finland), fetched live via
  `opensolr_vector_regions`. Additional dedicated regions can be deployed on
  request (paid add-on): [support@opensolr.com](mailto:support@opensolr.com).
- Every index is also plain Apache Solr with the native `/select` API —
  nothing is locked behind the tools.
- Python sibling for LangChain: [`langchain-opensolr`](https://pypi.org/project/langchain-opensolr/) ·
  Product page: [opensolr.com/langchain](https://opensolr.com/langchain)

## How writing works (Data Ingestion API)

Writes go through Opensolr's [Data Ingestion API](https://opensolr.com/learn/api-data-ingestion/204/data-ingestion-api-push-documents-to-your-opensolr-index-programmatically)
— the same pipeline the Drupal and WordPress connectors use. It is
**asynchronous**: documents are queued, then embeddings, sentiment, language
and all crawler-identical derived fields are computed **server-side**, and
documents become searchable within about a minute. Progress is visible in
**Control Panel → Data Ingestion** — a per-job status board (queued /
processing / completed / failed, with processed / success / failed document
counts per job) — and via the `ingest_status` API. Each document's
identity is its `uri` (the Solr id is `md5(uri)`): pass a real URL in
metadata (`{"uri": "https://..."}`), or a deterministic one is synthesized
from your id. Re-submitting the same `uri` updates the document. Pass
`{"rtf": True, "uri": "https://.../file.pdf"}` and the server extracts the
text from PDF/DOCX/XLSX for you.

## Lexical-only mode

Don't need vectors? Pure keyword search skips the embedding call entirely —
zero AI quota, and it works on **any** Opensolr index, including non-vector
ones and older Solr versions.

## Your index schema

Documents follow the Opensolr document model (`title`, `description`, `text`,
`meta_*` custom fields). To see the full schema: **Control Panel → click your
index → Configuration → Edit File → schema.xml**. Prefer zero-effort data
entry? Configure the **Web Crawler** in the Control Panel (Index Tools →
WebCrawler): add your site URL, validate it, and Opensolr indexes the whole
site for you.


### Search tuning

Retrieval (search and RAG grounding) runs through the platform's tuned
pipeline: global defaults → your index's saved **Search Tuning** (Control
Panel → Index Settings → Search Tuning: semantic↔lexical balance, field
weights, minimum match, search mode, vector candidate pool, content quality
boost) → optional per-call overrides via `tuning`:

```
tuning={"search_mode": "keywords_required", "fw_title": 0.2,
        "mm": "strict", "vector_topk": 500, "quality_boost": 0.3}
```

Defaults match the platform's PHP configuration exactly — customize in the
Control Panel once, or per call from code.

#### Fresh Results Bias

Rank newer documents higher without hiding anything older. Every score is
multiplied by a recency curve on `creation_date` — full weight for a document
published today, about half after a year:

```python
store.similarity_search_with_score("solar inverter warranty", fresh_bias=True)
client.hybrid_search(index, query, fresh_bias=True)
client.ai_answer(index, question, tuning={"fresh_bias": 1})
```

It **re-orders and never filters**: the hit count is identical either way,
nothing old becomes unreachable, and a document with no `creation_date` simply
keeps its place instead of being pushed to the bottom. It applies to all three
retrieval shapes — vector-only, keyword-only and the fused hybrid ranking —
because the boost wraps the final score rather than one half of it. Off by
default.

This is the same control visitors get as the **Fresh** toggle beside the sort
options on the hosted Opensolr search page, so a query behaves identically here
and there.

> `fresh_bias` and `freshness_boost` are two different knobs and the names
> invite confusion. `freshness_boost` is a hard window in **days** — anything
> older is filtered out and the hit count drops. `fresh_bias` filters nothing.

## How it's tested

Every release is validated against **live Opensolr infrastructure** — no mocks:

- **Unit tests** (offline): location aliases, filter→fq mapping, query building, escaping.
- **End-to-end suite**: the full write path through the async Data Ingestion
  queue (queued → server-side enrichment → searchable), semantic / hybrid /
  lexical retrieval, metadata round-trip, filters, id round-trip (your ids
  and the Solr `md5(uri)` ids), deletes by id and by query.
- **Real-corpus validation**: searches run against a 340-document replica of
  opensolr.com's own production search index. Verified: pure-semantic hits
  with zero keyword overlap ("how do I get my data back after a disaster" →
  backup &amp; restore docs), cross-lingual queries (Romanian query → English
  content), exact-term surfacing in hybrid mode, all four hybrid modes, and
  the full alpha range 0 → 1.
- **PDF ingestion**: a real PDF ingested via `rtf:true` — server-side text
  extraction (13k+ chars), automatic content-type detection, then retrieved
  with a purely semantic query against its contents.

The tools are exercised live (search modes, ingestion with wait, status,
deletes, RAG answers) before every release. RAG grounding is verified
end-to-end: a question answerable only from the ingested PDF returns the
correct answer sourced from the PDF's extracted text.

MIT license.
