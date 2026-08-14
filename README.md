# opensolr-mcp

mcp-name: com.opensolr/opensolr-mcp

MCP (Model Context Protocol) server for [Opensolr](https://opensolr.com) —
gives any AI agent **managed Apache Solr search** as tools: hybrid
(BM25 + kNN) retrieval, server-side GPU embeddings, document indexing, and
grounded RAG answers.

No embedding model to configure. No vector database to run. One API key.

## Tools

| Tool | What it does |
|---|---|
| `opensolr_search` | Hybrid (keyword + semantic) or pure semantic search, with Solr filters |
| `opensolr_ai_answer` | Grounded RAG answer generated only from your indexed content |
| `opensolr_add_documents` | Index plain text + metadata (embedded server-side) |
| `opensolr_delete_documents` | Remove documents by id |
| `opensolr_list_indexes` / `opensolr_index_info` | Inspect the account's indexes |
| `opensolr_create_index` | Provision a vector-enabled index (`us`, `de`, `fi`) |
| `opensolr_vector_regions` | Live list of vector-enabled regions |

## Setup

Get a free Opensolr account (15-day trial, no card) at
[opensolr.com/register](https://opensolr.com/register) and copy your API key
from **Account**.

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
documents become searchable within about a minute. Progress is visible in the
Opensolr Control Panel and via the `ingest_status` API. Each document's
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

MIT license.
