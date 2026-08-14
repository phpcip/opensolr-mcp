"""Opensolr MCP server — hybrid Solr search + RAG as Model Context Protocol tools."""

from opensolr_mcp.client import OpensolrClient, OpensolrError

__all__ = ["OpensolrClient", "OpensolrError"]
