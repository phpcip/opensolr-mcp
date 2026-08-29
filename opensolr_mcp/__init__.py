"""Opensolr MCP server — hybrid Solr search + RAG as Model Context Protocol tools."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opensolr_mcp.client import OpensolrClient, OpensolrError

try:
    # Read from the installed distribution rather than hard-coding a second copy of the
    # number — pyproject.toml is the single source of truth, and a duplicated literal is
    # how server.json ended up advertising 0.2.9 while the package was already 0.2.10.
    __version__ = _pkg_version("opensolr-mcp")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["OpensolrClient", "OpensolrError", "__version__"]
