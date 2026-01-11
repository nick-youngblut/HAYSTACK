# HAYSTACK Literature Search Specification

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Models](#3-data-models)
4. [Literature Search APIs](#4-literature-search-apis)
5. [Paper Acquisition Pipeline](#5-paper-acquisition-pipeline)
6. [PDF Processing](#6-pdf-processing)
7. [Literature Search Tools](#7-literature-search-tools)
8. [Integration Points](#8-integration-points)
9. [Caching and Deduplication](#9-caching-and-deduplication)
10. [Configuration](#10-configuration)
11. [Error Handling](#11-error-handling)
12. [Testing Strategy](#12-testing-strategy)
13. [Dependencies](#13-dependencies)

---

## 1. Overview

### 1.1 Purpose

The Literature Search module provides HAYSTACK with the ability to search, acquire, and process scientific literature from multiple sources. This capability is essential for:

1. **Query Understanding**: Gathering biological context about perturbations, cell types, diseases, and pathways
2. **Grounding Evaluation**: Validating STACK predictions against published evidence
3. **Report Generation**: Citing relevant literature in output reports

### 1.2 Design Principles

The architecture follows the patterns established in the `slack-paper-bot` project, adapted for HAYSTACK's async agent framework:

| Principle | Implementation |
|-----------|----------------|
| Multi-source search | Parallel queries to PubMed, Semantic Scholar, bioRxiv/medRxiv |
| Fallback acquisition | Ordered pipeline: Preprints → CORE → Europe PMC → Unpaywall |
| Full-text preference | Always attempt PDF acquisition before falling back to abstracts |
| Async-first | All I/O operations are async with shared HTTP client |
| Graceful degradation | Return partial results rather than failing completely |

### 1.3 Scope

**In Scope (MVP)**:
- Literature search across PubMed, Semantic Scholar, and bioRxiv/medRxiv
- Full-text PDF acquisition via multiple open-access sources
- PDF-to-markdown conversion using docling
- Abstract-only fallback when full-text unavailable
- Integration with Query Understanding and Grounding Evaluation agents

**Out of Scope (MVP)**:
- Paywalled content acquisition (requires institutional access)
- Citation network analysis
- Automatic paper summarization (handled by LLM)
- Literature-based hypothesis generation

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          LITERATURE SEARCH MODULE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        SEARCH LAYER                                     │    │
│  │                                                                         │    │
│  │   ┌─────────────┐   ┌─────────────────┐   ┌────────────────┐            │    │
│  │   │   PubMed    │   │ Semantic Scholar│   │ bioRxiv/medRxiv│            │    │
│  │   │  E-utilities│   │   Graph API     │   │   REST API     │            │    │
│  │   └──────┬──────┘   └───────┬─────────┘   └───────┬────────┘            │    │
│  │          │                  │                     │                     │    │
│  │          └──────────────────┼─────────────────────┘                     │    │
│  │                             │                                           │    │
│  │                    ┌────────▼────────┐                                  │    │
│  │                    │  Deduplication  │                                  │    │
│  │                    │   & Ranking     │                                  │    │
│  │                    └────────┬────────┘                                  │    │
│  └─────────────────────────────┼───────────────────────────────────────────┘    │
│                                │                                                │
│  ┌─────────────────────────────▼───────────────────────────────────────────┐    │
│  │                      ACQUISITION LAYER                                  │    │
│  │                                                                         │    │
│  │   ┌─────────────┐   ┌─────────────┐   ┌───────────┐   ┌───────────┐     │    │
│  │   │  Preprint   │   │  CORE API   │   │ Europe PMC│   │ Unpaywall │     │    │
│  │   │  Servers    │   │             │   │           │   │           │     │    │
│  │   │(arXiv,bioRxiv)│ │(API key req)│   │(free)     │   │(email req)│     │    │
│  │   └──────┬──────┘   └──────┬──────┘   └─────┬─────┘   └─────┬─────┘     │    │
│  │          │                 │                │               │           │    │
│  │          └─────────────────┴────────────────┴───────────────┘           │    │
│  │                                │                                        │    │
│  │                    ┌───────────▼───────────┐                            │    │
│  │                    │   PDF Download +      │                            │    │
│  │                    │   Validation          │                            │    │
│  │                    └───────────┬───────────┘                            │    │
│  └────────────────────────────────┼────────────────────────────────────────┘    │
│                                   │                                             │
│  ┌────────────────────────────────▼────────────────────────────────────────┐    │
│  │                      PROCESSING LAYER                                   │    │
│  │                                                                         │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │   │                    docling PDF Converter                        │   │    │
│  │   │                                                                 │   │    │
│  │   │   • Table extraction                                            │   │    │
│  │   │   • Figure handling                                             │   │    │
│  │   │   • Section parsing                                             │   │    │
│  │   │   • Optional: Image description via LLM                         │   │    │
│  │   └─────────────────────────────────────────────────────────────────┘   │    │
│  │                                │                                        │    │
│  │                    ┌───────────▼───────────┐                            │    │
│  │                    │   Markdown Output     │                            │    │
│  │                    └───────────────────────┘                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Integration with HAYSTACK Agents

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AGENT INTEGRATION                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ QUERY UNDERSTANDING SUBAGENT                                            │    │
│  │                                                                         │    │
│  │ Uses literature search to:                                              │    │
│  │ • Resolve ambiguous perturbation names                                  │    │
│  │ • Find drug targets and mechanisms of action                            │    │
│  │ • Identify expected pathways for perturbations                          │    │
│  │ • Gather cell type marker information                                   │    │
│  │                                                                         │    │
│  │ Tools:                                                                  │    │
│  │ • search_literature_tool                                                │    │
│  │ • acquire_paper_abstract_tool                                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ GROUNDING EVALUATION SUBAGENT                                           │    │
│  │                                                                         │    │
│  │ Uses literature search to:                                              │    │
│  │ • Validate DE genes against published studies                           │    │
│  │ • Check pathway enrichment against known biology                        │    │
│  │ • Find supporting/conflicting evidence for predictions                  │    │
│  │ • Compute literature_support score component                            │    │
│  │                                                                         │    │
│  │ Tools:                                                                  │    │
│  │ • search_literature_evidence_tool                                       │    │
│  │ • acquire_full_text_paper_tool                                          │    │
│  │ • extract_paper_findings_tool                                           │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ REPORT GENERATION (Orchestrator)                                        │    │
│  │                                                                         │    │
│  │ Uses literature search to:                                              │    │
│  │ • Cite supporting evidence in final report                              │    │
│  │ • Provide DOI links for further reading                                 │    │
│  │ • Note discrepancies with published literature                          │    │
│  │                                                                         │    │
│  │ Tools:                                                                  │    │
│  │ • format_citations_tool                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Models

### 3.1 Paper Record

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date


class PaperRecord(BaseModel):
    """Represents a paper from any source."""
    
    # Identifiers
    doi: Optional[str] = None
    pubmed_id: Optional[str] = None
    arxiv_id: Optional[str] = None
    
    # Metadata
    title: str
    authors: Optional[str] = None
    publication_date: Optional[str] = None  # YYYY-MM-DD or YYYY
    journal: Optional[str] = None
    
    # Content
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    content_type: Literal["full_text", "abstract", "metadata_only"] = "metadata_only"
    
    # Source tracking
    source: Literal[
        "pubmed",
        "semantic_scholar", 
        "biorxiv",
        "medrxiv",
        "arxiv",
        "crossref",
    ]
    
    # Acquisition metadata
    acquisition_source: Optional[Literal[
        "preprint_server",
        "core_api",
        "europe_pmc",
        "unpaywall",
    ]] = None
    
    def has_full_text(self) -> bool:
        """Check if full text is available."""
        return self.content_type == "full_text" and bool(self.full_text)
    
    def has_abstract(self) -> bool:
        """Check if abstract is available."""
        return bool(self.abstract)
    
    def get_best_content(self) -> str:
        """Return best available content."""
        return self.full_text or self.abstract or ""
    
    def to_markdown(self) -> str:
        """Convert to markdown format for LLM consumption."""
        lines = [f"# {self.title}", ""]
        
        if self.authors:
            lines.append(f"**Authors:** {self.authors}")
        if self.doi:
            lines.append(f"**DOI:** {self.doi}")
        if self.journal:
            lines.append(f"**Journal:** {self.journal}")
        if self.publication_date:
            lines.append(f"**Date:** {self.publication_date}")
        lines.append(f"**Content Type:** {self.content_type}")
        
        if self.abstract:
            lines.extend(["", "## Abstract", "", self.abstract])
        
        if self.full_text:
            lines.extend(["", "## Full Text", "", self.full_text])
        
        return "\n".join(lines)


class SearchResult(BaseModel):
    """Result from a literature search."""
    
    query: str
    papers: list[PaperRecord]
    total_found: int
    databases_searched: list[str]
    errors: list[str] = Field(default_factory=list)
    
    def deduplicated(self) -> "SearchResult":
        """Return copy with duplicates removed by DOI."""
        seen_dois: set[str] = set()
        unique_papers = []
        
        for paper in self.papers:
            doi = (paper.doi or "").strip().lower()
            if doi:
                if doi in seen_dois:
                    continue
                seen_dois.add(doi)
            unique_papers.append(paper)
        
        return SearchResult(
            query=self.query,
            papers=unique_papers,
            total_found=self.total_found,
            databases_searched=self.databases_searched,
            errors=self.errors,
        )


class LiteratureEvidence(BaseModel):
    """Evidence extracted from literature for grounding evaluation."""
    
    paper: PaperRecord
    relevance_score: float = Field(ge=0.0, le=1.0)
    evidence_type: Literal[
        "supports_prediction",
        "contradicts_prediction",
        "provides_context",
        "mechanism_description",
    ]
    extracted_findings: list[str]
    genes_mentioned: list[str] = Field(default_factory=list)
    pathways_mentioned: list[str] = Field(default_factory=list)
```

### 3.2 Acquisition Outcome

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DownloadOutcome:
    """Result of a PDF download attempt."""
    
    source: Literal[
        "arxiv_preprint",
        "biorxiv_preprint",
        "medrxiv_preprint",
        "core_api",
        "europe_pmc",
        "unpaywall",
    ]
    path: str  # Path to temporary PDF
    temp_dir: Optional[str] = None  # Directory to clean up
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        Path(self.path).unlink(missing_ok=True)
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


@dataclass  
class AcquisitionResult:
    """Result of full-text acquisition attempt."""
    
    status: Literal["success", "failed", "abstract_only"]
    doi: str
    markdown_path: Optional[str] = None
    markdown_content: Optional[str] = None
    source: Optional[str] = None
    content_type: Literal["full_text", "abstract"] = "abstract"
    message: str = ""
    error: bool = False
```

### 3.3 Literature Context (Runtime)

```python
from dataclasses import dataclass
import httpx


@dataclass
class LiteratureContext:
    """Runtime context for literature search tools.
    
    Provides shared resources via dependency injection pattern
    from LangChain's ToolRuntime.
    """
    
    # Shared HTTP client for connection pooling
    http_client: httpx.AsyncClient
    
    # API credentials
    core_api_key: Optional[str] = None
    unpaywall_email: Optional[str] = None
    
    # Configuration
    max_results_per_database: int = 20
    request_timeout: float = 30.0
    max_retries: int = 3
    
    # Rate limiting
    requests_per_minute: int = 30
    
    # PDF processing
    max_full_text_chars: int = 100_000
    enable_image_descriptions: bool = False
    
    # Temporary storage
    temp_dir: str = "/tmp/haystack/papers"
```

---

## 4. Literature Search APIs

### 4.1 PubMed E-utilities

PubMed provides comprehensive biomedical literature coverage via the NCBI E-utilities API.

```python
import asyncio
from xml.etree import ElementTree as ET
from typing import Any, Optional

import httpx

from haystack.shared.models.literature import PaperRecord


PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

MONTH_LOOKUP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _normalize_month(value: Optional[str]) -> Optional[str]:
    """Convert month name or number to two-digit format."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return value.zfill(2)
    return MONTH_LOOKUP.get(value.lower()[:3])


def _format_pub_date(year: str | None, month: str | None, day: str | None) -> str | None:
    """Format publication date as YYYY-MM-DD."""
    if not year:
        return None
    parts = [year.strip()]
    month_norm = _normalize_month(month)
    if month_norm:
        parts.append(month_norm)
        if day and day.isdigit():
            parts.append(day.zfill(2))
    return "-".join(parts)


def _collect_author_names(article: ET.Element) -> str | None:
    """Extract author names from PubMed article XML."""
    authors = []
    for author in article.findall(".//Author"):
        collective = author.find("CollectiveName")
        if collective is not None and collective.text:
            authors.append(collective.text.strip())
            continue
        last = author.find("LastName")
        fore = author.find("ForeName")
        if last is not None and last.text:
            name = last.text.strip()
            if fore is not None and fore.text:
                name = f"{fore.text.strip()} {name}"
            authors.append(name)
    return ", ".join(authors) if authors else None


def _collect_abstract(article: ET.Element) -> str | None:
    """Extract abstract text from PubMed article XML."""
    texts = []
    for abstract_text in article.findall(".//Abstract/AbstractText"):
        label = abstract_text.get("Label")
        text = abstract_text.text or ""
        if label:
            texts.append(f"{label}: {text.strip()}")
        elif text.strip():
            texts.append(text.strip())
    return "\n".join(texts) if texts else None


async def search_pubmed(
    query: str,
    max_results: int = 20,
    *,
    client: httpx.AsyncClient,
) -> list[PaperRecord]:
    """Search PubMed via E-utilities API.
    
    Args:
        query: Search query string (supports PubMed query syntax)
        max_results: Maximum number of results
        client: Shared HTTP client
        
    Returns:
        List of PaperRecord objects
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results <= 0:
        return []
    
    # Step 1: Search for PMIDs
    search_resp = await client.get(
        f"{PUBMED_BASE_URL}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        },
    )
    search_resp.raise_for_status()
    pmids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    
    if not pmids:
        return []
    
    # Step 2: Fetch article details
    fetch_resp = await client.get(
        f"{PUBMED_BASE_URL}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
    )
    fetch_resp.raise_for_status()
    
    # Step 3: Parse XML
    root = ET.fromstring(fetch_resp.text)
    papers = []
    
    for article in root.findall(".//PubmedArticle"):
        # Extract DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi" and article_id.text:
                doi = article_id.text.strip()
                break
        
        pmid_elem = article.find(".//PMID")
        title_elem = article.find(".//ArticleTitle")
        date_elem = article.find(".//PubDate")
        journal_elem = article.find(".//Journal/Title")
        
        year = date_elem.findtext("Year") if date_elem is not None else None
        month = date_elem.findtext("Month") if date_elem is not None else None
        day = date_elem.findtext("Day") if date_elem is not None else None
        
        papers.append(PaperRecord(
            pubmed_id=pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else None,
            doi=doi,
            title=title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown",
            authors=_collect_author_names(article),
            publication_date=_format_pub_date(year, month, day),
            journal=journal_elem.text.strip() if journal_elem is not None and journal_elem.text else None,
            abstract=_collect_abstract(article),
            source="pubmed",
            content_type="abstract" if _collect_abstract(article) else "metadata_only",
        ))
    
    return papers
```

### 4.2 Semantic Scholar Graph API

Semantic Scholar provides citation-aware search with broad coverage.

```python
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


async def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    *,
    client: httpx.AsyncClient,
) -> list[PaperRecord]:
    """Search Semantic Scholar Graph API.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        client: Shared HTTP client
        
    Returns:
        List of PaperRecord objects
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results <= 0:
        return []
    
    resp = await client.get(
        SEMANTIC_SCHOLAR_URL,
        params={
            "query": query,
            "limit": min(max_results, 100),  # API max is 100
            "fields": "title,authors,year,venue,externalIds,abstract",
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    
    papers = []
    for paper in payload.get("data", []):
        external_ids = paper.get("externalIds") or {}
        authors = ", ".join(
            author.get("name", "").strip()
            for author in paper.get("authors", [])
            if author.get("name")
        )
        year = paper.get("year")
        abstract = paper.get("abstract")
        
        papers.append(PaperRecord(
            pubmed_id=external_ids.get("PubMed"),
            doi=external_ids.get("DOI"),
            arxiv_id=external_ids.get("ArXiv"),
            title=paper.get("title") or "Unknown",
            authors=authors or None,
            publication_date=str(year) if year else None,
            journal=paper.get("venue"),
            abstract=abstract,
            source="semantic_scholar",
            content_type="abstract" if abstract else "metadata_only",
        ))
    
    return papers
```

### 4.3 bioRxiv/medRxiv API

Preprint servers provide the latest research before peer review.

```python
from datetime import datetime, timedelta


BIORXIV_BASE_URL = "https://api.biorxiv.org/details"


async def search_biorxiv(
    query: str,
    max_results: int = 20,
    *,
    date_window_days: int = 365,
    include_medrxiv: bool = True,
    client: httpx.AsyncClient,
) -> list[PaperRecord]:
    """Search bioRxiv/medRxiv within date window.
    
    Note: bioRxiv API doesn't support query search directly.
    We fetch recent papers and filter client-side.
    
    Args:
        query: Search query (filtered client-side)
        max_results: Maximum number of results
        date_window_days: Number of days to search back
        include_medrxiv: Also search medRxiv
        client: Shared HTTP client
        
    Returns:
        List of PaperRecord objects
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results <= 0:
        return []
    
    now = datetime.now()
    start = now - timedelta(days=date_window_days)
    
    servers = ["biorxiv"]
    if include_medrxiv:
        servers.append("medrxiv")
    
    papers = []
    query_terms = query.lower().split()
    
    for server in servers:
        cursor = 0
        while len(papers) < max_results and cursor < 200:
            url = f"{BIORXIV_BASE_URL}/{server}/{start:%Y-%m-%d}/{now:%Y-%m-%d}/{cursor}"
            
            resp = await client.get(url, timeout=30.0)
            resp.raise_for_status()
            collection = resp.json().get("collection", [])
            
            if not collection:
                break
            
            for entry in collection:
                # Client-side filtering
                text = f"{entry.get('title', '')} {entry.get('abstract', '')}".lower()
                if not all(term in text for term in query_terms):
                    continue
                
                doi = entry.get("doi")
                source = "medrxiv" if server == "medrxiv" else "biorxiv"
                
                papers.append(PaperRecord(
                    doi=doi,
                    title=entry.get("title") or "Unknown",
                    authors=entry.get("authors"),
                    publication_date=entry.get("date"),
                    journal=source.capitalize(),
                    abstract=entry.get("abstract"),
                    source=source,
                    content_type="abstract" if entry.get("abstract") else "metadata_only",
                ))
                
                if len(papers) >= max_results:
                    break
            
            cursor += 100
    
    return papers
```

### 4.4 Unified Search Function

```python
async def search_literature_parallel(
    query: str,
    max_results_per_db: int = 20,
    databases: list[str] | None = None,
    *,
    client: httpx.AsyncClient,
) -> SearchResult:
    """Search multiple literature databases in parallel.
    
    Args:
        query: Search query string
        max_results_per_db: Maximum results per database
        databases: List of databases to search (pubmed, semantic_scholar, biorxiv)
        client: Shared HTTP client
        
    Returns:
        SearchResult with deduplicated papers
    """
    databases = databases or ["pubmed", "semantic_scholar", "biorxiv"]
    
    database_functions = {
        "pubmed": search_pubmed,
        "semantic_scholar": search_semantic_scholar,
        "biorxiv": search_biorxiv,
    }
    
    # Validate database names
    for db in databases:
        if db not in database_functions:
            raise ValueError(f"Unknown database: {db}")
    
    # Search all databases in parallel
    coroutines = [
        database_functions[db](query, max_results_per_db, client=client)
        for db in databases
    ]
    results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Aggregate results
    all_papers = []
    errors = []
    
    for db, result in zip(databases, results):
        if isinstance(result, Exception):
            errors.append(f"{db}: {result}")
        else:
            all_papers.extend(result)
    
    search_result = SearchResult(
        query=query,
        papers=all_papers,
        total_found=len(all_papers),
        databases_searched=databases,
        errors=errors,
    )
    
    return search_result.deduplicated()
```

---

## 5. Paper Acquisition Pipeline

### 5.1 Acquisition Chain

The acquisition pipeline tries multiple sources in order until one succeeds:

```
1. Preprint Servers (arXiv, bioRxiv, medRxiv)
   ↓ (if DOI doesn't match preprint pattern)
2. CORE API (requires API key)
   ↓ (if not found or no API key)
3. Europe PMC (free, covers PubMed Central)
   ↓ (if not found)
4. Unpaywall (requires email, finds OA versions)
   ↓ (if all fail)
5. Abstract-only fallback
```

### 5.2 Preprint Server Downloads

```python
async def try_preprint_servers(
    doi: str,
    temp_path: str,
    *,
    client: httpx.AsyncClient,
) -> DownloadOutcome | None:
    """Download PDF directly from preprint servers.
    
    Args:
        doi: Paper DOI
        temp_path: Path to save temporary PDF
        client: Shared HTTP client
        
    Returns:
        DownloadOutcome if successful, None otherwise
    """
    doi_lower = doi.lower()
    
    # arXiv DOI: 10.48550/arXiv.XXXX.XXXX
    if "arxiv" in doi_lower:
        arxiv_id = doi_lower.split("arxiv.")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        source = "arxiv_preprint"
    
    # bioRxiv/medRxiv DOI: 10.1101/...
    elif doi_lower.startswith("10.1101/"):
        pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
        source = "biorxiv_preprint"
    
    else:
        return None
    
    try:
        resp = await client.get(pdf_url, follow_redirects=True)
        resp.raise_for_status()
        
        # Validate PDF magic bytes
        if b"%PDF" not in resp.content[:1024]:
            return None
        
        Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_path).write_bytes(resp.content)
        
        return DownloadOutcome(source=source, path=temp_path)
    
    except httpx.HTTPError:
        return None
```

### 5.3 CORE API

```python
CORE_BASE_URL = "https://api.core.ac.uk/v3"


async def try_core_api(
    doi: str,
    temp_path: str,
    *,
    api_key: str | None,
    client: httpx.AsyncClient,
) -> DownloadOutcome | None:
    """Download PDF via CORE API.
    
    Args:
        doi: Paper DOI
        temp_path: Path to save temporary PDF
        api_key: CORE API key
        client: Shared HTTP client
        
    Returns:
        DownloadOutcome if successful, None otherwise
    """
    if not api_key:
        return None
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Search for paper
        search_resp = await client.get(
            f"{CORE_BASE_URL}/search/works/",
            params={"q": f"doi:{doi}", "page": 1, "pageSize": 1},
            headers=headers,
        )
        search_resp.raise_for_status()
        
        results = search_resp.json().get("results", [])
        if not results:
            return None
        
        # Find download URL
        paper = results[0]
        pdf_url = paper.get("downloadUrl") or paper.get("pdfUrl")
        
        if not pdf_url:
            links = paper.get("fullTextLinks") or []
            pdf_url = links[0] if links else None
        
        if not pdf_url:
            return None
        
        # Download PDF
        pdf_resp = await client.get(pdf_url, headers=headers, follow_redirects=True)
        pdf_resp.raise_for_status()
        
        if b"%PDF" not in pdf_resp.content[:1024]:
            return None
        
        Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_path).write_bytes(pdf_resp.content)
        
        return DownloadOutcome(source="core_api", path=temp_path)
    
    except httpx.HTTPError:
        return None
```

### 5.4 Europe PMC

```python
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


async def try_europe_pmc(
    doi: str,
    temp_path: str,
    *,
    client: httpx.AsyncClient,
) -> DownloadOutcome | None:
    """Download PDF via Europe PMC.
    
    Args:
        doi: Paper DOI
        temp_path: Path to save temporary PDF
        client: Shared HTTP client
        
    Returns:
        DownloadOutcome if successful, None otherwise
    """
    try:
        # Search for paper
        search_resp = await client.get(
            EUROPE_PMC_SEARCH,
            params={"query": f"DOI:{doi}", "format": "json", "pageSize": 1},
        )
        search_resp.raise_for_status()
        
        results = search_resp.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        
        result = results[0]
        pdf_url = None
        
        # Try PMC PDF first
        if pmcid := result.get("pmcid"):
            pdf_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextPDF"
        
        # Fall back to fullTextUrlList
        if not pdf_url:
            for entry in result.get("fullTextUrlList", {}).get("fullTextUrl", []):
                if entry.get("documentStyle", "").lower() == "pdf":
                    pdf_url = entry.get("url")
                    break
        
        if not pdf_url:
            return None
        
        # Download PDF
        pdf_resp = await client.get(pdf_url, follow_redirects=True)
        pdf_resp.raise_for_status()
        
        if b"%PDF" not in pdf_resp.content[:1024]:
            return None
        
        Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_path).write_bytes(pdf_resp.content)
        
        return DownloadOutcome(source="europe_pmc", path=temp_path)
    
    except httpx.HTTPError:
        return None
```

### 5.5 Unpaywall

```python
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


async def try_unpaywall(
    doi: str,
    temp_path: str,
    *,
    email: str | None,
    client: httpx.AsyncClient,
) -> DownloadOutcome | None:
    """Download PDF via Unpaywall OA lookup.
    
    Args:
        doi: Paper DOI
        temp_path: Path to save temporary PDF
        email: Contact email (required by Unpaywall ToS)
        client: Shared HTTP client
        
    Returns:
        DownloadOutcome if successful, None otherwise
    """
    if not email:
        return None
    
    try:
        # Lookup OA status
        lookup_resp = await client.get(
            f"{UNPAYWALL_BASE}/{doi}",
            params={"email": email},
        )
        lookup_resp.raise_for_status()
        
        data = lookup_resp.json()
        best_location = data.get("best_oa_location") or {}
        pdf_url = best_location.get("url_for_pdf")
        
        if not pdf_url:
            return None
        
        # Download PDF
        pdf_resp = await client.get(pdf_url, follow_redirects=True)
        pdf_resp.raise_for_status()
        
        if b"%PDF" not in pdf_resp.content[:1024]:
            return None
        
        Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_path).write_bytes(pdf_resp.content)
        
        return DownloadOutcome(source="unpaywall", path=temp_path)
    
    except httpx.HTTPError:
        return None
```

### 5.6 Complete Acquisition Function

```python
async def acquire_full_text(
    doi: str,
    temp_dir: str,
    *,
    client: httpx.AsyncClient,
    core_api_key: str | None = None,
    unpaywall_email: str | None = None,
) -> DownloadOutcome | None:
    """Try all acquisition strategies in order.
    
    Args:
        doi: Paper DOI
        temp_dir: Directory for temporary files
        client: Shared HTTP client
        core_api_key: Optional CORE API key
        unpaywall_email: Optional email for Unpaywall
        
    Returns:
        DownloadOutcome if any source succeeds, None otherwise
    """
    if not doi:
        return None
    
    safe_doi = doi.replace("/", "_").replace(":", "_")
    temp_path = str(Path(temp_dir) / f"{safe_doi}.pdf")
    
    # Try each source in order
    for acquire_fn, kwargs in [
        (try_preprint_servers, {}),
        (try_core_api, {"api_key": core_api_key}),
        (try_europe_pmc, {}),
        (try_unpaywall, {"email": unpaywall_email}),
    ]:
        outcome = await acquire_fn(doi, temp_path, client=client, **kwargs)
        if outcome:
            return outcome
    
    return None
```

---

## 6. PDF Processing

### 6.1 PDF to Markdown Conversion

```python
from pathlib import Path


def convert_pdf_to_markdown(
    pdf_path: str,
    max_chars: int = 100_000,
    enable_image_descriptions: bool = False,
) -> str:
    """Convert PDF to markdown using docling.
    
    Args:
        pdf_path: Path to input PDF
        max_chars: Maximum characters to extract
        enable_image_descriptions: Use LLM to describe images
        
    Returns:
        Markdown string
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling is required for PDF conversion. Install docling>=2.60.0."
        ) from exc
    
    pipeline_options = PdfPipelineOptions()
    
    if enable_image_descriptions:
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            pipeline_options.do_picture_description = True
            pipeline_options.enable_remote_services = True
            pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                url="https://api.openai.com/v1/chat/completions",
                params={
                    "model": "gpt-4o-mini",
                    "seed": 42,
                    "max_completion_tokens": 300,
                },
                prompt="Describe this figure from a scientific paper in 2-3 sentences.",
                timeout=30,
                headers={"Authorization": f"Bearer {api_key}"},
            )
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(pdf_path)
    markdown = result.document.export_to_markdown()
    
    return markdown[:max_chars]
```

---

## 7. Literature Search Tools

### 7.1 Search Literature Tool

```python
from langchain.tools import tool, ToolRuntime


@tool
async def search_literature(
    query: str,
    max_results: int = 10,
    databases: list[str] | None = None,
    runtime: ToolRuntime[LiteratureContext],
) -> str:
    """Search scientific literature databases for relevant papers.
    
    Args:
        query: Search query (supports standard search syntax)
        max_results: Maximum results per database (default 10)
        databases: Databases to search (pubmed, semantic_scholar, biorxiv)
        
    Returns:
        Formatted list of papers with titles, authors, abstracts
    """
    ctx = runtime.context
    
    result = await search_literature_parallel(
        query,
        max_results_per_db=max_results,
        databases=databases,
        client=ctx.http_client,
    )
    
    if not result.papers:
        return f"No papers found for query: {query}"
    
    # Format results
    lines = [f"Found {len(result.papers)} papers for: {query}", ""]
    
    for i, paper in enumerate(result.papers[:20], 1):
        lines.append(f"## {i}. {paper.title}")
        if paper.authors:
            lines.append(f"**Authors:** {paper.authors}")
        if paper.doi:
            lines.append(f"**DOI:** {paper.doi}")
        if paper.journal:
            lines.append(f"**Journal:** {paper.journal} ({paper.publication_date or 'n.d.'})")
        if paper.abstract:
            abstract_preview = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
            lines.append(f"\n{abstract_preview}")
        lines.append("")
    
    if result.errors:
        lines.append(f"\n*Search errors: {', '.join(result.errors)}*")
    
    return "\n".join(lines)
```

### 7.2 Acquire Full Text Paper Tool

```python
@tool
async def acquire_full_text_paper(
    doi: str,
    runtime: ToolRuntime[LiteratureContext],
) -> str:
    """Acquire full-text paper content and convert to markdown.
    
    Tries preprint servers, CORE API, Europe PMC, and Unpaywall in order.
    Falls back to abstract if full text unavailable.
    
    Args:
        doi: Paper DOI (e.g., "10.1016/j.cell.2024.01.001")
        
    Returns:
        Paper content as markdown, or error message
    """
    ctx = runtime.context
    
    if not doi or not doi.strip():
        return "Error: DOI is required"
    
    # Try full-text acquisition
    outcome = await acquire_full_text(
        doi,
        ctx.temp_dir,
        client=ctx.http_client,
        core_api_key=ctx.core_api_key,
        unpaywall_email=ctx.unpaywall_email,
    )
    
    if outcome:
        try:
            markdown = convert_pdf_to_markdown(
                outcome.path,
                max_chars=ctx.max_full_text_chars,
                enable_image_descriptions=ctx.enable_image_descriptions,
            )
            outcome.cleanup()
            
            return f"# Full Text ({outcome.source})\n\n**DOI:** {doi}\n\n{markdown}"
        
        except Exception as exc:
            outcome.cleanup()
            # Fall through to abstract
    
    # Fall back to abstract
    abstract_result = await fetch_abstract_by_doi(doi, client=ctx.http_client)
    
    if abstract_result:
        return abstract_result.to_markdown()
    
    return f"Failed to acquire content for DOI: {doi}"
```

### 7.3 Search Literature Evidence Tool

```python
@tool
async def search_literature_evidence(
    genes: list[str],
    perturbation: str | None = None,
    cell_type: str | None = None,
    max_papers: int = 5,
    runtime: ToolRuntime[LiteratureContext],
) -> str:
    """Search for literature evidence supporting gene expression patterns.
    
    Constructs targeted queries to find papers discussing the specified
    genes in the context of perturbations and cell types.
    
    Args:
        genes: List of gene symbols to search for
        perturbation: Optional perturbation context (e.g., "TGF-beta")
        cell_type: Optional cell type context (e.g., "fibroblast")
        max_papers: Maximum papers to return
        
    Returns:
        Formatted evidence summary with citations
    """
    ctx = runtime.context
    
    # Build targeted query
    gene_str = " OR ".join(genes[:10])  # Limit genes in query
    query_parts = [f"({gene_str})"]
    
    if perturbation:
        query_parts.append(perturbation)
    if cell_type:
        query_parts.append(cell_type)
    
    query = " AND ".join(query_parts)
    
    result = await search_literature_parallel(
        query,
        max_results_per_db=max_papers,
        databases=["pubmed", "semantic_scholar"],
        client=ctx.http_client,
    )
    
    if not result.papers:
        return f"No literature evidence found for genes: {', '.join(genes)}"
    
    # Format as evidence
    lines = [f"## Literature Evidence for {', '.join(genes[:5])}{'...' if len(genes) > 5 else ''}", ""]
    
    for paper in result.papers[:max_papers]:
        lines.append(f"### {paper.title}")
        if paper.doi:
            lines.append(f"DOI: {paper.doi}")
        if paper.abstract:
            lines.append(f"\n{paper.abstract[:300]}...")
        lines.append("")
    
    return "\n".join(lines)
```

---

## 8. Integration Points

### 8.1 Query Understanding Integration

The Query Understanding subagent uses literature search to gather biological context:

```python
# In orchestrator/agents/query_understanding.py

QUERY_UNDERSTANDING_TOOLS = [
    # Cell Ontology tools (primary for cell type resolution)
    resolve_cell_type_semantic,
    get_cell_type_neighbors,
    query_cell_ontology_ols,

    # Other entity resolution tools
    resolve_perturbation_tool,
    resolve_tissue_tool,
    resolve_disease_tool,

    get_drug_targets_tool,
    get_pathway_priors_tool,
    # Literature tools
    search_literature,
    acquire_full_text_paper,
]

QUERY_UNDERSTANDING_PROMPT = """
You are a biological query understanding agent.

When resolving perturbations (drugs, cytokines):
1. First check internal databases (DrugBank, KEGG)
2. If targets are unclear, search literature for mechanism studies
3. Extract known targets and affected pathways from papers

When gathering biological context:
1. Search for recent reviews on the perturbation
2. Look for studies in similar cell types
3. Identify expected gene expression changes from literature
"""
```

### 8.2 Grounding Evaluation Integration

The Grounding Evaluation subagent uses literature to validate predictions:

```python
# In orchestrator/agents/grounding_evaluation.py

GROUNDING_EVALUATION_TOOLS = [
    extract_de_genes_tool,
    run_pathway_enrichment_tool,
    check_target_activation_tool,
    # Literature tools
    search_literature_evidence,
    acquire_full_text_paper,
    compute_grounding_score_tool,
]

GROUNDING_EVALUATION_PROMPT = """
You are a biological grounding evaluation agent.

For LITERATURE SUPPORT scoring (1-10):
1. Search for papers discussing the DE genes in context
2. Check if predicted direction matches published evidence
3. Score based on:
   - 9-10: Multiple papers directly support predictions
   - 7-8: Some supporting evidence, no contradictions
   - 5-6: Limited evidence, results plausible
   - 3-4: No direct evidence, results speculative
   - 1-2: Evidence contradicts predictions

Always cite specific papers with DOIs in your evaluation.
"""
```

### 8.3 Runtime Context Setup

```python
# In orchestrator/main.py

async def create_literature_context(config: HaystackConfig) -> LiteratureContext:
    """Create literature context for the orchestrator run."""
    
    return LiteratureContext(
        http_client=httpx.AsyncClient(
            timeout=config.database_apis.request_timeout,
            follow_redirects=True,
            headers={"User-Agent": "HAYSTACK/1.0 (Arc Institute)"},
        ),
        core_api_key=os.environ.get("CORE_API_KEY"),
        unpaywall_email=os.environ.get("UNPAYWALL_EMAIL"),
        max_results_per_database=20,
        request_timeout=30.0,
        max_retries=3,
        requests_per_minute=30,
        max_full_text_chars=100_000,
        enable_image_descriptions=False,
        temp_dir=f"/tmp/haystack/{run_id}/papers",
    )
```

---

## 9. Caching and Deduplication

### 9.1 Search Result Caching

```python
from functools import lru_cache
from hashlib import sha256
import json


class LiteratureCache:
    """Simple in-memory cache for literature search results."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: dict[str, tuple[float, SearchResult]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _make_key(self, query: str, databases: list[str]) -> str:
        """Create cache key from query parameters."""
        data = json.dumps({"query": query.lower(), "databases": sorted(databases)})
        return sha256(data.encode()).hexdigest()[:16]
    
    def get(self, query: str, databases: list[str]) -> SearchResult | None:
        """Get cached result if valid."""
        import time
        key = self._make_key(query, databases)
        
        if key in self._cache:
            timestamp, result = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                return result
            del self._cache[key]
        
        return None
    
    def set(self, query: str, databases: list[str], result: SearchResult) -> None:
        """Cache search result."""
        import time
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        
        key = self._make_key(query, databases)
        self._cache[key] = (time.time(), result)
```

### 9.2 DOI-based Deduplication

```python
def deduplicate_papers(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Remove duplicate papers by DOI, keeping the record with most content."""
    
    doi_to_paper: dict[str, PaperRecord] = {}
    no_doi_papers: list[PaperRecord] = []
    
    for paper in papers:
        doi = (paper.doi or "").strip().lower()
        
        if not doi:
            no_doi_papers.append(paper)
            continue
        
        if doi in doi_to_paper:
            # Keep the one with more content
            existing = doi_to_paper[doi]
            if paper.has_full_text() and not existing.has_full_text():
                doi_to_paper[doi] = paper
            elif paper.has_abstract() and not existing.has_abstract():
                doi_to_paper[doi] = paper
        else:
            doi_to_paper[doi] = paper
    
    return list(doi_to_paper.values()) + no_doi_papers
```

---

## 10. Configuration

### 10.1 Literature Configuration Schema

```yaml
# In backend/settings.yml

default:
  literature:
    # Search settings
    max_results_per_database: 20
    default_databases:
      - pubmed
      - semantic_scholar
      - biorxiv
    
    # Acquisition settings
    enable_full_text: true
    max_full_text_chars: 100_000
    enable_image_descriptions: false
    
    # Rate limiting
    requests_per_minute: 30
    request_timeout_seconds: 30
    max_retries: 3
    
    # Caching
    cache_ttl_seconds: 3600
    cache_max_size: 100
    
    # bioRxiv/medRxiv specific
    biorxiv_date_window_days: 365
    include_medrxiv: true

dev:
  literature:
    # Reduce limits for development
    max_results_per_database: 5
    cache_ttl_seconds: 60

prod:
  literature:
    # Full limits for production
    max_results_per_database: 20
    cache_ttl_seconds: 7200
```

### 10.2 Environment Variables

```bash
# .env.example

# Literature API credentials
CORE_API_KEY=          # Optional: CORE API key for expanded access
UNPAYWALL_EMAIL=       # Required for Unpaywall: contact email

# Optional: For image descriptions in PDFs
OPENAI_API_KEY=        # OpenAI key for docling image descriptions
```

---

## 11. Error Handling

### 11.1 Retry Strategy

```python
import asyncio
import logging
from typing import TypeVar, Callable

import httpx

logger = logging.getLogger(__name__)
T = TypeVar("T")


def should_retry_http_error(exc: Exception) -> bool:
    """Determine if HTTP error is retryable."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


async def async_retry(
    func: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (httpx.HTTPError,),
    should_retry: Callable[[Exception], bool] = should_retry_http_error,
) -> T:
    """Retry async function with exponential backoff."""
    last_exc = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as exc:
            last_exc = exc
            if not should_retry(exc) or attempt == max_attempts - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed: {exc}. Retrying in {delay}s")
            await asyncio.sleep(delay)
    
    raise last_exc  # type: ignore
```

### 11.2 Graceful Degradation

```python
async def search_with_fallback(
    query: str,
    ctx: LiteratureContext,
) -> SearchResult:
    """Search with graceful degradation on partial failures."""
    
    results = []
    errors = []
    
    # Try each database independently
    for db in ctx.default_databases:
        try:
            if db == "pubmed":
                papers = await search_pubmed(query, ctx.max_results_per_database, client=ctx.http_client)
            elif db == "semantic_scholar":
                papers = await search_semantic_scholar(query, ctx.max_results_per_database, client=ctx.http_client)
            elif db == "biorxiv":
                papers = await search_biorxiv(query, ctx.max_results_per_database, client=ctx.http_client)
            else:
                continue
            results.extend(papers)
        except Exception as exc:
            errors.append(f"{db}: {str(exc)}")
            logger.warning(f"Search failed for {db}: {exc}")
    
    # Return partial results even if some databases failed
    return SearchResult(
        query=query,
        papers=deduplicate_papers(results),
        total_found=len(results),
        databases_searched=ctx.default_databases,
        errors=errors,
    ).deduplicated()
```

---

## 12. Testing Strategy

### 12.1 Unit Tests

```python
# tests/unit/test_literature_search.py

import pytest
from unittest.mock import AsyncMock, patch

from haystack.orchestrator.services.literature import (
    search_pubmed,
    search_semantic_scholar,
    search_biorxiv,
    deduplicate_papers,
)


@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    return AsyncMock()


class TestPubMedSearch:
    async def test_empty_query_raises(self, mock_http_client):
        with pytest.raises(ValueError, match="non-empty string"):
            await search_pubmed("", client=mock_http_client)
    
    async def test_parses_xml_correctly(self, mock_http_client):
        # Mock response with sample PubMed XML
        mock_http_client.get.return_value.json.return_value = {
            "esearchresult": {"idlist": ["12345"]}
        }
        # ... test XML parsing


class TestDeduplication:
    def test_removes_duplicate_dois(self):
        papers = [
            PaperRecord(doi="10.1234/test", title="Paper 1", source="pubmed"),
            PaperRecord(doi="10.1234/test", title="Paper 1 Copy", source="semantic_scholar"),
        ]
        result = deduplicate_papers(papers)
        assert len(result) == 1
    
    def test_keeps_paper_with_most_content(self):
        papers = [
            PaperRecord(doi="10.1234/test", title="Paper 1", source="pubmed"),
            PaperRecord(doi="10.1234/test", title="Paper 1", source="semantic_scholar", abstract="Has abstract"),
        ]
        result = deduplicate_papers(papers)
        assert result[0].abstract == "Has abstract"
```

### 12.2 Integration Tests

```python
# tests/integration/test_literature_acquisition.py

import pytest
import httpx

from haystack.orchestrator.services.literature import (
    acquire_full_text,
    search_literature_parallel,
)


@pytest.mark.integration
class TestLiteratureAcquisition:
    """Integration tests requiring network access."""
    
    @pytest.fixture
    def http_client(self):
        return httpx.AsyncClient(timeout=30.0)
    
    async def test_arxiv_acquisition(self, http_client, tmp_path):
        """Test arXiv PDF download."""
        doi = "10.48550/arXiv.2301.12345"
        outcome = await acquire_full_text(
            doi,
            str(tmp_path),
            client=http_client,
        )
        
        if outcome:  # May fail if paper doesn't exist
            assert outcome.source == "arxiv_preprint"
            assert Path(outcome.path).exists()
            outcome.cleanup()
    
    async def test_pubmed_search(self, http_client):
        """Test PubMed search returns results."""
        result = await search_literature_parallel(
            "CRISPR gene editing",
            max_results_per_db=5,
            databases=["pubmed"],
            client=http_client,
        )
        
        assert len(result.papers) > 0
        assert all(p.source == "pubmed" for p in result.papers)
```

---

## 13. Dependencies

### 13.1 Python Dependencies

```toml
# pyproject.toml additions

[project.optional-dependencies]
literature = [
    "httpx>=0.27.0",
    "docling>=2.60.0",
    "lxml>=5.0.0",
]
```

### 13.2 API Dependencies

| API | Required | Rate Limits | Authentication |
|-----|----------|-------------|----------------|
| PubMed E-utilities | No | 3 req/sec (10 with key) | Optional API key |
| Semantic Scholar | No | 100 req/5min | None required |
| bioRxiv API | No | Not documented | None required |
| CORE API | Optional | 10 req/min (free tier) | API key required |
| Europe PMC | No | Not documented | None required |
| Unpaywall | Optional | 100k req/day | Email required |

### 13.3 External Services

- **docling**: For PDF-to-markdown conversion (local processing)
- **OpenAI API**: Optional, for image descriptions in PDFs

## Related Specs

- `specification/agents.md`
- `specification/tools.md`
- `specification/data-models.md`
