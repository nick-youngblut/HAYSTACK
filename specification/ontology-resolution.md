# Cell Ontology Resolution Specification

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Data Model](#3-data-model)
4. [Database Schema](#4-database-schema)
5. [Ontology Tools](#5-ontology-tools)
6. [Integration with HAYSTACK Components](#6-integration-with-haystack-components)
7. [Data Loading Pipeline](#7-data-loading-pipeline)
8. [Configuration](#8-configuration)
9. [Error Handling](#9-error-handling)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. Overview

### 1.1 Purpose

The Cell Ontology (CL) resolution module provides HAYSTACK with the ability to map free-text cell type descriptions to standardized Cell Ontology terms. This capability is essential for:

1. **Query Understanding**: Resolving user-specified cell types (e.g., "lung fibroblast", "activated T cell") to canonical CL IDs
2. **Ontology-Guided Retrieval**: Finding related cell types via the CL hierarchy for prompt cell selection
3. **Metadata Harmonization**: Standardizing cell type annotations across heterogeneous atlases

### 1.2 Design Principles

The architecture follows the patterns established in the `arc-ontology-mcp` project, adapted for HAYSTACK's native integration:

| Principle | Implementation |
|-----------|----------------|
| Semantic search first | Use OpenAI embeddings for fuzzy text-to-term mapping |
| Graph traversal | Leverage CL relationships (is_a, part_of, develops_from) for hierarchical queries |
| OLS fallback | Query EBI Ontology Lookup Service when local database lacks matches |
| Batch operations | Support multiple labels in single calls with automatic deduplication |
| Async-first | All database and API operations are async |

### 1.3 Scope

**In Scope (MVP)**:
- Cell Ontology (CL) term resolution via semantic search
- CL graph traversal for related terms (parents, children, siblings)
- OLS API fallback for edge cases
- Integration with Query Understanding and Prompt Generation agents
- Pre-populated ontology data loaded from GCS

**Out of Scope (MVP)**:
- Tissue Ontology (UBERON) resolution (future extension)
- Disease Ontology (MONDO) resolution (future extension)
- Real-time ontology updates (manual refresh via CLI)
- Custom ontology term additions

---

## 2. Architecture

### 2.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CELL ONTOLOGY RESOLUTION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                           AGENT TOOLS                                   │    │
│  │                                                                         │    │
│  │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │    │
│  │   │ resolve_cell_   │  │ get_cell_type_  │  │ query_cell_     │         │    │
│  │   │ type_semantic   │  │ neighbors       │  │ ontology_ols    │         │    │
│  │   │                 │  │                 │  │                 │         │    │
│  │   │ Free-text →     │  │ CL ID →         │  │ Keyword →       │         │    │
│  │   │ CL ID(s)        │  │ Related terms   │  │ CL terms        │         │    │
│  │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │    │
│  │            │                    │                    │                  │    │
│  └────────────┼────────────────────┼────────────────────┼──────────────────┘    │
│               │                    │                    │                       │
│               ▼                    ▼                    ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      ONTOLOGY SERVICE LAYER                             │    │
│  │                                                                         │    │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │   │                    CellOntologyService                          │   │    │
│  │   │                                                                 │   │    │
│  │   │   • semantic_search(labels, k, threshold)                       │   │    │
│  │   │   • get_neighbors(term_ids)                                     │   │    │
│  │   │   • query_ols(search_terms)                                     │   │    │
│  │   │   • get_term_by_id(term_id)                                     │   │    │
│  │   │   • get_lineage(term_id)                                        │   │    │
│  │   └─────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│               │                    │                    │                       │
│               ▼                    ▼                    ▼                       │
│  ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────┐    │
│  │   Cloud SQL          │  │   OpenAI API          │  │   OLS API         │    │
│  │   (PostgreSQL +      │  │   (Embeddings)        │  │   (EBI)           │    │
│  │    pgvector)         │  │                       │  │                   │    │
│  │                      │  │   text-embedding-     │  │   ols.ebi.ac.uk   │    │
│  │   • ontology_terms   │  │   3-small             │  │                   │    │
│  │   • ontology_rels    │  │   1536 dimensions     │  │                   │    │
│  └───────────────────────┘  └───────────────────────┘  └───────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Integration with HAYSTACK

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    HAYSTACK AGENT INTEGRATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ QUERY UNDERSTANDING SUBAGENT                                            │    │
│  │                                                                         │    │
│  │ Uses Cell Ontology tools to:                                            │    │
│  │ • Resolve free-text cell types to CL IDs                                │    │
│  │ • Get synonyms for cell type matching                                   │    │
│  │ • Retrieve cell type definitions for context                           │    │
│  │                                                                         │    │
│  │ Tools:                                                                  │    │
│  │ • resolve_cell_type_semantic (primary)                                  │    │
│  │ • query_cell_ontology_ols (fallback)                                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ PROMPT GENERATION SUBAGENT                                              │    │
│  │                                                                         │    │
│  │ Uses Cell Ontology tools to:                                            │    │
│  │ • Find related cell types when exact match unavailable                  │    │
│  │ • Navigate CL hierarchy for fallback cell selection                     │    │
│  │ • Identify parent/child cell types for ontology-guided retrieval        │    │
│  │                                                                         │    │
│  │ Tools:                                                                  │    │
│  │ • get_cell_type_neighbors                                               │    │
│  │ • resolve_cell_type_semantic                                            │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Model

### 3.1 Ontology Term Model

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class OntologyTerm(BaseModel):
    """A Cell Ontology term with embedding."""
    
    term_id: str = Field(
        description="Cell Ontology ID (e.g., 'CL:0000057')"
    )
    name: str = Field(
        description="Canonical term name (e.g., 'fibroblast')"
    )
    definition: Optional[str] = Field(
        default=None,
        description="Term definition from ontology"
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description="Alternative names for the term"
    )
    ontology_type: str = Field(
        default="cell",
        description="Ontology type identifier"
    )
    version: str = Field(
        description="Ontology version (YYYY-MM-DD format)"
    )
    
    class Config:
        frozen = True


class OntologyRelationship(BaseModel):
    """A relationship between two ontology terms."""
    
    subject_term_id: str = Field(
        description="Source term ID"
    )
    object_term_id: str = Field(
        description="Target term ID"
    )
    relationship_type: str = Field(
        description="Relationship type (is_a, part_of, develops_from, etc.)"
    )
    ontology_type: str = Field(
        default="cell",
        description="Ontology type identifier"
    )
    version: str = Field(
        description="Ontology version"
    )


class OntologySearchResult(BaseModel):
    """Result from semantic search or OLS query."""
    
    term_id: str
    name: str
    definition: Optional[str] = None
    distance: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Cosine distance (0 = identical, 1 = orthogonal)"
    )
    relationship_type: Optional[str] = Field(
        default=None,
        description="Relationship type if from neighbor query"
    )


class CellTypeResolution(BaseModel):
    """Result of resolving a free-text cell type label."""
    
    query_label: str = Field(description="Original query text")
    resolved: bool = Field(description="Whether resolution succeeded")
    
    # Best match
    term_id: Optional[str] = Field(default=None)
    term_name: Optional[str] = Field(default=None)
    definition: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence score (1 - distance)"
    )
    
    # Alternative matches
    alternatives: list[OntologySearchResult] = Field(default_factory=list)
    
    # Resolution method
    method: str = Field(
        description="Resolution method: 'semantic', 'ols', or 'none'"
    )
```

### 3.2 Resolution Request/Response Models

```python
class CellTypeResolutionRequest(BaseModel):
    """Request to resolve cell type labels."""
    
    labels: list[str] = Field(
        description="Free-text cell type labels to resolve",
        min_length=1,
        max_length=100
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of candidates per label"
    )
    distance_threshold: float = Field(
        default=0.7,
        ge=0,
        le=1,
        description="Maximum cosine distance for matches"
    )
    include_alternatives: bool = Field(
        default=True,
        description="Include alternative matches in response"
    )


class CellTypeNeighborRequest(BaseModel):
    """Request to get related cell types."""
    
    term_ids: list[str] = Field(
        description="Cell Ontology term IDs (e.g., ['CL:0000057'])",
        min_length=1,
        max_length=50
    )
    relationship_types: Optional[list[str]] = Field(
        default=None,
        description="Filter by relationship types (is_a, part_of, etc.)"
    )
    max_distance: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Maximum graph distance to traverse"
    )
```

---

## 4. Database Schema

### 4.1 Ontology Tables

The Cell Ontology data is stored in HAYSTACK's Cloud SQL database alongside the cell metadata tables.

```sql
-- =============================================================================
-- CELL ONTOLOGY TABLES
-- =============================================================================

-- Ontology terms table with vector embeddings
CREATE TABLE ontology_terms (
    id SERIAL PRIMARY KEY,
    
    -- Term identifiers
    term_id VARCHAR(32) NOT NULL,
    name VARCHAR(512) NOT NULL,
    definition TEXT,
    
    -- Ontology metadata
    ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
    version VARCHAR(16) NOT NULL,
    
    -- Vector embedding for semantic search
    -- OpenAI text-embedding-3-small produces 1536-dimensional vectors
    embedding vector(1536),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT uq_ontology_terms_id_version 
        UNIQUE (term_id, ontology_type, version)
);

-- Ontology relationships table
CREATE TABLE ontology_relationships (
    id SERIAL PRIMARY KEY,
    
    -- Relationship endpoints
    subject_term_id VARCHAR(32) NOT NULL,
    object_term_id VARCHAR(32) NOT NULL,
    relationship_type VARCHAR(64) NOT NULL,
    
    -- Ontology metadata
    ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
    version VARCHAR(16) NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT uq_ontology_rels 
        UNIQUE (subject_term_id, object_term_id, relationship_type, ontology_type, version)
);

-- =============================================================================
-- INDEXES
-- =============================================================================

-- Term lookup indexes
CREATE INDEX idx_ontology_terms_term_id ON ontology_terms(term_id);
CREATE INDEX idx_ontology_terms_name ON ontology_terms(name);
CREATE INDEX idx_ontology_terms_type_version ON ontology_terms(ontology_type, version);

-- HNSW vector index for semantic search
-- Build AFTER loading data for best performance
CREATE INDEX idx_ontology_terms_embedding ON ontology_terms
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Relationship indexes
CREATE INDEX idx_ontology_rels_subject ON ontology_relationships(subject_term_id);
CREATE INDEX idx_ontology_rels_object ON ontology_relationships(object_term_id);
CREATE INDEX idx_ontology_rels_type ON ontology_relationships(relationship_type);
CREATE INDEX idx_ontology_rels_version ON ontology_relationships(ontology_type, version);
```

### 4.2 Database Access Patterns

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector


# Query: Semantic search for cell type labels
SEMANTIC_SEARCH_SQL = """
    SELECT 
        term_id,
        name,
        definition,
        1 - (embedding <=> $1::vector) as similarity,
        embedding <=> $1::vector as distance
    FROM ontology_terms
    WHERE ontology_type = $2
      AND version = $3
      AND embedding IS NOT NULL
    ORDER BY embedding <=> $1::vector
    LIMIT $4
"""


# Query: Get neighbors for a term
GET_NEIGHBORS_SQL = """
    SELECT DISTINCT
        r.object_term_id as term_id,
        t.name,
        t.definition,
        r.relationship_type
    FROM ontology_relationships r
    JOIN ontology_terms t ON r.object_term_id = t.term_id 
        AND t.ontology_type = r.ontology_type 
        AND t.version = r.version
    WHERE r.subject_term_id = $1
      AND r.ontology_type = $2
      AND r.version = $3
    
    UNION
    
    SELECT DISTINCT
        r.subject_term_id as term_id,
        t.name,
        t.definition,
        r.relationship_type || '_inverse' as relationship_type
    FROM ontology_relationships r
    JOIN ontology_terms t ON r.subject_term_id = t.term_id 
        AND t.ontology_type = r.ontology_type 
        AND t.version = r.version
    WHERE r.object_term_id = $1
      AND r.ontology_type = $2
      AND r.version = $3
"""


# Query: Get latest ontology version
GET_LATEST_VERSION_SQL = """
    SELECT DISTINCT version
    FROM ontology_terms
    WHERE ontology_type = $1
    ORDER BY version DESC
    LIMIT 1
"""


# Query: Get term by ID
GET_TERM_BY_ID_SQL = """
    SELECT term_id, name, definition
    FROM ontology_terms
    WHERE term_id = $1
      AND ontology_type = $2
      AND version = $3
"""
```

---

## 5. Ontology Tools

### 5.1 resolve_cell_type_semantic

Primary tool for mapping free-text cell type labels to Cell Ontology terms using semantic search.

```python
from langchain_core.tools import tool
from typing import Optional
import yaml


@tool
async def resolve_cell_type_semantic(
    cell_labels: str,
    k: int = 3,
    distance_threshold: float = 0.7,
) -> str:
    """
    Map free-text cell type labels to Cell Ontology terms using semantic search.
    
    Uses OpenAI embeddings to find the most semantically similar CL terms.
    Input labels are automatically deduplicated.
    
    Args:
        cell_labels: Semicolon-separated list of free-text cell type labels.
                     Example: "fibroblast; activated T cell; lung epithelial"
        k: Number of nearest neighbors to return per label (1-10, default 3)
        distance_threshold: Maximum cosine distance for matches (0-1, default 0.7).
                           Lower values = stricter matching.
    
    Returns:
        YAML-formatted results mapping each label to matched CL terms.
        
        Example output:
        ```yaml
        fibroblast:
          - term_id: CL:0000057
            name: fibroblast
            definition: A connective tissue cell which secretes...
            distance: 0.05
          - term_id: CL:0000058
            name: chondroblast
            definition: A cell that secretes cartilage matrix...
            distance: 0.32
        activated T cell:
          - term_id: CL:0000911
            name: activated T cell
            definition: A T cell that has been activated...
            distance: 0.08
        unknown cell type: No ontology ID found
        ```
    
    Notes:
        - Distance scores range from 0 (identical) to 1 (orthogonal)
        - Labels with no matches above threshold return "No ontology ID found"
        - Use this tool to resolve cell types mentioned in user queries
        - For hierarchical relationships, use get_cell_type_neighbors
    """
    from haystack.orchestrator.services.ontology import CellOntologyService
    
    # Parse semicolon-separated labels
    labels = [label.strip() for label in cell_labels.split(";") if label.strip()]
    
    if not labels:
        return "Error: No valid cell labels provided"
    
    # Deduplicate while preserving order
    seen = set()
    unique_labels = []
    for label in labels:
        if label.lower() not in seen:
            seen.add(label.lower())
            unique_labels.append(label)
    
    # Perform semantic search
    service = CellOntologyService.get_instance()
    results = await service.semantic_search(
        labels=unique_labels,
        k=k,
        distance_threshold=distance_threshold,
    )
    
    # Format as YAML
    return yaml.dump(results, sort_keys=False, indent=2, allow_unicode=True)
```

### 5.2 get_cell_type_neighbors

Tool for traversing the Cell Ontology graph to find related cell types.

```python
@tool
async def get_cell_type_neighbors(
    term_ids: str,
) -> str:
    """
    Get related Cell Ontology terms through ontology relationships.
    
    Retrieves all neighbor terms connected to the specified CL IDs via
    ontology relationships (is_a, part_of, develops_from, etc.).
    
    Args:
        term_ids: Semicolon-separated list of Cell Ontology term IDs.
                  Example: "CL:0000057; CL:0000236"
    
    Returns:
        YAML-formatted results mapping each term ID to its neighbors.
        
        Example output:
        ```yaml
        CL:0000057:
          - term_id: CL:0000548
            name: animal cell
            definition: A cell of the body of an animal...
            relationship_type: is_a
          - term_id: CL:0002553
            name: fibroblast of lung
            definition: A fibroblast that is part of lung...
            relationship_type: is_a_inverse
        CL:0000236:
          - term_id: CL:0000945
            name: lymphocyte
            definition: A leukocyte of the lymphoid lineage...
            relationship_type: is_a
        CL:9999999: Error: Invalid term ID format. Expected CL:XXXXXXX
        ```
    
    Relationship types:
        - is_a: Term is a subtype of neighbor (child → parent)
        - is_a_inverse: Neighbor is a subtype of term (parent → child)
        - part_of: Term is part of neighbor
        - part_of_inverse: Neighbor is part of term
        - develops_from: Term develops from neighbor
        - develops_from_inverse: Neighbor develops from term
    
    Notes:
        - Use this tool when exact cell type match isn't available
        - Navigate hierarchy to find broader (parent) or narrower (child) types
        - Invalid term IDs return an error message
        - Terms with no relationships return an empty list
    """
    from haystack.orchestrator.services.ontology import CellOntologyService
    
    # Parse semicolon-separated term IDs
    ids = [tid.strip() for tid in term_ids.split(";") if tid.strip()]
    
    if not ids:
        return "Error: No valid term IDs provided"
    
    # Validate and deduplicate
    valid_ids = []
    invalid_ids = []
    seen = set()
    
    for tid in ids:
        if tid in seen:
            continue
        seen.add(tid)
        
        # Validate CL ID format
        if tid.startswith("CL:") and len(tid) == 10:
            valid_ids.append(tid)
        else:
            invalid_ids.append(tid)
    
    # Get neighbors
    service = CellOntologyService.get_instance()
    results = await service.get_neighbors(term_ids=valid_ids)
    
    # Add invalid IDs to results
    for tid in invalid_ids:
        results[tid] = f"Error: Invalid term ID format. Expected CL:XXXXXXX"
    
    # Format as YAML
    return yaml.dump(results, sort_keys=False, indent=2, allow_unicode=True)
```

### 5.3 query_cell_ontology_ols

Fallback tool for querying the EBI Ontology Lookup Service when local database lacks matches.

```python
@tool
async def query_cell_ontology_ols(
    search_terms: str,
) -> str:
    """
    Query the Ontology Lookup Service (OLS) for Cell Ontology terms.
    
    Uses keyword search against the EBI OLS API as a fallback when 
    semantic search doesn't find matches. Useful for very specific or
    newly added ontology terms.
    
    Args:
        search_terms: Semicolon-separated list of search terms.
                      Example: "fibroblast; B lymphocyte; stem cell"
    
    Returns:
        YAML-formatted results mapping each search term to matched CL terms.
        
        Example output:
        ```yaml
        fibroblast:
          - term_id: CL:0000057
            name: fibroblast
            definition: A connective tissue cell which secretes...
        B lymphocyte:
          - term_id: CL:0000236
            name: B cell
            definition: A lymphocyte of B lineage...
        unknown term: []
        ```
    
    Notes:
        - Only returns Cell Ontology (CL:) terms
        - Results are filtered by exact ontology prefix
        - Use this as a fallback when semantic search returns no matches
        - Rate limited to avoid overwhelming the OLS API
        - Empty list indicates no results found
    """
    from haystack.orchestrator.services.ontology import CellOntologyService
    
    # Parse semicolon-separated terms
    terms = [term.strip() for term in search_terms.split(";") if term.strip()]
    
    if not terms:
        return "Error: No valid search terms provided"
    
    # Deduplicate while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique_terms.append(term)
    
    # Query OLS
    service = CellOntologyService.get_instance()
    results = await service.query_ols(search_terms=unique_terms)
    
    # Format as YAML
    return yaml.dump(results, sort_keys=False, indent=2, allow_unicode=True)
```

---

## 6. Integration with HAYSTACK Components

### 6.1 CellOntologyService Implementation

```python
# orchestrator/services/ontology.py
"""Cell Ontology service for HAYSTACK."""

import asyncio
from typing import Any, Optional
from contextlib import asynccontextmanager

import httpx
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from haystack.shared.config import HaystackConfig
from haystack.orchestrator.services.database import get_session_factory


class CellOntologyService:
    """Service for Cell Ontology resolution and graph traversal."""
    
    _instance: Optional["CellOntologyService"] = None
    
    # Constants
    ONTOLOGY_TYPE = "cell"
    ONTOLOGY_PREFIX = "CL"
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536
    OLS_BASE_URL = "https://www.ebi.ac.uk/ols4/api"
    
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        openai_client: AsyncOpenAI,
        http_client: httpx.AsyncClient,
        config: HaystackConfig,
    ):
        """
        Initialize the Cell Ontology service.
        
        Args:
            session_factory: SQLAlchemy async session factory
            openai_client: OpenAI client for embeddings
            http_client: HTTP client for OLS API
            config: HAYSTACK configuration
        """
        self.session_factory = session_factory
        self.openai_client = openai_client
        self.http_client = http_client
        self.config = config
        self._version_cache: Optional[str] = None
    
    @classmethod
    def get_instance(cls) -> "CellOntologyService":
        """Get singleton instance (initialized during orchestrator startup)."""
        if cls._instance is None:
            raise RuntimeError(
                "CellOntologyService not initialized. "
                "Call initialize() during orchestrator startup."
            )
        return cls._instance
    
    @classmethod
    async def initialize(cls, config: HaystackConfig) -> "CellOntologyService":
        """Initialize the singleton instance."""
        if cls._instance is not None:
            return cls._instance
        
        session_factory = await get_session_factory(config.database)
        openai_client = AsyncOpenAI()
        http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "HAYSTACK/1.0 (Arc Institute)"},
        )
        
        cls._instance = cls(
            session_factory=session_factory,
            openai_client=openai_client,
            http_client=http_client,
            config=config,
        )
        return cls._instance
    
    async def get_latest_version(self) -> str:
        """Get the latest Cell Ontology version in the database."""
        if self._version_cache:
            return self._version_cache
        
        async with self._get_session() as session:
            result = await session.execute(
                """
                SELECT DISTINCT version
                FROM ontology_terms
                WHERE ontology_type = :ontology_type
                ORDER BY version DESC
                LIMIT 1
                """,
                {"ontology_type": self.ONTOLOGY_TYPE}
            )
            row = result.fetchone()
            
            if not row:
                raise RuntimeError(
                    f"No Cell Ontology data found in database. "
                    f"Please load ontology data from GCS."
                )
            
            self._version_cache = row[0]
            return self._version_cache
    
    @asynccontextmanager
    async def _get_session(self):
        """Get database session."""
        async with self.session_factory() as session:
            yield session
    
    async def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text strings."""
        response = await self.openai_client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]
    
    async def semantic_search(
        self,
        labels: list[str],
        k: int = 3,
        distance_threshold: float = 0.7,
    ) -> dict[str, list[dict[str, Any]] | str]:
        """
        Map free-text labels to CL terms using semantic search.
        
        Args:
            labels: List of cell type labels
            k: Number of results per label
            distance_threshold: Maximum cosine distance
        
        Returns:
            Dict mapping labels to results or error message
        """
        version = await self.get_latest_version()
        
        # Generate embeddings for all labels
        embeddings = await self._generate_embeddings(labels)
        
        results: dict[str, list[dict[str, Any]] | str] = {}
        
        async with self._get_session() as session:
            for label, embedding in zip(labels, embeddings):
                # Query database with vector similarity
                result = await session.execute(
                    """
                    SELECT 
                        term_id,
                        name,
                        definition,
                        embedding <=> :embedding::vector as distance
                    FROM ontology_terms
                    WHERE ontology_type = :ontology_type
                      AND version = :version
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :k
                    """,
                    {
                        "embedding": embedding,
                        "ontology_type": self.ONTOLOGY_TYPE,
                        "version": version,
                        "k": k,
                    }
                )
                rows = result.fetchall()
                
                # Filter by distance threshold
                matches = [
                    {
                        "term_id": row[0],
                        "name": row[1],
                        "definition": row[2],
                        "distance": round(row[3], 4),
                    }
                    for row in rows
                    if row[3] <= distance_threshold
                ]
                
                if matches:
                    results[label] = matches
                else:
                    results[label] = "No ontology ID found"
        
        return results
    
    async def get_neighbors(
        self,
        term_ids: list[str],
    ) -> dict[str, list[dict[str, Any]] | str]:
        """
        Get related terms through ontology relationships.
        
        Args:
            term_ids: List of CL term IDs
        
        Returns:
            Dict mapping term IDs to neighbor lists
        """
        version = await self.get_latest_version()
        results: dict[str, list[dict[str, Any]] | str] = {}
        
        async with self._get_session() as session:
            for term_id in term_ids:
                # Query both directions of relationships
                result = await session.execute(
                    """
                    SELECT DISTINCT
                        r.object_term_id as term_id,
                        t.name,
                        t.definition,
                        r.relationship_type
                    FROM ontology_relationships r
                    JOIN ontology_terms t ON r.object_term_id = t.term_id 
                        AND t.ontology_type = r.ontology_type 
                        AND t.version = r.version
                    WHERE r.subject_term_id = :term_id
                      AND r.ontology_type = :ontology_type
                      AND r.version = :version
                    
                    UNION
                    
                    SELECT DISTINCT
                        r.subject_term_id as term_id,
                        t.name,
                        t.definition,
                        r.relationship_type || '_inverse' as relationship_type
                    FROM ontology_relationships r
                    JOIN ontology_terms t ON r.subject_term_id = t.term_id 
                        AND t.ontology_type = r.ontology_type 
                        AND t.version = r.version
                    WHERE r.object_term_id = :term_id
                      AND r.ontology_type = :ontology_type
                      AND r.version = :version
                    """,
                    {
                        "term_id": term_id,
                        "ontology_type": self.ONTOLOGY_TYPE,
                        "version": version,
                    }
                )
                rows = result.fetchall()
                
                neighbors = [
                    {
                        "term_id": row[0],
                        "name": row[1],
                        "definition": row[2],
                        "relationship_type": row[3],
                    }
                    for row in rows
                ]
                
                results[term_id] = neighbors if neighbors else []
        
        return results
    
    async def query_ols(
        self,
        search_terms: list[str],
        max_concurrent: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Query EBI Ontology Lookup Service for CL terms.
        
        Args:
            search_terms: List of search terms
            max_concurrent: Maximum concurrent API requests
        
        Returns:
            Dict mapping search terms to results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def query_single(term: str) -> tuple[str, list[dict[str, Any]]]:
            async with semaphore:
                try:
                    response = await self.http_client.get(
                        f"{self.OLS_BASE_URL}/search",
                        params={
                            "q": term,
                            "ontology": "cl",
                            "type": "class",
                            "local": "true",
                            "rows": 10,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    results = []
                    for doc in data.get("response", {}).get("docs", []):
                        obo_id = doc.get("obo_id", "")
                        if obo_id.startswith(f"{self.ONTOLOGY_PREFIX}:"):
                            results.append({
                                "term_id": obo_id,
                                "name": doc.get("label", ""),
                                "definition": doc.get("description", [""])[0] if doc.get("description") else None,
                            })
                    
                    return term, results
                    
                except Exception as e:
                    return term, []
        
        tasks = [query_single(term) for term in search_terms]
        results_list = await asyncio.gather(*tasks)
        
        return dict(results_list)
    
    async def get_term_by_id(self, term_id: str) -> Optional[dict[str, Any]]:
        """Get a single term by its CL ID."""
        version = await self.get_latest_version()
        
        async with self._get_session() as session:
            result = await session.execute(
                """
                SELECT term_id, name, definition
                FROM ontology_terms
                WHERE term_id = :term_id
                  AND ontology_type = :ontology_type
                  AND version = :version
                """,
                {
                    "term_id": term_id,
                    "ontology_type": self.ONTOLOGY_TYPE,
                    "version": version,
                }
            )
            row = result.fetchone()
            
            if row:
                return {
                    "term_id": row[0],
                    "name": row[1],
                    "definition": row[2],
                }
            return None
    
    async def get_lineage(
        self,
        term_id: str,
        max_depth: int = 5,
    ) -> list[str]:
        """
        Get the lineage (ancestor chain) for a term via is_a relationships.
        
        Args:
            term_id: CL term ID
            max_depth: Maximum depth to traverse
        
        Returns:
            List of ancestor term names from child to root
        """
        lineage = []
        current_id = term_id
        version = await self.get_latest_version()
        
        async with self._get_session() as session:
            for _ in range(max_depth):
                result = await session.execute(
                    """
                    SELECT 
                        r.object_term_id,
                        t.name
                    FROM ontology_relationships r
                    JOIN ontology_terms t ON r.object_term_id = t.term_id
                        AND t.ontology_type = r.ontology_type
                        AND t.version = r.version
                    WHERE r.subject_term_id = :term_id
                      AND r.relationship_type = 'is_a'
                      AND r.ontology_type = :ontology_type
                      AND r.version = :version
                    LIMIT 1
                    """,
                    {
                        "term_id": current_id,
                        "ontology_type": self.ONTOLOGY_TYPE,
                        "version": version,
                    }
                )
                row = result.fetchone()
                
                if row:
                    lineage.append(row[1])
                    current_id = row[0]
                else:
                    break
        
        return lineage
```

### 6.2 Query Understanding Integration

Update the Query Understanding subagent to use the new Cell Ontology tools:

```python
# orchestrator/agents/query_understanding.py

QUERY_UNDERSTANDING_TOOLS = [
    # Cell Ontology tools (primary for cell type resolution)
    resolve_cell_type_semantic,
    get_cell_type_neighbors,
    query_cell_ontology_ols,
    
    # Other entity resolution tools
    resolve_perturbation_tool,
    resolve_tissue_tool,
    resolve_disease_tool,
    
    # Knowledge gathering tools
    get_drug_targets_tool,
    get_pathway_priors_tool,
    get_cell_type_markers_tool,
    
    # Literature tools
    search_literature_tool,
    acquire_full_text_paper_tool,
]

QUERY_UNDERSTANDING_PROMPT = """
You are a biological query understanding agent. Your job is to:

1. DETERMINE TASK TYPE:
   - If query mentions drug/cytokine/perturbation effects → PERTURBATIONAL
   - If query asks to predict/impute cell types for a donor → OBSERVATIONAL
   - If query involves cross-dataset generation → HYBRID

2. EXTRACT AND RESOLVE CELL TYPES:
   For ALL tasks, you MUST resolve cell types to Cell Ontology IDs:
   
   a) Use resolve_cell_type_semantic to map free-text cell types to CL IDs
      - Example: "lung fibroblast" → CL:0002553
      - Example: "activated T cell" → CL:0000911
   
   b) If semantic search returns no matches or low confidence:
      - Try query_cell_ontology_ols as a fallback
      - Look for alternative phrasings in the results
   
   c) Capture synonyms from the resolution results for use in retrieval
   
   d) Use get_cell_type_neighbors to understand the cell type hierarchy:
      - Find parent types (more general) if exact match unavailable
      - Find child types (more specific) for refinement
      - Document the lineage for context

3. EXTRACT OTHER ENTITIES:
   For PERTURBATIONAL:
   - Perturbation name and type
   - Known targets and pathways
   
   For OBSERVATIONAL:
   - Target donor/sample identifier
   - Tissue type (UBERON)
   - Disease state (MONDO) if applicable
   - Reference dataset preferences

4. GATHER BIOLOGICAL CONTEXT:
   For PERTURBATIONAL: Drug targets, affected pathways
   For OBSERVATIONAL: Cell type markers, tissue signatures, disease-associated genes
   
   When targets or pathways are unclear:
   - Search literature for mechanism studies and reviews
   - Extract relevant findings from abstracts or full text
   - Use evidence to expand expected targets/pathways

Output a StructuredQuery with all resolved information.

Be thorough in resolving cell types - this information guides prompt selection.
The cell_type_cl_id field MUST be populated for successful retrieval.
"""
```

### 6.3 Prompt Generation Integration

Update the Ontology-Guided Strategy to use the new tools:

```python
# orchestrator/strategies/ontology_guided.py

class OntologyGuidedStrategy(RetrievalStrategy):
    """Find cells via Cell Ontology hierarchy using native CL tools."""
    
    @property
    def strategy_name(self) -> str:
        return "ontology"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellSetCandidate]:
        """
        Find cell sets with related cell types via Cell Ontology.
        
        Strategy:
        1. Use get_cell_type_neighbors to find parent/child/sibling types
        2. Search for cells of related types with query perturbation
        3. Score by ontology distance
        """
        if not query.cell_type_cl_id:
            return []
        
        candidates = []
        
        # Get related cell types via ontology service
        ontology_service = CellOntologyService.get_instance()
        neighbors_result = await ontology_service.get_neighbors(
            term_ids=[query.cell_type_cl_id]
        )
        
        neighbors = neighbors_result.get(query.cell_type_cl_id, [])
        if not neighbors or isinstance(neighbors, str):
            return []
        
        # Group neighbors by relationship type
        parents = [n for n in neighbors if n["relationship_type"] == "is_a"]
        children = [n for n in neighbors if n["relationship_type"] == "is_a_inverse"]
        related = [n for n in neighbors if n["relationship_type"] not in ("is_a", "is_a_inverse")]
        
        # Search for cells of related types (prioritize parents > children > other)
        for distance, neighbor_group in enumerate([parents, children, related], start=1):
            for neighbor in neighbor_group[:max_results // 3]:
                matches = await self._search_by_cell_type(
                    cell_type_cl_id=neighbor["term_id"],
                    perturbation_name=query.perturbation_resolved,
                    max_results=max_results // len(neighbor_group) if neighbor_group else max_results,
                    exclude_keys={c.selection_key() for c in candidates},
                )
                
                for match in matches:
                    match.relevance_score = 1.0 / (distance + 1)
                    match.rationale = (
                        f"Ontology {neighbor['relationship_type']}: "
                        f"{neighbor['name']} (distance={distance})"
                    )
                
                candidates.extend(matches)
        
        return candidates
```

---

## 7. Data Loading Pipeline

### 7.1 GCS Data Location

The Cell Ontology data is pre-computed and stored in GCS:

```
gs://haystack-data/ontology/
├── cell/
│   ├── 2025-01-01/
│   │   ├── terms.parquet          # Ontology terms with embeddings
│   │   ├── relationships.parquet  # Ontology relationships
│   │   └── metadata.json          # Version metadata
│   └── latest -> 2025-01-01       # Symlink to latest version
└── README.md
```

### 7.2 Data Format

**terms.parquet schema:**
```python
{
    "term_id": str,           # e.g., "CL:0000057"
    "name": str,              # e.g., "fibroblast"
    "definition": str | None, # Full definition text
    "embedding": list[float], # 1536-dimensional vector
    "ontology_type": str,     # "cell"
    "version": str,           # "2025-01-01"
}
```

**relationships.parquet schema:**
```python
{
    "subject_term_id": str,    # e.g., "CL:0000057"
    "object_term_id": str,     # e.g., "CL:0000548"
    "relationship_type": str,  # e.g., "is_a"
    "ontology_type": str,      # "cell"
    "version": str,            # "2025-01-01"
}
```

### 7.3 Loading Script

```python
# scripts/load_ontology.py
"""Load Cell Ontology data from GCS into Cloud SQL."""

import asyncio
import click
from google.cloud import storage
import pandas as pd

from haystack.orchestrator.services.database import get_database_engine
from haystack.shared.config import get_config


@click.command()
@click.option("--version", default="latest", help="Ontology version to load")
@click.option("--bucket", default="haystack-data", help="GCS bucket name")
@click.option("--replace", is_flag=True, help="Replace existing version")
@click.option("--dry-run", is_flag=True, help="Show what would be loaded")
def load_ontology(version: str, bucket: str, replace: bool, dry_run: bool):
    """Load Cell Ontology data from GCS into Cloud SQL."""
    asyncio.run(_load_ontology(version, bucket, replace, dry_run))


async def _load_ontology(
    version: str,
    bucket: str,
    replace: bool,
    dry_run: bool,
):
    """Async implementation of ontology loading."""
    config = get_config()
    
    # Resolve "latest" version
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    
    if version == "latest":
        # List versions and find most recent
        blobs = list(bucket_obj.list_blobs(prefix="ontology/cell/"))
        versions = sorted(set(
            b.name.split("/")[2] for b in blobs 
            if len(b.name.split("/")) > 2 and b.name.split("/")[2] != "latest"
        ))
        if not versions:
            raise click.ClickException("No ontology versions found in GCS")
        version = versions[-1]
    
    click.echo(f"Loading Cell Ontology version: {version}")
    
    # Download data
    terms_path = f"ontology/cell/{version}/terms.parquet"
    rels_path = f"ontology/cell/{version}/relationships.parquet"
    
    click.echo(f"Downloading {terms_path}...")
    terms_blob = bucket_obj.blob(terms_path)
    terms_df = pd.read_parquet(f"gs://{bucket}/{terms_path}")
    
    click.echo(f"Downloading {rels_path}...")
    rels_df = pd.read_parquet(f"gs://{bucket}/{rels_path}")
    
    click.echo(f"  Terms: {len(terms_df):,}")
    click.echo(f"  Relationships: {len(rels_df):,}")
    
    if dry_run:
        click.echo("Dry run - no data loaded")
        return
    
    # Load into database
    engine = await get_database_engine(config.database)
    
    async with engine.begin() as conn:
        # Delete existing version if replacing
        if replace:
            click.echo(f"Deleting existing version {version}...")
            await conn.execute(
                "DELETE FROM ontology_terms WHERE ontology_type = 'cell' AND version = :version",
                {"version": version}
            )
            await conn.execute(
                "DELETE FROM ontology_relationships WHERE ontology_type = 'cell' AND version = :version",
                {"version": version}
            )
        
        # Insert terms
        click.echo("Inserting terms...")
        for _, row in terms_df.iterrows():
            await conn.execute(
                """
                INSERT INTO ontology_terms 
                    (term_id, name, definition, embedding, ontology_type, version)
                VALUES 
                    (:term_id, :name, :definition, :embedding::vector, :ontology_type, :version)
                ON CONFLICT (term_id, ontology_type, version) DO UPDATE SET
                    name = EXCLUDED.name,
                    definition = EXCLUDED.definition,
                    embedding = EXCLUDED.embedding
                """,
                {
                    "term_id": row["term_id"],
                    "name": row["name"],
                    "definition": row.get("definition"),
                    "embedding": row["embedding"],
                    "ontology_type": "cell",
                    "version": version,
                }
            )
        
        # Insert relationships
        click.echo("Inserting relationships...")
        for _, row in rels_df.iterrows():
            await conn.execute(
                """
                INSERT INTO ontology_relationships 
                    (subject_term_id, object_term_id, relationship_type, ontology_type, version)
                VALUES 
                    (:subject, :object, :rel_type, :ontology_type, :version)
                ON CONFLICT DO NOTHING
                """,
                {
                    "subject": row["subject_term_id"],
                    "object": row["object_term_id"],
                    "rel_type": row["relationship_type"],
                    "ontology_type": "cell",
                    "version": version,
                }
            )
    
    click.echo(f"✓ Loaded Cell Ontology version {version}")


if __name__ == "__main__":
    load_ontology()
```

---

## 8. Configuration

### 8.1 Settings

```yaml
# settings.yml
default:
  ontology:
    # Cell Ontology settings
    cell:
      enabled: true
      embedding_model: "text-embedding-3-small"
      embedding_dimension: 1536
      default_k: 3
      default_distance_threshold: 0.7
    
    # OLS API settings
    ols:
      base_url: "https://www.ebi.ac.uk/ols4/api"
      request_timeout: 30
      max_concurrent_requests: 5
      retry_attempts: 3
    
    # GCS data location
    gcs:
      bucket: "haystack-data"
      prefix: "ontology"

development:
  ontology:
    cell:
      default_distance_threshold: 0.8  # More lenient in dev

production:
  ontology:
    ols:
      max_concurrent_requests: 10
```

### 8.2 Environment Variables

```bash
# OpenAI API for embeddings
OPENAI_API_KEY=sk-...

# GCS bucket (optional override)
ONTOLOGY_GCS_BUCKET=haystack-data
```

---

## 9. Error Handling

### 9.1 Exception Types

```python
# shared/exceptions.py

class OntologyError(HaystackError):
    """Base exception for ontology-related errors."""
    pass


class OntologyNotFoundError(OntologyError):
    """Raised when ontology data is not found in database."""
    pass


class OntologyResolutionError(OntologyError):
    """Raised when cell type resolution fails."""
    pass


class OLSAPIError(OntologyError):
    """Raised when OLS API request fails."""
    pass


class EmbeddingGenerationError(OntologyError):
    """Raised when embedding generation fails."""
    pass
```

### 9.2 Error Recovery

```python
async def resolve_cell_type_with_fallback(
    label: str,
    service: CellOntologyService,
) -> CellTypeResolution:
    """
    Resolve cell type with graceful fallback.
    
    1. Try semantic search (primary)
    2. Fall back to OLS if no semantic matches
    3. Return unresolved if all methods fail
    """
    # Try semantic search first
    try:
        results = await service.semantic_search(
            labels=[label],
            k=3,
            distance_threshold=0.7,
        )
        
        if results.get(label) and not isinstance(results[label], str):
            best = results[label][0]
            return CellTypeResolution(
                query_label=label,
                resolved=True,
                term_id=best["term_id"],
                term_name=best["name"],
                definition=best.get("definition"),
                confidence=1.0 - best["distance"],
                alternatives=results[label][1:] if len(results[label]) > 1 else [],
                method="semantic",
            )
    except Exception as e:
        logger.warning(f"Semantic search failed for '{label}': {e}")
    
    # Fall back to OLS
    try:
        ols_results = await service.query_ols(search_terms=[label])
        
        if ols_results.get(label):
            best = ols_results[label][0]
            return CellTypeResolution(
                query_label=label,
                resolved=True,
                term_id=best["term_id"],
                term_name=best["name"],
                definition=best.get("definition"),
                confidence=0.8,  # Lower confidence for OLS
                alternatives=ols_results[label][1:] if len(ols_results[label]) > 1 else [],
                method="ols",
            )
    except Exception as e:
        logger.warning(f"OLS query failed for '{label}': {e}")
    
    # No resolution found
    return CellTypeResolution(
        query_label=label,
        resolved=False,
        method="none",
    )
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/unit/test_ontology_service.py

import pytest
from unittest.mock import AsyncMock, patch

from haystack.orchestrator.services.ontology import CellOntologyService


class TestCellOntologyService:
    """Unit tests for Cell Ontology service."""
    
    @pytest.fixture
    def mock_service(self):
        """Create service with mocked dependencies."""
        service = CellOntologyService.__new__(CellOntologyService)
        service.session_factory = AsyncMock()
        service.openai_client = AsyncMock()
        service.http_client = AsyncMock()
        service._version_cache = "2025-01-01"
        return service
    
    @pytest.mark.asyncio
    async def test_semantic_search_returns_matches(self, mock_service):
        """Test semantic search returns expected results."""
        # Setup mock
        mock_session = AsyncMock()
        mock_session.execute.return_value.fetchall.return_value = [
            ("CL:0000057", "fibroblast", "A connective tissue cell...", 0.05),
        ]
        mock_service.session_factory.return_value.__aenter__.return_value = mock_session
        mock_service.openai_client.embeddings.create.return_value.data = [
            AsyncMock(embedding=[0.1] * 1536)
        ]
        
        # Execute
        results = await mock_service.semantic_search(
            labels=["fibroblast"],
            k=3,
            distance_threshold=0.7,
        )
        
        # Assert
        assert "fibroblast" in results
        assert results["fibroblast"][0]["term_id"] == "CL:0000057"
        assert results["fibroblast"][0]["distance"] == 0.05
    
    @pytest.mark.asyncio
    async def test_semantic_search_no_matches(self, mock_service):
        """Test semantic search with no matches above threshold."""
        mock_session = AsyncMock()
        mock_session.execute.return_value.fetchall.return_value = [
            ("CL:0000057", "fibroblast", "...", 0.9),  # Above threshold
        ]
        mock_service.session_factory.return_value.__aenter__.return_value = mock_session
        mock_service.openai_client.embeddings.create.return_value.data = [
            AsyncMock(embedding=[0.1] * 1536)
        ]
        
        results = await mock_service.semantic_search(
            labels=["unknown cell"],
            k=3,
            distance_threshold=0.7,
        )
        
        assert results["unknown cell"] == "No ontology ID found"
    
    @pytest.mark.asyncio
    async def test_get_neighbors_returns_related_terms(self, mock_service):
        """Test get_neighbors returns related terms."""
        mock_session = AsyncMock()
        mock_session.execute.return_value.fetchall.return_value = [
            ("CL:0000548", "animal cell", "A cell of an animal...", "is_a"),
        ]
        mock_service.session_factory.return_value.__aenter__.return_value = mock_session
        
        results = await mock_service.get_neighbors(term_ids=["CL:0000057"])
        
        assert "CL:0000057" in results
        assert len(results["CL:0000057"]) == 1
        assert results["CL:0000057"][0]["term_id"] == "CL:0000548"
        assert results["CL:0000057"][0]["relationship_type"] == "is_a"
```

### 10.2 Integration Tests

```python
# tests/integration/test_ontology_integration.py

import pytest

from haystack.orchestrator.services.ontology import CellOntologyService
from haystack.shared.config import get_config


@pytest.mark.integration
class TestCellOntologyIntegration:
    """Integration tests requiring database connection."""
    
    @pytest.fixture
    async def service(self):
        """Initialize real service."""
        config = get_config()
        return await CellOntologyService.initialize(config)
    
    @pytest.mark.asyncio
    async def test_semantic_search_real_database(self, service):
        """Test semantic search against real database."""
        results = await service.semantic_search(
            labels=["fibroblast", "T cell"],
            k=3,
            distance_threshold=0.7,
        )
        
        # Should find fibroblast
        assert "fibroblast" in results
        if isinstance(results["fibroblast"], list):
            assert any(r["term_id"] == "CL:0000057" for r in results["fibroblast"])
        
        # Should find T cell
        assert "T cell" in results
        if isinstance(results["T cell"], list):
            assert any("CL:0000084" in r["term_id"] for r in results["T cell"])
    
    @pytest.mark.asyncio
    async def test_ols_fallback(self, service):
        """Test OLS API fallback."""
        results = await service.query_ols(search_terms=["fibroblast"])
        
        assert "fibroblast" in results
        assert len(results["fibroblast"]) > 0
        assert results["fibroblast"][0]["term_id"].startswith("CL:")
```

---

## Related Specifications

- `specification/tools.md` - Complete tool specifications
- `specification/agents.md` - Agent system prompts and tool assignments
- `specification/database.md` - Database schema and configuration
- `specification/prompt-retrieval.md` - Ontology-guided retrieval strategy
- `specification/data-models.md` - Pydantic model definitions
