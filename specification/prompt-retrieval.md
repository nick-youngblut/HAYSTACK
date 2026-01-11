# HAYSTACK Prompt Cell Retrieval Specification

## Table of Contents

1. [Overview](#1-overview)
2. [Atlas Data Model](#2-atlas-data-model)
3. [Database Schema (GCP Cloud SQL)](#3-database-schema-gcp-cloud-sql)
4. [Retrieval Strategy Architecture](#4-retrieval-strategy-architecture)
5. [Direct Match Strategy](#5-direct-match-strategy)
6. [Mechanistic Match Strategy](#6-mechanistic-match-strategy)
7. [Semantic Match Strategy](#7-semantic-match-strategy)
8. [Ontology-Guided Strategy](#8-ontology-guided-strategy)
9. [Donor Context Strategy](#9-donor-context-strategy)
10. [Tissue Atlas Strategy](#10-tissue-atlas-strategy)
11. [Candidate Ranking and Selection](#11-candidate-ranking-and-selection)
12. [Index Building Pipeline](#12-index-building-pipeline)
13. [GCP Cloud SQL Configuration](#13-gcp-cloud-sql-configuration)
14. [Python Database Client](#14-python-database-client)
15. [Open Questions](#15-open-questions)

---

## 1. Overview

### 1.1 Problem Statement

HAYSTACK must select appropriate "prompt cells" for STACK's in-context learning from three heterogeneous atlases:
- **Parse PBMC**: ~10M cells, 90 cytokine perturbations, 12 donors
- **OpenProblems**: ~500K cells, 147 drug conditions, 3 donors
- **Tabula Sapiens**: ~500K cells, unperturbed, 25 tissues, 24 donors

Given a natural language query (e.g., "How would lung fibroblasts respond to TGF-beta?" or "Impute missing fibroblasts for donor A"), the system must:
1. Determine the ICL task type (perturbational, observational, hybrid)
2. Resolve the core biological entities (cell types, perturbations, donors, tissues)
3. Find the most biologically relevant prompt cells from available data
4. Handle cases where exact matches don't exist

### 1.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Database | GCP Cloud SQL (PostgreSQL + pgvector) | Full SQL for agent queries, HNSW for vectors, production-ready at 10M+ scale |
| Multiple embeddings per cell | Yes (text-based) | Enables semantic similarity search for perturbations, cell types, and sample context |
| Strategy chaining | Yes | Higher-level strategies (Mechanistic, Ontology, Donor Context) produce filters; lower-level strategies (Direct, Semantic, Tissue Atlas) retrieve cells |
| Metadata harmonization | Pre-indexed | Harmonization happens at index build time, not query time |
| Retrieval granularity | Cell groups | Return groups of cells sharing (perturbation, cell_type, donor), not individual cells |

### 1.3 Strategy Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL STRATEGY HIERARCHY                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              FILTER STRATEGIES (narrow candidates)              │    │
│  │                                                                 │    │
│  │   ┌─────────────────┐         ┌─────────────────┐               │    │
│  │   │  MECHANISTIC    │         │   ONTOLOGY      │               │    │
│  │   │  MATCH          │         │   GUIDED        │               │    │
│  │   │                 │         │                 │               │    │
│  │   │ Input: targets, │         │ Input: CL ID    │               │    │
│  │   │   pathways      │         │                 │               │    │
│  │   │ Output: list of │         │ Output: list of │               │    │
│  │   │   perturbations │         │   CL IDs        │               │    │
│  │   └────────┬────────┘         └────────┬────────┘               │    │
│  │            │                           │                        │    │
│  │            ▼                           ▼                        │    │
│  │        Perturbation              Cell Type                      │    │
│  │        Candidates                Candidates                     │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │             RETRIEVAL STRATEGIES (find cells)                   │    │
│  │                                                                 │    │
│  │   ┌─────────────────┐         ┌─────────────────┐               │    │
│  │   │  DIRECT         │         │   SEMANTIC      │               │    │
│  │   │  MATCH          │         │   (VECTOR)      │               │    │
│  │   │                 │         │                 │               │    │
│  │   │ Input: exact    │         │ Input: query    │               │    │
│  │   │   pert + cell   │         │   text          │               │    │
│  │   │   type          │         │ Output: similar │               │    │
│  │   │ Output: cell    │         │   cells by      │               │    │
│  │   │   groups        │         │   embedding     │               │    │
│  │   └────────┬────────┘         └────────┬────────┘               │    │
│  │            │                           │                        │    │
│  │            └───────────┬───────────────┘                        │    │
│  │                        ▼                                        │    │
│  │                 Cell Group                                      │    │
│  │                 Candidates                                      │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    CANDIDATE RANKING                            │    │
│  │                                                                 │    │
│  │   Score = w1*relevance + w2*diversity + w3*quality              │    │
│  │                                                                 │    │
│  │   → Select top K cell groups for prompt                         │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

For observational tasks, the retrieval layer adds two strategies:
- **DONOR_CONTEXT**: find donors with similar clinical context (tissue, disease, demographics)
- **TISSUE_ATLAS**: pull high-quality reference populations from large atlases

---

## 2. Atlas Data Model

### 2.1 Harmonized Cell Metadata

All three atlases are harmonized to a common metadata schema before indexing.

```python
from pydantic import BaseModel, Field
from typing import Optional


class HarmonizedCellMetadata(BaseModel):
    """Harmonized metadata for a single cell."""
    
    # Identifiers
    cell_index: int = Field(description="Index in source H5AD file")
    dataset: str = Field(description="Source dataset: parse_pbmc, openproblems, tabula_sapiens")
    
    # Cell type (harmonized to Cell Ontology)
    cell_type_original: str = Field(description="Original cell type annotation")
    cell_type_cl_id: Optional[str] = Field(description="Cell Ontology ID (e.g., CL:0000235)")
    cell_type_name: Optional[str] = Field(description="Canonical CL name")
    
    # Perturbation (harmonized)
    perturbation_original: Optional[str] = Field(description="Original perturbation annotation")
    perturbation_name: Optional[str] = Field(description="Harmonized perturbation name")
    perturbation_type: Optional[str] = Field(description="drug, cytokine, genetic, or None")
    is_control: bool = Field(default=False, description="Is this a control/unperturbed cell?")
    
    # External IDs (for linking to knowledge bases)
    perturbation_external_ids: dict[str, str] = Field(
        default_factory=dict,
        description="External IDs: drugbank, pubchem, chebi, uniprot, etc."
    )
    
    # Pre-computed biological knowledge
    perturbation_targets: list[str] = Field(
        default_factory=list,
        description="Known target genes"
    )
    perturbation_pathways: list[str] = Field(
        default_factory=list,
        description="Associated pathway IDs (KEGG, Reactome)"
    )
    
    # Tissue/organ (for Tabula Sapiens)
    tissue_original: Optional[str] = Field(description="Original tissue annotation")
    tissue_uberon_id: Optional[str] = Field(description="UBERON ID")
    tissue_name: Optional[str] = Field(description="Canonical tissue name")
    
    # Donor information
    donor_id: str = Field(description="Donor identifier (prefixed with dataset)")
    donor_age_category: Optional[str] = Field(description="young, middle-aged, elderly")
    donor_sex: Optional[str] = Field(description="Sex if available")
    
    # Quality metrics
    n_genes: int = Field(description="Number of detected genes")
    total_counts: float = Field(description="Total UMI counts")

    # Observational metadata
    disease_mondo_id: Optional[str] = Field(description="Disease MONDO ID")
    disease_name: Optional[str] = Field(description="Disease name")
    sample_condition: Optional[str] = Field(description="Sample condition (healthy, AKI, CKD)")
    sample_metadata: dict[str, str] = Field(default_factory=dict, description="Additional metadata")
```

### 2.2 Cell Groups

Cells are grouped for retrieval efficiency. A group contains cells that share the same (perturbation, cell_type, donor), with observational metadata (tissue, disease, condition) stored at the group level.

```python
class CellGroup(BaseModel):
    """A group of cells that can serve as a prompt unit."""
    
    group_id: str = Field(description="Unique group identifier")
    
    # Defining attributes (cells in group share these)
    dataset: str
    perturbation_name: Optional[str]
    cell_type_cl_id: Optional[str]
    donor_id: str
    tissue_uberon_id: Optional[str] = None
    tissue_name: Optional[str] = None
    disease_mondo_id: Optional[str] = None
    disease_name: Optional[str] = None
    sample_condition: Optional[str] = None
    sample_metadata: dict[str, str] = Field(default_factory=dict)
    is_reference_sample: bool = False
    
    # Group statistics
    n_cells: int
    cell_indices: list[int] = Field(description="Indices into atlas H5AD file")
    
    # Aggregated metadata
    mean_n_genes: float
    mean_total_counts: float
    
    # For retrieval
    has_control: bool = Field(description="Whether matching control cells exist")
    control_group_id: Optional[str] = Field(description="ID of corresponding control group")
```

### 2.3 Dataset-Specific Harmonization Rules

#### Parse PBMC
| Original Field | Harmonized Field | Transformation |
|----------------|------------------|----------------|
| `cell_type` | `cell_type_original` | Direct copy |
| `cell_type` | `cell_type_cl_id` | Lookup in mapping table |
| `stim` | `perturbation_original` | Direct copy |
| `stim` | `perturbation_name` | Normalize (e.g., "IFNg" → "IFN-gamma") |
| `donor` | `donor_id` | Prefix with "parse_" |
| N/A | `tissue_harmonized` | Set to "blood" |

#### OpenProblems
| Original Field | Harmonized Field | Transformation |
|----------------|------------------|----------------|
| `cell_type` | `cell_type_original` | Direct copy |
| `sm_name` | `perturbation_original` | Direct copy |
| `sm_name` | `perturbation_name` | PubChem/ChEMBL name resolution |
| `SMILES` | `perturbation_external_ids.smiles` | Direct copy |
| `donor_id` | `donor_id` | Prefix with "op_" |

#### Tabula Sapiens
| Original Field | Harmonized Field | Transformation |
|----------------|------------------|----------------|
| `cell_ontology_class` | `cell_type_original` | Direct copy |
| `cell_ontology_id` | `cell_type_cl_id` | Direct copy (already has CL IDs) |
| `tissue` | `tissue_original` | Direct copy |
| `donor` | `donor_id` | Prefix with "ts_" |
| N/A | `perturbation_name` | Set to None (unperturbed) |
| N/A | `is_control` | Set to True |

---

## 3. Database Schema (GCP Cloud SQL)

### 3.1 Overview

HAYSTACK uses **GCP Cloud SQL (PostgreSQL 15 + pgvector)** as its unified database, providing:
- Full SQL for flexible agent queries
- HNSW indexes for vector similarity search
- Proven scale to 10M+ vectors
- Read-only role enforcement for agents
- Managed infrastructure with automatic backups
- Private IP connectivity via VPC connector

### 3.2 Multi-Embedding Design

Each cell record stores multiple text embedding vectors:

| Column | Model | Dimension | Purpose |
|--------|-------|-----------|---------|
| `perturbation_embedding` | text-embedding-3-large | 1536 | Semantic perturbation search |
| `cell_type_embedding` | text-embedding-3-large | 1536 | Semantic cell type search |
| `sample_context_embedding` | text-embedding-3-large | 1536 | Donor/tissue/disease similarity |

> **Note**: STACK cell embeddings (transcriptomic state) are excluded from MVP. They would require running STACK inference to generate query embeddings, adding complexity. Text embeddings are sufficient for the retrieval strategies. Cell embeddings may be added in a future version for refinement iterations.

### 3.3 Text Embedding Generation

Text embeddings are generated from structured descriptions:

```python
def generate_perturbation_text(metadata: HarmonizedCellMetadata) -> str:
    """Generate text description for perturbation embedding."""
    if metadata.is_control:
        return "unperturbed control cell"
    
    parts = [metadata.perturbation_name]
    
    if metadata.perturbation_type:
        parts.append(f"({metadata.perturbation_type})")
    
    if metadata.perturbation_targets:
        targets = ", ".join(metadata.perturbation_targets[:5])
        parts.append(f"targeting {targets}")
    
    if metadata.perturbation_pathways:
        pathway_names = resolve_pathway_names(metadata.perturbation_pathways[:3])
        parts.append(f"affecting {', '.join(pathway_names)}")
    
    return " ".join(parts)
    # Example: "TGF-beta (cytokine) targeting TGFBR1, TGFBR2, SMAD2 affecting TGF-beta signaling, EMT"


def generate_cell_type_text(metadata: HarmonizedCellMetadata) -> str:
    """Generate text description for cell type embedding."""
    parts = [metadata.cell_type_name or metadata.cell_type_original]
    
    if metadata.tissue_original:
        parts.append(f"from {metadata.tissue_original}")
    
    # Add lineage information from Cell Ontology
    if metadata.cell_type_cl_id:
        lineage = get_cl_lineage(metadata.cell_type_cl_id)
        if lineage:
            parts.append(f"({' > '.join(lineage[:3])})")
    
    return " ".join(parts)
    # Example: "fibroblast from lung (mesenchymal cell > stromal cell > connective tissue cell)"


def generate_sample_context_text(metadata: HarmonizedCellMetadata) -> str:
    """Generate text description for sample context embedding."""
    parts = []
    
    if metadata.tissue_name or metadata.tissue_original:
        parts.append(f"tissue: {metadata.tissue_name or metadata.tissue_original}")
    
    if metadata.disease_name or metadata.disease_mondo_id:
        parts.append(f"disease: {metadata.disease_name or metadata.disease_mondo_id}")
    
    if metadata.sample_condition:
        parts.append(f"condition: {metadata.sample_condition}")
    
    if metadata.donor_age_category:
        parts.append(f"age: {metadata.donor_age_category}")
    
    if metadata.donor_sex:
        parts.append(f"sex: {metadata.donor_sex}")
    
    return "; ".join(parts) or "unknown clinical context"
```

### 3.4 Complete Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main cells table
CREATE TABLE cells (
    id SERIAL PRIMARY KEY,
    
    -- Identifiers
    cell_index INT NOT NULL,
    group_id VARCHAR(64) NOT NULL,
    dataset VARCHAR(32) NOT NULL,
    
    -- Cell type (harmonized)
    cell_type_original VARCHAR(256),
    cell_type_cl_id VARCHAR(32),
    cell_type_name VARCHAR(256),
    
    -- Perturbation (harmonized)
    perturbation_original VARCHAR(256),
    perturbation_name VARCHAR(256),
    perturbation_type VARCHAR(32),
    is_control BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Additional metadata
    tissue_original VARCHAR(256),
    tissue_uberon_id VARCHAR(32),
    tissue_name VARCHAR(256),
    donor_id VARCHAR(64),
    disease_mondo_id VARCHAR(32),
    disease_name VARCHAR(256),
    sample_condition VARCHAR(256),
    sample_metadata JSONB DEFAULT '{}',
    
    -- External IDs (JSONB for flexibility)
    perturbation_external_ids JSONB DEFAULT '{}',
    perturbation_targets TEXT[],
    perturbation_pathways TEXT[],
    
    -- Quality metrics
    n_genes INT,
    total_counts FLOAT,
    
    -- Text embeddings for semantic search (text-embedding-3-large, 1536 dim)
    perturbation_embedding vector(1536),
    cell_type_embedding vector(1536),
    sample_context_embedding vector(1536),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cell groups table (aggregated view for retrieval)
CREATE TABLE cell_groups (
    group_id VARCHAR(64) PRIMARY KEY,
    dataset VARCHAR(32) NOT NULL,
    perturbation_name VARCHAR(256),
    cell_type_cl_id VARCHAR(32),
    cell_type_name VARCHAR(256),
    donor_id VARCHAR(64),
    tissue_uberon_id VARCHAR(32),
    tissue_name VARCHAR(256),
    disease_mondo_id VARCHAR(32),
    disease_name VARCHAR(256),
    sample_condition VARCHAR(256),
    sample_metadata JSONB DEFAULT '{}',
    is_reference_sample BOOLEAN DEFAULT FALSE,
    
    n_cells INT NOT NULL,
    cell_indices INT[] NOT NULL,
    
    mean_n_genes FLOAT,
    mean_total_counts FLOAT,
    
    has_control BOOLEAN DEFAULT FALSE,
    control_group_id VARCHAR(64),
    
    -- Representative embeddings (mean of cells in group)
    perturbation_embedding vector(1536),
    cell_type_embedding vector(1536),
    sample_context_embedding vector(1536),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Donor lookup table for observational similarity search
CREATE TABLE donors (
    donor_id VARCHAR(64) PRIMARY KEY,
    dataset VARCHAR(32) NOT NULL,
    
    -- Demographics
    age_category VARCHAR(32),
    sex VARCHAR(16),
    
    -- Clinical metadata
    disease_states TEXT[],
    disease_names TEXT[],
    tissue_types TEXT[],
    
    -- Statistics
    n_cells INT,
    cell_types_present TEXT[],
    
    -- Embedding for donor similarity search
    clinical_embedding vector(1536)
);

-- Perturbation lookup table (for Mechanistic Match)
CREATE TABLE perturbations (
    perturbation_name VARCHAR(256) PRIMARY KEY,
    perturbation_type VARCHAR(32),
    external_ids JSONB DEFAULT '{}',
    targets TEXT[],
    pathways TEXT[],
    datasets_present TEXT[],
    total_cells INT,
    cell_types_present TEXT[]
);

-- Unified conditions table for perturbations and observational contexts
CREATE TABLE conditions (
    condition_id VARCHAR(64) PRIMARY KEY,
    condition_type VARCHAR(32) NOT NULL,  -- perturbation, disease, demographic
    
    -- Perturbations
    perturbation_name VARCHAR(256),
    perturbation_type VARCHAR(32),
    targets TEXT[],
    pathways TEXT[],
    
    -- Observational conditions
    disease_mondo_id VARCHAR(32),
    tissue_uberon_id VARCHAR(32),
    clinical_attributes JSONB DEFAULT '{}',
    
    -- Unified embedding
    condition_embedding vector(1536)
);

-- Cell type lookup table (for Ontology-Guided)
CREATE TABLE cell_types (
    cell_type_cl_id VARCHAR(32) PRIMARY KEY,
    cell_type_name VARCHAR(256) NOT NULL,
    lineage TEXT[],
    parent_cl_ids TEXT[],
    child_cl_ids TEXT[],
    datasets_present TEXT[],
    total_cells INT,
    perturbations_present TEXT[]
);

-- Synonym table for fuzzy matching (Direct Match)
CREATE TABLE synonyms (
    id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(256) NOT NULL,
    synonym VARCHAR(256) NOT NULL,
    entity_type VARCHAR(32) NOT NULL  -- 'perturbation' or 'cell_type'
);

-- Indexes for efficient querying
CREATE INDEX idx_cells_dataset ON cells(dataset);
CREATE INDEX idx_cells_cell_type ON cells(cell_type_cl_id);
CREATE INDEX idx_cells_perturbation ON cells(perturbation_name);
CREATE INDEX idx_cells_tissue ON cells(tissue_uberon_id);
CREATE INDEX idx_cells_is_control ON cells(is_control);
CREATE INDEX idx_cells_group ON cells(group_id);
CREATE INDEX idx_cells_donor ON cells(donor_id);
CREATE INDEX idx_cells_disease ON cells(disease_mondo_id);
CREATE INDEX idx_cells_condition ON cells(sample_condition);

-- HNSW vector indexes for similarity search
-- Note: Build these AFTER loading data for best performance
CREATE INDEX idx_cells_perturbation_embedding ON cells 
USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_cells_cell_type_embedding ON cells 
USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Group-level vector indexes
CREATE INDEX idx_groups_perturbation_embedding ON cell_groups
USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_groups_cell_type_embedding ON cell_groups
USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_groups_sample_context_embedding ON cell_groups
USING hnsw (sample_context_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index on synonym table
CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
CREATE INDEX idx_synonyms_synonym_lower ON synonyms(LOWER(synonym));
CREATE INDEX idx_synonyms_type ON synonyms(entity_type);

-- Index on lookup tables
CREATE INDEX idx_perturbations_type ON perturbations(perturbation_type);
CREATE INDEX idx_perturbations_targets ON perturbations USING GIN(targets);
CREATE INDEX idx_perturbations_pathways ON perturbations USING GIN(pathways);

CREATE INDEX idx_cell_types_parent ON cell_types USING GIN(parent_cl_ids);
CREATE INDEX idx_cell_types_child ON cell_types USING GIN(child_cl_ids);
CREATE INDEX idx_donors_dataset ON donors(dataset);
CREATE INDEX idx_conditions_type ON conditions(condition_type);
```

---

## 4. Retrieval Strategy Architecture

### 4.1 Strategy Interface

All retrieval strategies implement a common interface:

```python
from abc import ABC, abstractmethod
from typing import Optional


class RetrievalStrategy(ABC):
    """Base class for cell retrieval strategies."""
    
    def __init__(self, db: HaystackDatabase):
        """
        Initialize strategy with database connection.
        
        Args:
            db: Async database client
        """
        self.db = db
    
    @abstractmethod
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        """
        Retrieve cell group candidates for a query.
        
        Args:
            query: Parsed user query with resolved entities
            max_results: Maximum cell groups to return
            filters: Additional SQL filters
        
        Returns:
            List of cell group candidates with scores
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy name for logging."""
        pass


class CellGroupCandidate(BaseModel):
    """A candidate cell group with retrieval metadata."""
    
    group_id: str
    dataset: str
    perturbation_name: Optional[str]
    cell_type_cl_id: Optional[str]
    cell_type_name: Optional[str]
    n_cells: int
    
    # Strategy-specific scores
    strategy: str
    relevance_score: float = Field(ge=0, le=1)
    
    # Explanation
    rationale: str
    
    # Additional metadata for ranking
    has_control: bool = False
    control_group_id: Optional[str] = None
```

### 4.2 Strategy Orchestration

```python
class StrategyOrchestrator:
    """Orchestrates multiple retrieval strategies."""
    
    def __init__(self, db: HaystackDatabase):
        """
        Initialize orchestrator with all strategies.
        
        Args:
            db: Async database client
        """
        self.perturbational_strategies = [
            DirectMatchStrategy(db),
            MechanisticMatchStrategy(db),
            SemanticMatchStrategy(db),
            OntologyGuidedStrategy(db),
        ]
        self.observational_strategies = [
            DonorContextStrategy(db),
            TissueAtlasStrategy(db),
            SemanticMatchStrategy(db),
            OntologyGuidedStrategy(db),
        ]
        self.hybrid_strategies = [
            DirectMatchStrategy(db),
            MechanisticMatchStrategy(db),
            DonorContextStrategy(db),
            TissueAtlasStrategy(db),
            SemanticMatchStrategy(db),
            OntologyGuidedStrategy(db),
        ]
    
    async def retrieve_all(
        self,
        query: StructuredQuery,
        max_per_strategy: int = 20,
    ) -> list[CellGroupCandidate]:
        """
        Run all strategies in parallel and combine results.
        
        Args:
            query: Parsed user query
            max_per_strategy: Max results per strategy
        
        Returns:
            Combined and deduplicated candidates
        """
        import asyncio
        
        # Run strategies in parallel
        if query.task_type in [
            ICLTaskType.PERTURBATION_NOVEL_CELL_TYPES,
            ICLTaskType.PERTURBATION_NOVEL_SAMPLES,
        ]:
            strategies = self.perturbational_strategies
        elif query.task_type in [
            ICLTaskType.CELL_TYPE_IMPUTATION,
            ICLTaskType.DONOR_EXPRESSION_PREDICTION,
        ]:
            strategies = self.observational_strategies
        else:
            strategies = self.hybrid_strategies
        
        tasks = [
            strategy.retrieve(query, max_results=max_per_strategy)
            for strategy in strategies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine and deduplicate
        all_candidates = []
        seen_groups = set()
        
        for strategy, result in zip(strategies, results):
            if isinstance(result, Exception):
                logger.warning(f"Strategy {strategy.strategy_name} failed: {result}")
                continue
            
            for candidate in result:
                if candidate.group_id not in seen_groups:
                    all_candidates.append(candidate)
                    seen_groups.add(candidate.group_id)
        
        return all_candidates
```

---

## 5. Direct Match Strategy

### 5.1 Description

Direct Match finds cells that exactly match (or fuzzy match) the query's perturbation and cell type. This is the highest-precision strategy.

### 5.2 Implementation

```python
class DirectMatchStrategy(RetrievalStrategy):
    """Find exact or fuzzy matches for perturbation and cell type."""
    
    @property
    def strategy_name(self) -> str:
        return "direct"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        """
        Find cell groups matching query perturbation and cell type.
        
        Strategy:
        1. Try exact match on perturbation_name AND cell_type_cl_id
        2. If no results, try fuzzy match via synonyms
        3. If still no results, relax to partial matches
        """
        candidates = []
        
        # Step 1: Exact match
        if query.perturbation_resolved and query.cell_type_cl_id:
            exact_matches = await self._exact_match(
                perturbation=query.perturbation_resolved,
                cell_type_cl_id=query.cell_type_cl_id,
                max_results=max_results,
            )
            candidates.extend(exact_matches)
        
        # Step 2: Fuzzy match via synonyms
        if len(candidates) < max_results:
            fuzzy_matches = await self._fuzzy_match(
                perturbation_query=query.perturbation_query,
                cell_type_query=query.cell_type_query,
                max_results=max_results - len(candidates),
                exclude_groups={c.group_id for c in candidates},
            )
            candidates.extend(fuzzy_matches)
        
        # Step 3: Partial matches (perturbation OR cell type)
        if len(candidates) < max_results // 2:
            partial_matches = await self._partial_match(
                query=query,
                max_results=max_results - len(candidates),
                exclude_groups={c.group_id for c in candidates},
            )
            candidates.extend(partial_matches)
        
        return candidates
    
    async def _exact_match(
        self,
        perturbation: str,
        cell_type_cl_id: str,
        max_results: int,
    ) -> list[CellGroupCandidate]:
        """Find exact matches."""
        sql = """
            SELECT 
                g.group_id,
                g.dataset,
                g.perturbation_name,
                g.cell_type_cl_id,
                g.cell_type_name,
                g.n_cells,
                g.has_control,
                g.control_group_id
            FROM cell_groups g
            WHERE g.perturbation_name = $1
              AND g.cell_type_cl_id = $2
            ORDER BY g.n_cells DESC
            LIMIT $3
        """
        
        rows = await self.db.execute_query(sql, (perturbation, cell_type_cl_id, max_results))
        
        return [
            CellGroupCandidate(
                group_id=row["group_id"],
                dataset=row["dataset"],
                perturbation_name=row["perturbation_name"],
                cell_type_cl_id=row["cell_type_cl_id"],
                cell_type_name=row["cell_type_name"],
                n_cells=row["n_cells"],
                strategy="direct",
                relevance_score=1.0,
                rationale=f"Exact match: {perturbation} in {row['cell_type_name']}",
                has_control=row["has_control"],
                control_group_id=row["control_group_id"],
            )
            for row in rows
        ]
    
    async def _fuzzy_match(
        self,
        perturbation_query: str,
        cell_type_query: str,
        max_results: int,
        exclude_groups: set[str],
    ) -> list[CellGroupCandidate]:
        """Find matches via synonym lookup."""
        # Resolve synonyms
        pert_canonical = await self._resolve_synonym(perturbation_query, "perturbation")
        ct_canonical = await self._resolve_synonym(cell_type_query, "cell_type")
        
        if not pert_canonical and not ct_canonical:
            return []
        
        sql = """
            SELECT 
                g.group_id,
                g.dataset,
                g.perturbation_name,
                g.cell_type_cl_id,
                g.cell_type_name,
                g.n_cells,
                g.has_control,
                g.control_group_id
            FROM cell_groups g
            WHERE ($1::text IS NULL OR g.perturbation_name = $1)
              AND ($2::text IS NULL OR g.cell_type_cl_id = $2)
              AND g.group_id != ALL($3)
            ORDER BY g.n_cells DESC
            LIMIT $4
        """
        
        rows = await self.db.execute_query(
            sql, 
            (pert_canonical, ct_canonical, list(exclude_groups), max_results)
        )
        
        return [
            CellGroupCandidate(
                group_id=row["group_id"],
                dataset=row["dataset"],
                perturbation_name=row["perturbation_name"],
                cell_type_cl_id=row["cell_type_cl_id"],
                cell_type_name=row["cell_type_name"],
                n_cells=row["n_cells"],
                strategy="direct",
                relevance_score=0.9,
                rationale=f"Fuzzy match via synonym: {perturbation_query} → {pert_canonical}",
                has_control=row["has_control"],
                control_group_id=row["control_group_id"],
            )
            for row in rows
        ]
    
    async def _resolve_synonym(
        self,
        query: str,
        entity_type: str,
    ) -> Optional[str]:
        """Resolve a query string to its canonical name via synonyms."""
        sql = """
            SELECT canonical_name
            FROM synonyms
            WHERE LOWER(synonym) = LOWER($1)
              AND entity_type = $2
            LIMIT 1
        """
        rows = await self.db.execute_query(sql, (query, entity_type))
        return rows[0]["canonical_name"] if rows else None
```

---

## 6. Mechanistic Match Strategy

### 6.1 Description

Mechanistic Match finds perturbations that share biological targets or pathways with the query perturbation. Even if the exact perturbation isn't in the atlas, cells perturbed with mechanistically similar agents may serve as good prompts.

### 6.2 Implementation

```python
class MechanisticMatchStrategy(RetrievalStrategy):
    """Find cells with mechanistically similar perturbations."""
    
    @property
    def strategy_name(self) -> str:
        return "mechanistic"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        """
        Find cell groups with perturbations sharing targets/pathways.
        
        Strategy:
        1. Get expected targets and pathways from query
        2. Find perturbations in atlas that share targets
        3. Find perturbations in atlas that share pathways
        4. Score by overlap and retrieve cell groups
        """
        if not query.expected_targets and not query.expected_pathways:
            return []
        
        candidates = []
        
        # Find perturbations by target overlap
        if query.expected_targets:
            target_matches = await self._find_by_targets(
                targets=query.expected_targets,
                cell_type_cl_id=query.cell_type_cl_id,
                max_results=max_results // 2,
            )
            candidates.extend(target_matches)
        
        # Find perturbations by pathway overlap
        if query.expected_pathways:
            pathway_matches = await self._find_by_pathways(
                pathways=query.expected_pathways,
                cell_type_cl_id=query.cell_type_cl_id,
                max_results=max_results // 2,
                exclude_groups={c.group_id for c in candidates},
            )
            candidates.extend(pathway_matches)
        
        return candidates
    
    async def _find_by_targets(
        self,
        targets: list[str],
        cell_type_cl_id: Optional[str],
        max_results: int,
    ) -> list[CellGroupCandidate]:
        """Find perturbations sharing target genes."""
        sql = """
            WITH target_overlap AS (
                SELECT 
                    p.perturbation_name,
                    p.perturbation_type,
                    p.targets,
                    CARDINALITY(
                        ARRAY(SELECT UNNEST(p.targets) INTERSECT SELECT UNNEST($1::text[]))
                    ) as overlap_count,
                    CARDINALITY(p.targets) as total_targets
                FROM perturbations p
                WHERE p.targets && $1::text[]
            )
            SELECT 
                g.group_id,
                g.dataset,
                g.perturbation_name,
                g.cell_type_cl_id,
                g.cell_type_name,
                g.n_cells,
                g.has_control,
                g.control_group_id,
                t.overlap_count,
                t.total_targets
            FROM cell_groups g
            JOIN target_overlap t ON g.perturbation_name = t.perturbation_name
            WHERE ($2::text IS NULL OR g.cell_type_cl_id = $2)
            ORDER BY t.overlap_count DESC, g.n_cells DESC
            LIMIT $3
        """
        
        rows = await self.db.execute_query(sql, (targets, cell_type_cl_id, max_results))
        
        return [
            CellGroupCandidate(
                group_id=row["group_id"],
                dataset=row["dataset"],
                perturbation_name=row["perturbation_name"],
                cell_type_cl_id=row["cell_type_cl_id"],
                cell_type_name=row["cell_type_name"],
                n_cells=row["n_cells"],
                strategy="mechanistic",
                relevance_score=min(1.0, row["overlap_count"] / len(targets)),
                rationale=f"Shares {row['overlap_count']}/{row['total_targets']} targets",
                has_control=row["has_control"],
                control_group_id=row["control_group_id"],
            )
            for row in rows
        ]
    
    async def _find_by_pathways(
        self,
        pathways: list[str],
        cell_type_cl_id: Optional[str],
        max_results: int,
        exclude_groups: set[str],
    ) -> list[CellGroupCandidate]:
        """Find perturbations sharing pathway annotations."""
        sql = """
            WITH pathway_overlap AS (
                SELECT 
                    p.perturbation_name,
                    CARDINALITY(
                        ARRAY(SELECT UNNEST(p.pathways) INTERSECT SELECT UNNEST($1::text[]))
                    ) as overlap_count
                FROM perturbations p
                WHERE p.pathways && $1::text[]
            )
            SELECT 
                g.group_id,
                g.dataset,
                g.perturbation_name,
                g.cell_type_cl_id,
                g.cell_type_name,
                g.n_cells,
                g.has_control,
                g.control_group_id,
                po.overlap_count
            FROM cell_groups g
            JOIN pathway_overlap po ON g.perturbation_name = po.perturbation_name
            WHERE ($2::text IS NULL OR g.cell_type_cl_id = $2)
              AND g.group_id != ALL($3)
            ORDER BY po.overlap_count DESC, g.n_cells DESC
            LIMIT $4
        """
        
        rows = await self.db.execute_query(
            sql, 
            (pathways, cell_type_cl_id, list(exclude_groups), max_results)
        )
        
        return [
            CellGroupCandidate(
                group_id=row["group_id"],
                dataset=row["dataset"],
                perturbation_name=row["perturbation_name"],
                cell_type_cl_id=row["cell_type_cl_id"],
                cell_type_name=row["cell_type_name"],
                n_cells=row["n_cells"],
                strategy="mechanistic",
                relevance_score=min(1.0, row["overlap_count"] / len(pathways)),
                rationale=f"Shares {row['overlap_count']}/{len(pathways)} pathways",
                has_control=row["has_control"],
                control_group_id=row["control_group_id"],
            )
            for row in rows
        ]
```

---

## 7. Semantic Match Strategy

### 7.1 Description

Semantic Match uses vector similarity search to find cells whose perturbations, cell types, or sample contexts are semantically similar to the query, even if they don't share explicit targets or ontology relationships. For observational tasks, it uses the sample context embedding to match tissue/disease/condition descriptions.

### 7.2 Implementation

```python
class SemanticMatchStrategy(RetrievalStrategy):
    """Find cells via vector similarity search."""
    
    def __init__(self, db: HaystackDatabase, embedding_client: EmbeddingClient):
        """
        Initialize with database and embedding client.
        
        Args:
            db: Async database client
            embedding_client: Client for generating query embeddings
        """
        super().__init__(db)
        self.embedding_client = embedding_client
    
    @property
    def strategy_name(self) -> str:
        return "semantic"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        """
        Find cell groups via vector similarity.
        
        Strategy:
        1. Generate embedding for perturbation query (perturbational tasks)
        2. Generate embedding for cell type query
        3. Generate embedding for sample context (observational tasks)
        4. Search for similar perturbations, cell types, and contexts
        5. Combine with weighted scores
        """
        candidates = []
        
        # Search by perturbation similarity
        if query.perturbation_query:
            pert_embedding = await self.embedding_client.embed_text(
                generate_perturbation_query_text(query)
            )
            pert_matches = await self._vector_search(
                embedding=pert_embedding,
                search_type="perturbation",
                cell_type_filter=query.cell_type_cl_id,
                max_results=max_results // 2,
            )
            candidates.extend(pert_matches)
        
        # Search by cell type similarity
        if query.cell_type_query:
            ct_embedding = await self.embedding_client.embed_text(
                generate_cell_type_query_text(query)
            )
            ct_matches = await self._vector_search(
                embedding=ct_embedding,
                search_type="cell_type",
                perturbation_filter=query.perturbation_resolved,
                max_results=max_results // 2,
                exclude_groups={c.group_id for c in candidates},
            )
            candidates.extend(ct_matches)
        
        # Search by sample context similarity (observational)
        if query.task_type in [
            ICLTaskType.CELL_TYPE_IMPUTATION,
            ICLTaskType.DONOR_EXPRESSION_PREDICTION,
        ]:
            context_embedding = await self.embedding_client.embed_text(
                generate_sample_context_query_text(query)
            )
            ctx_matches = await self._vector_search(
                embedding=context_embedding,
                search_type="sample_context",
                cell_type_filter=query.cell_type_cl_id,
                max_results=max_results // 2,
                exclude_groups={c.group_id for c in candidates},
            )
            candidates.extend(ctx_matches)
        
        return candidates
    
    async def _vector_search(
        self,
        embedding: list[float],
        search_type: str,
        cell_type_filter: Optional[str] = None,
        perturbation_filter: Optional[str] = None,
        max_results: int = 25,
        exclude_groups: Optional[set[str]] = None,
    ) -> list[CellGroupCandidate]:
        """Perform vector similarity search."""
        embedding_col = f"{search_type}_embedding"
        exclude_groups = exclude_groups or set()
        
        sql = f"""
            SELECT 
                g.group_id,
                g.dataset,
                g.perturbation_name,
                g.cell_type_cl_id,
                g.cell_type_name,
                g.n_cells,
                g.has_control,
                g.control_group_id,
                1 - (g.{embedding_col} <=> $1::vector) as similarity
            FROM cell_groups g
            WHERE g.{embedding_col} IS NOT NULL
              AND ($2::text IS NULL OR g.cell_type_cl_id = $2)
              AND ($3::text IS NULL OR g.perturbation_name = $3)
              AND g.group_id != ALL($4)
            ORDER BY g.{embedding_col} <=> $1::vector
            LIMIT $5
        """
        
        rows = await self.db.execute_query(
            sql,
            (embedding, cell_type_filter, perturbation_filter, 
             list(exclude_groups), max_results)
        )
        
        return [
            CellGroupCandidate(
                group_id=row["group_id"],
                dataset=row["dataset"],
                perturbation_name=row["perturbation_name"],
                cell_type_cl_id=row["cell_type_cl_id"],
                cell_type_name=row["cell_type_name"],
                n_cells=row["n_cells"],
                strategy="semantic",
                relevance_score=row["similarity"],
                rationale=f"Semantic similarity ({search_type}): {row['similarity']:.3f}",
                has_control=row["has_control"],
                control_group_id=row["control_group_id"],
            )
            for row in rows
            if row["similarity"] >= 0.5  # Minimum threshold
        ]


def generate_perturbation_query_text(query: StructuredQuery) -> str:
    """Generate query text for perturbation embedding."""
    parts = [query.perturbation_query]
    
    if query.perturbation_type != PerturbationType.UNKNOWN:
        parts.append(f"({query.perturbation_type.value})")
    
    if query.expected_targets:
        parts.append(f"targeting {', '.join(query.expected_targets[:5])}")
    
    return " ".join(parts)


def generate_cell_type_query_text(query: StructuredQuery) -> str:
    """Generate query text for cell type embedding."""
    return query.cell_type_query
```

---

## 8. Ontology-Guided Strategy

### 8.1 Description

Ontology-Guided Strategy uses the Cell Ontology hierarchy to find related cell types. If the exact cell type isn't available, it searches for parent or child cell types in the ontology.

### 8.2 Implementation

```python
class OntologyGuidedStrategy(RetrievalStrategy):
    """Find cells via Cell Ontology hierarchy."""
    
    @property
    def strategy_name(self) -> str:
        return "ontology"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        """
        Find cell groups with related cell types via ontology.
        
        Strategy:
        1. Get parent cell types (more general)
        2. Get child cell types (more specific)
        3. Get sibling cell types (same parent)
        4. Search for cells of related types with query perturbation
        """
        if not query.cell_type_cl_id:
            return []
        
        candidates = []
        
        # Get related cell types
        related_types = await self._get_related_cell_types(
            cl_id=query.cell_type_cl_id,
            max_distance=2,
        )
        
        if not related_types:
            return []
        
        # Search for cells of related types
        for related_cl_id, distance, relationship in related_types:
            matches = await self._search_by_cell_type(
                cell_type_cl_id=related_cl_id,
                perturbation_name=query.perturbation_resolved,
                max_results=max_results // len(related_types),
                exclude_groups={c.group_id for c in candidates},
            )
            
            for match in matches:
                match.relevance_score = 1.0 / (distance + 1)
                match.rationale = f"Ontology {relationship}: distance={distance}"
            
            candidates.extend(matches)
        
        return candidates
    
    async def _get_related_cell_types(
        self,
        cl_id: str,
        max_distance: int = 2,
    ) -> list[tuple[str, int, str]]:
        """
        Get related cell types from ontology.
        
        Returns:
            List of (cl_id, distance, relationship_type) tuples
        """
        sql = """
            WITH RECURSIVE related AS (
                -- Parents (distance 1)
                SELECT 
                    UNNEST(parent_cl_ids) as cl_id,
                    1 as distance,
                    'parent' as relationship
                FROM cell_types
                WHERE cell_type_cl_id = $1
                
                UNION
                
                -- Children (distance 1)
                SELECT 
                    UNNEST(child_cl_ids) as cl_id,
                    1 as distance,
                    'child' as relationship
                FROM cell_types
                WHERE cell_type_cl_id = $1
                
                UNION
                
                -- Grandparents/grandchildren (distance 2)
                SELECT 
                    UNNEST(ct.parent_cl_ids) as cl_id,
                    r.distance + 1 as distance,
                    'ancestor' as relationship
                FROM related r
                JOIN cell_types ct ON ct.cell_type_cl_id = r.cl_id
                WHERE r.distance < $2 AND r.relationship IN ('parent', 'ancestor')
                
                UNION
                
                SELECT 
                    UNNEST(ct.child_cl_ids) as cl_id,
                    r.distance + 1 as distance,
                    'descendant' as relationship
                FROM related r
                JOIN cell_types ct ON ct.cell_type_cl_id = r.cl_id
                WHERE r.distance < $2 AND r.relationship IN ('child', 'descendant')
            )
            SELECT DISTINCT cl_id, MIN(distance) as distance, relationship
            FROM related
            WHERE cl_id IN (SELECT cell_type_cl_id FROM cell_types WHERE total_cells > 0)
            GROUP BY cl_id, relationship
            ORDER BY distance
        """
        
        rows = await self.db.execute_query(sql, (cl_id, max_distance))
        return [(row["cl_id"], row["distance"], row["relationship"]) for row in rows]
    
    async def _search_by_cell_type(
        self,
        cell_type_cl_id: str,
        perturbation_name: Optional[str],
        max_results: int,
        exclude_groups: set[str],
    ) -> list[CellGroupCandidate]:
        """Search for cell groups by cell type."""
        sql = """
            SELECT 
                g.group_id,
                g.dataset,
                g.perturbation_name,
                g.cell_type_cl_id,
                g.cell_type_name,
                g.n_cells,
                g.has_control,
                g.control_group_id
            FROM cell_groups g
            WHERE g.cell_type_cl_id = $1
              AND ($2::text IS NULL OR g.perturbation_name = $2)
              AND g.group_id != ALL($3)
            ORDER BY g.n_cells DESC
            LIMIT $4
        """
        
        rows = await self.db.execute_query(
            sql,
            (cell_type_cl_id, perturbation_name, list(exclude_groups), max_results)
        )
        
        return [
            CellGroupCandidate(
                group_id=row["group_id"],
                dataset=row["dataset"],
                perturbation_name=row["perturbation_name"],
                cell_type_cl_id=row["cell_type_cl_id"],
                cell_type_name=row["cell_type_name"],
                n_cells=row["n_cells"],
                strategy="ontology",
                relevance_score=0.0,  # Set by caller
                rationale="",  # Set by caller
                has_control=row["has_control"],
                control_group_id=row["control_group_id"],
            )
            for row in rows
        ]
```

---

## 9. Donor Context Strategy

### 9.1 Purpose

The Donor Context strategy selects reference cell groups from donors with similar clinical context to the target donor for observational ICL tasks. It prioritizes matching tissue, disease state, and donor demographics before falling back to embedding similarity.

### 9.2 Retrieval Logic

```python
class DonorContextStrategy(RetrievalStrategy):
    """Find cells from donors with similar clinical context."""
    
    @property
    def strategy_name(self) -> str:
        return "donor_context"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        context_text = self._build_context_text(query)
        embedding = await self.embedder.embed(context_text)
        
        sql = """
            SELECT 
                g.group_id,
                g.dataset,
                g.cell_type_name,
                g.cell_type_cl_id,
                g.donor_id,
                g.tissue_name,
                g.disease_name,
                g.n_cells,
                1 - (d.clinical_embedding <=> $1::vector) as donor_similarity
            FROM cell_groups g
            JOIN donors d ON g.donor_id = d.donor_id
            WHERE g.cell_type_cl_id = $2
              AND g.donor_id != $3
              AND ($4::text IS NULL OR g.tissue_uberon_id = $4)
              AND ($5::text IS NULL OR g.disease_mondo_id = $5)
            ORDER BY d.clinical_embedding <=> $1::vector
            LIMIT $6
        """
        ...
```

## 10. Tissue Atlas Strategy

### 10.1 Purpose

The Tissue Atlas strategy selects high-quality reference cell populations from curated atlases for observational ICL tasks. It prioritizes comprehensive atlases (Tabula Sapiens) and tissue-specific datasets with large cell counts.

### 10.2 Retrieval Logic

```python
class TissueAtlasStrategy(RetrievalStrategy):
    """Find cells from comprehensive tissue atlases."""
    
    @property
    def strategy_name(self) -> str:
        return "tissue_atlas"
    
    async def retrieve(
        self,
        query: StructuredQuery,
        max_results: int = 50,
        filters: Optional[dict] = None,
    ) -> list[CellGroupCandidate]:
        sql = """
            SELECT 
                g.group_id,
                g.dataset,
                g.cell_type_name,
                g.cell_type_cl_id,
                g.donor_id,
                g.tissue_name,
                g.n_cells,
                g.mean_n_genes
            FROM cell_groups g
            WHERE g.cell_type_cl_id = $1
              AND g.is_reference_sample = TRUE
              AND ($2::text IS NULL OR g.tissue_uberon_id = $2)
            ORDER BY 
                CASE g.dataset 
                    WHEN 'tabula_sapiens' THEN 1 
                    ELSE 2 
                END,
                g.n_cells DESC
            LIMIT $3
        """
        ...
```

## 11. Candidate Ranking and Selection

### 9.1 Ranking Algorithm

```python
class CandidateRanker:
    """Rank and select final prompt candidates."""
    
    def __init__(
        self,
        relevance_weight: float = 0.4,
        diversity_weight: float = 0.3,
        quality_weight: float = 0.3,
    ):
        """
        Initialize ranker with scoring weights.
        
        Args:
            relevance_weight: Weight for relevance score
            diversity_weight: Weight for diversity contribution
            quality_weight: Weight for data quality
        """
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
    
    def rank_candidates(
        self,
        candidates: list[CellGroupCandidate],
        top_k: int = 10,
    ) -> list[CellGroupCandidate]:
        """
        Rank candidates and select top K.
        
        Scoring factors:
        1. Relevance: Strategy-specific relevance score
        2. Diversity: Penalize redundant selections
        3. Quality: Prefer groups with more cells and control pairs
        
        Args:
            candidates: All candidates from strategies
            top_k: Number to select
        
        Returns:
            Top K candidates with final scores
        """
        if not candidates:
            return []
        
        # Compute quality scores
        max_cells = max(c.n_cells for c in candidates)
        for c in candidates:
            c._quality_score = self._compute_quality_score(c, max_cells)
        
        # Greedy selection with diversity
        selected = []
        remaining = list(candidates)
        
        while len(selected) < top_k and remaining:
            # Score remaining candidates
            for c in remaining:
                c._final_score = self._compute_final_score(c, selected)
            
            # Select best
            remaining.sort(key=lambda c: c._final_score, reverse=True)
            selected.append(remaining.pop(0))
        
        return selected
    
    def _compute_quality_score(
        self,
        candidate: CellGroupCandidate,
        max_cells: int,
    ) -> float:
        """Compute quality score for a candidate."""
        # Cell count (normalized)
        cell_score = candidate.n_cells / max_cells
        
        # Has control bonus
        control_bonus = 0.2 if candidate.has_control else 0.0
        
        return cell_score * 0.8 + control_bonus
    
    def _compute_final_score(
        self,
        candidate: CellGroupCandidate,
        selected: list[CellGroupCandidate],
    ) -> float:
        """Compute final score including diversity."""
        # Relevance
        relevance = candidate.relevance_score
        
        # Diversity (penalize if similar to already selected)
        diversity = self._compute_diversity(candidate, selected)
        
        # Quality
        quality = candidate._quality_score
        
        return (
            self.relevance_weight * relevance +
            self.diversity_weight * diversity +
            self.quality_weight * quality
        )
    
    def _compute_diversity(
        self,
        candidate: CellGroupCandidate,
        selected: list[CellGroupCandidate],
    ) -> float:
        """Compute diversity contribution."""
        if not selected:
            return 1.0
        
        # Penalize same perturbation
        same_pert = sum(
            1 for s in selected 
            if s.perturbation_name == candidate.perturbation_name
        )
        
        # Penalize same cell type
        same_ct = sum(
            1 for s in selected 
            if s.cell_type_cl_id == candidate.cell_type_cl_id
        )
        
        # Penalize same dataset
        same_dataset = sum(
            1 for s in selected 
            if s.dataset == candidate.dataset
        )
        
        # Diversity score
        diversity = 1.0
        diversity -= 0.3 * (same_pert / len(selected))
        diversity -= 0.2 * (same_ct / len(selected))
        diversity -= 0.1 * (same_dataset / len(selected))
        
        return max(0.0, diversity)
```

---

## 12. Index Building Pipeline

### 10.1 Overview

The index is built once, offline, before HAYSTACK can run queries. This involves:

1. Loading and harmonizing all three atlases
2. Generating text descriptions for perturbations and cell types
3. Computing text embeddings via OpenAI API
4. Loading data into Cloud SQL (PostgreSQL + pgvector)
5. Building HNSW indexes on vector columns
6. Building auxiliary lookup tables

### 10.2 Pipeline Implementation

```python
import asyncio
from typing import Iterator
import scanpy as sc
import numpy as np


async def build_haystack_index(
    parse_pbmc_path: str,
    openproblems_path: str,
    tabula_sapiens_path: str,
    db_connection_string: str,
    text_embedding_model: str = "text-embedding-3-large",
    batch_size: int = 1000,
):
    """
    Build complete HAYSTACK PostgreSQL database.
    
    Estimated time: 2-4 hours depending on text embedding API throughput.
    
    Args:
        parse_pbmc_path: Path to Parse PBMC H5AD (local or GCS)
        openproblems_path: Path to OpenProblems H5AD
        tabula_sapiens_path: Path to Tabula Sapiens H5AD
        db_connection_string: Cloud SQL connection string
        text_embedding_model: Text embedding model name
        batch_size: Batch size for embedding generation
    """
    # Initialize database
    db = HaystackDatabase(db_connection_string)
    await db.connect()
    
    # Initialize embedding client
    embedding_client = EmbeddingClient(model=text_embedding_model)
    
    try:
        # Step 1: Load and harmonize metadata
        print("Step 1: Harmonizing atlas metadata...")
        parse_metadata = await harmonize_parse_pbmc(parse_pbmc_path)
        op_metadata = await harmonize_openproblems(openproblems_path)
        ts_metadata = await harmonize_tabula_sapiens(tabula_sapiens_path)
        
        all_metadata = parse_metadata + op_metadata + ts_metadata
        print(f"  Total cells: {len(all_metadata):,}")
        
        # Step 2: Build cell groups
        print("Step 2: Building cell groups...")
        cell_groups = build_cell_groups(all_metadata)
        print(f"  Total groups: {len(cell_groups):,}")
        
        # Step 3: Generate text descriptions
        print("Step 3: Generating text descriptions...")
        perturbation_texts = [generate_perturbation_text(m) for m in all_metadata]
        cell_type_texts = [generate_cell_type_text(m) for m in all_metadata]
        sample_context_texts = [generate_sample_context_text(m) for m in all_metadata]
        
        # Step 4: Compute text embeddings (batched)
        print("Step 4: Computing text embeddings...")
        perturbation_embeddings = await batch_embed_texts(
            embedding_client, perturbation_texts, batch_size
        )
        cell_type_embeddings = await batch_embed_texts(
            embedding_client, cell_type_texts, batch_size
        )
        sample_context_embeddings = await batch_embed_texts(
            embedding_client, sample_context_texts, batch_size
        )
        
        # Step 5: Load into PostgreSQL
        print("Step 5: Loading into PostgreSQL...")
        await load_cells_to_postgres(
            db,
            metadata=all_metadata,
            perturbation_embeddings=perturbation_embeddings,
            cell_type_embeddings=cell_type_embeddings,
            sample_context_embeddings=sample_context_embeddings,
        )
        
        # Step 6: Load cell groups
        print("Step 6: Loading cell groups...")
        await load_groups_to_postgres(
            db,
            cell_groups,
            perturbation_embeddings,
            cell_type_embeddings,
            sample_context_embeddings,
        )
        
        # Step 7: Build HNSW indexes
        print("Step 7: Building HNSW indexes...")
        await build_hnsw_indexes(db)
        
        # Step 8: Build auxiliary tables
        print("Step 8: Building auxiliary tables...")
        await build_perturbation_lookup_table(all_metadata, db)
        await build_cell_type_lookup_table(all_metadata, db)
        await build_synonym_table(db)
        
        print("Done!")
        
    finally:
        await db.close()


async def batch_embed_texts(
    client: EmbeddingClient,
    texts: list[str],
    batch_size: int = 1000,
) -> list[list[float]]:
    """Embed texts in batches with rate limiting."""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await client.embed_batch(batch)
        embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % 10000 == 0:
            print(f"  Embedded {i + batch_size:,} / {len(texts):,} texts")
        
        # Rate limiting
        await asyncio.sleep(0.1)
    
    return embeddings


async def load_cells_to_postgres(
    db: HaystackDatabase,
    metadata: list[HarmonizedCellMetadata],
    perturbation_embeddings: list[list[float]],
    cell_type_embeddings: list[list[float]],
    sample_context_embeddings: list[list[float]],
    batch_size: int = 5000,
):
    """Load cell data into PostgreSQL in batches."""
    async with db.connection() as conn:
        for i in range(0, len(metadata), batch_size):
            batch_meta = metadata[i:i + batch_size]
            batch_pert_emb = perturbation_embeddings[i:i + batch_size]
            batch_ct_emb = cell_type_embeddings[i:i + batch_size]
            batch_ctx_emb = sample_context_embeddings[i:i + batch_size]
            
            # Prepare batch insert
            values = [
                (
                    m.cell_index,
                    m.dataset,
                    f"{m.dataset}_{m.perturbation_name}_{m.cell_type_cl_id}_{m.donor_id}",
                    m.cell_type_original,
                    m.cell_type_cl_id,
                    m.cell_type_name,
                    m.perturbation_original,
                    m.perturbation_name,
                    m.perturbation_type,
                    m.is_control,
                    m.tissue_original,
                    m.tissue_uberon_id,
                    m.tissue_name,
                    m.donor_id,
                    m.disease_mondo_id,
                    m.disease_name,
                    m.sample_condition,
                    json.dumps(m.sample_metadata),
                    json.dumps(m.perturbation_external_ids),
                    m.perturbation_targets,
                    m.perturbation_pathways,
                    m.n_genes,
                    m.total_counts,
                    pert_emb,
                    ct_emb,
                    ctx_emb,
                )
                for m, pert_emb, ct_emb, ctx_emb in zip(
                    batch_meta, batch_pert_emb, batch_ct_emb, batch_ctx_emb
                )
            ]
            
            await conn.executemany(
                """
                INSERT INTO cells (
                    cell_index, dataset, group_id,
                    cell_type_original, cell_type_cl_id, cell_type_name,
                    perturbation_original, perturbation_name, perturbation_type, is_control,
                    tissue_original, tissue_uberon_id, tissue_name, donor_id,
                    disease_mondo_id, disease_name, sample_condition, sample_metadata,
                    perturbation_external_ids, perturbation_targets, perturbation_pathways,
                    n_genes, total_counts,
                    perturbation_embedding, cell_type_embedding, sample_context_embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
                """,
                values,
            )
            
            if (i + batch_size) % 100000 == 0:
                print(f"  Loaded {i + batch_size:,} / {len(metadata):,} cells")


async def build_hnsw_indexes(db: HaystackDatabase):
    """Build HNSW indexes after data is loaded."""
    async with db.connection() as conn:
        # Drop existing indexes if they exist
        await conn.execute("DROP INDEX IF EXISTS idx_cells_perturbation_embedding")
        await conn.execute("DROP INDEX IF EXISTS idx_cells_cell_type_embedding")
        await conn.execute("DROP INDEX IF EXISTS idx_cells_sample_context_embedding")
        await conn.execute("DROP INDEX IF EXISTS idx_groups_perturbation_embedding")
        await conn.execute("DROP INDEX IF EXISTS idx_groups_cell_type_embedding")
        await conn.execute("DROP INDEX IF EXISTS idx_groups_sample_context_embedding")
        
        # Build new indexes
        print("  Building perturbation embedding index on cells...")
        await conn.execute("""
            CREATE INDEX idx_cells_perturbation_embedding ON cells 
            USING hnsw (perturbation_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        
        print("  Building cell type embedding index on cells...")
        await conn.execute("""
            CREATE INDEX idx_cells_cell_type_embedding ON cells 
            USING hnsw (cell_type_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        
        print("  Building sample context embedding index on cells...")
        await conn.execute("""
            CREATE INDEX idx_cells_sample_context_embedding ON cells 
            USING hnsw (sample_context_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        
        print("  Building perturbation embedding index on groups...")
        await conn.execute("""
            CREATE INDEX idx_groups_perturbation_embedding ON cell_groups
            USING hnsw (perturbation_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        
        print("  Building cell type embedding index on groups...")
        await conn.execute("""
            CREATE INDEX idx_groups_cell_type_embedding ON cell_groups
            USING hnsw (cell_type_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        
        print("  Building sample context embedding index on groups...")
        await conn.execute("""
            CREATE INDEX idx_groups_sample_context_embedding ON cell_groups
            USING hnsw (sample_context_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
```

---

## 13. GCP Cloud SQL Configuration

### 11.1 Instance Setup

```bash
# Create Cloud SQL instance
gcloud sql instances create haystack-prod \
    --database-version=POSTGRES_15 \
    --tier=db-custom-4-15360 \
    --region=us-east1 \
    --storage-size=100GB \
    --storage-auto-increase \
    --availability-type=REGIONAL \
    --network=default \
    --no-assign-ip

# Create database
gcloud sql databases create haystack \
    --instance=haystack-prod

# Create users
gcloud sql users create haystack_app \
    --instance=haystack-prod \
    --password="secure_password"

gcloud sql users create haystack_agent \
    --instance=haystack-prod \
    --password="secure_password"
```

### 11.2 Enable pgvector Extension

```sql
-- Connect as postgres superuser
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 11.3 VPC Connector for Cloud Run

```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create haystack-vpc \
    --region=us-east1 \
    --network=default \
    --range=10.8.0.0/28

# Update Cloud Run service to use connector
gcloud run services update haystack-prod \
    --region=us-east1 \
    --vpc-connector=haystack-vpc \
    --vpc-egress=private-ranges-only
```

### 11.4 Connection from Cloud Run

```python
# backend/services/database.py
"""Cloud SQL database client."""

import asyncpg
from google.cloud.sql.connector import Connector, IPTypes
from contextlib import asynccontextmanager


class HaystackDatabase:
    """Async database client for Cloud SQL."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration with instance_connection_name
        """
        self.config = config
        self._pool = None
        self._connector = Connector()
    
    async def _get_conn(self):
        """Get connection using Cloud SQL Python Connector."""
        conn = await self._connector.connect_async(
            self.config.instance_connection_name,
            "asyncpg",
            user=self.config.user,
            db=self.config.database_name,
            ip_type=IPTypes.PRIVATE,  # Use private IP
        )
        return conn
    
    async def connect(self):
        """Initialize connection pool."""
        # For Cloud Run, create pool with connector
        self._pool = await asyncpg.create_pool(
            min_size=2,
            max_size=self.config.pool_size,
            max_inactive_connection_lifetime=300,
            dsn=None,  # Use connector instead
            connect=self._get_conn,
        )
    
    async def close(self):
        """Close connection pool and connector."""
        if self._pool:
            await self._pool.close()
        self._connector.close()
```

---

## 14. Python Database Client

### 12.1 Complete Client Implementation

```python
"""Complete HAYSTACK database client."""

import asyncpg
from google.cloud.sql.connector import Connector, IPTypes
from contextlib import asynccontextmanager
from typing import Optional, Any
import json


class HaystackDatabase:
    """Unified async database client for HAYSTACK."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database connection.
        
        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._connector = Connector()
    
    async def _get_conn(self):
        """Get connection using Cloud SQL Python Connector."""
        return await self._connector.connect_async(
            self.config.instance_connection_name,
            "asyncpg",
            user=self.config.user,
            db=self.config.database_name,
            ip_type=IPTypes.PRIVATE,
        )
    
    async def connect(self):
        """Initialize connection pool."""
        self._pool = await asyncpg.create_pool(
            min_size=2,
            max_size=self.config.pool_size,
            max_inactive_connection_lifetime=300,
            connect=self._get_conn,
        )
    
    async def close(self):
        """Close connection pool and connector."""
        if self._pool:
            await self._pool.close()
        self._connector.close()
    
    @asynccontextmanager
    async def connection(self):
        """Get a database connection from pool."""
        async with self._pool.acquire() as conn:
            yield conn
    
    async def execute_query(
        self,
        sql: str,
        params: Optional[tuple] = None,
        max_rows: int = 1000,
    ) -> list[dict]:
        """
        Execute a read-only SQL query.
        
        Args:
            sql: SQL query string
            params: Query parameters (tuple)
            max_rows: Maximum rows to return
        
        Returns:
            List of result dictionaries
        """
        async with self.connection() as conn:
            if params:
                rows = await conn.fetch(sql, *params, timeout=30)
            else:
                rows = await conn.fetch(sql, timeout=30)
            return [dict(row) for row in rows[:max_rows]]
    
    async def semantic_search(
        self,
        query_embedding: list[float],
        search_type: str,
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Vector similarity search using pgvector HNSW index.
        
        Args:
            query_embedding: Query vector (1536 dim)
            search_type: 'perturbation', 'cell_type', or 'sample_context'
            top_k: Number of results
            filters: Optional SQL filters
        
        Returns:
            List of results with similarity scores
        """
        embedding_col = f"{search_type}_embedding"
        
        # Build query
        sql = f"""
            SELECT 
                group_id,
                dataset,
                perturbation_name,
                cell_type_cl_id,
                cell_type_name,
                n_cells,
                has_control,
                control_group_id,
                1 - ({embedding_col} <=> $1::vector) as similarity
            FROM cell_groups
            WHERE {embedding_col} IS NOT NULL
        """
        
        params = [query_embedding]
        param_idx = 2
        
        if filters:
            for key, value in filters.items():
                if value is not None:
                    sql += f" AND {key} = ${param_idx}"
                    params.append(value)
                    param_idx += 1
        
        sql += f"""
            ORDER BY {embedding_col} <=> $1::vector
            LIMIT {top_k}
        """
        
        async with self.connection() as conn:
            rows = await conn.fetch(sql, *params, timeout=30)
            return [dict(row) for row in rows]
    
    async def get_cell_group(self, group_id: str) -> Optional[dict]:
        """Get a single cell group by ID."""
        sql = """
            SELECT *
            FROM cell_groups
            WHERE group_id = $1
        """
        rows = await self.execute_query(sql, (group_id,))
        return rows[0] if rows else None
    
    async def get_cell_indices(self, group_id: str) -> list[int]:
        """Get cell indices for a group."""
        sql = """
            SELECT cell_indices
            FROM cell_groups
            WHERE group_id = $1
        """
        rows = await self.execute_query(sql, (group_id,))
        return rows[0]["cell_indices"] if rows else []
    
    async def list_perturbations(
        self,
        dataset: Optional[str] = None,
        perturbation_type: Optional[str] = None,
    ) -> list[dict]:
        """List available perturbations."""
        sql = """
            SELECT 
                perturbation_name,
                perturbation_type,
                datasets_present,
                total_cells,
                cell_types_present
            FROM perturbations
            WHERE ($1::text IS NULL OR $1 = ANY(datasets_present))
              AND ($2::text IS NULL OR perturbation_type = $2)
            ORDER BY total_cells DESC
        """
        return await self.execute_query(sql, (dataset, perturbation_type))
    
    async def list_cell_types(
        self,
        dataset: Optional[str] = None,
    ) -> list[dict]:
        """List available cell types."""
        sql = """
            SELECT 
                cell_type_cl_id,
                cell_type_name,
                datasets_present,
                total_cells,
                perturbations_present
            FROM cell_types
            WHERE ($1::text IS NULL OR $1 = ANY(datasets_present))
            ORDER BY total_cells DESC
        """
        return await self.execute_query(sql, (dataset,))
```

---

## 15. Open Questions

### 13.1 Resolved Questions

| Question | Resolution |
|----------|------------|
| Which vector database to use? | PostgreSQL + pgvector on GCP Cloud SQL |
| How to handle 10M+ cells? | Cell groups reduce search space; HNSW indexes for vectors |
| Text vs STACK embeddings? | Text embeddings for MVP; STACK embeddings for future |
| How to score candidates? | Weighted combination of relevance, diversity, quality |

### 13.2 Open Questions

| Question | Options | Current Thinking |
|----------|---------|------------------|
| How many prompt cells optimal? | 10-50 per strategy | Start with 20, tune based on STACK performance |
| Include unperturbed controls? | Always / Sometimes / Never | Always when available (has_control flag) |
| Cross-dataset prompts? | Allow / Restrict | Allow but penalize in diversity score |
| Handle batch effects? | Ignore / Normalize / Model | Ignore for MVP; consider donor as group key |
| Embedding dimension reduction? | None / PCA / Matryoshka | None for MVP; consider for latency optimization |

### 13.3 Future Improvements

1. **STACK Cell Embeddings**: Add STACK-derived embeddings for transcriptomic similarity search during refinement iterations
2. **Caching Layer**: Redis cache for embedding API responses and frequent queries
3. **Read Replicas**: Cloud SQL read replicas for scaling concurrent agent queries
4. **Approximate Counts**: HyperLogLog for fast cardinality estimates in large GROUP BY queries
5. **Index Tuning**: Tune HNSW parameters (m, ef_construction, ef_search) based on recall/latency benchmarks
