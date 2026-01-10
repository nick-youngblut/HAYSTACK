# HAYSTACK Prompt Cell Retrieval Specification

## Table of Contents

1. [Overview](#1-overview)
2. [Atlas Data Model](#2-atlas-data-model)
3. [Database Schema (PostgreSQL + pgvector)](#3-database-schema-postgresql--pgvector)
4. [Retrieval Strategy Architecture](#4-retrieval-strategy-architecture)
5. [Direct Match Strategy](#5-direct-match-strategy)
6. [Mechanistic Match Strategy](#6-mechanistic-match-strategy)
7. [Semantic Match Strategy](#7-semantic-match-strategy)
8. [Ontology-Guided Strategy](#8-ontology-guided-strategy)
9. [Candidate Ranking and Selection](#9-candidate-ranking-and-selection)
10. [Index Building Pipeline](#10-index-building-pipeline)
11. [Unified Database: PostgreSQL + pgvector](#11-unified-database-postgresql--pgvector)
12. [Open Questions](#12-open-questions)

---

## 1. Overview

### 1.1 Problem Statement

HAYSTACK must select appropriate "prompt cells" for STACK's in-context learning from three heterogeneous atlases:
- **Parse PBMC**: ~10M cells, 90 cytokine perturbations, 12 donors
- **OpenProblems**: ~500K cells, 147 drug conditions, 3 donors
- **Tabula Sapiens**: ~500K cells, unperturbed, 25 tissues, 24 donors

Given a natural language query (e.g., "How would lung fibroblasts respond to TGF-beta?"), the system must:
1. Understand what the user is asking for (cell type, perturbation)
2. Find the most biologically relevant prompt cells from available data
3. Handle cases where exact matches don't exist

### 1.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multiple embeddings per cell | Yes | Enables both semantic (text) and cellular (STACK) similarity search |
| Strategy chaining | Yes | Higher-level strategies (Mechanistic, Ontology) produce filters; lower-level strategies (Direct, Semantic) retrieve cells |
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
│  │   ┌─────────────────┐        ┌─────────────────┐                │    │
│  │   │  Mechanistic    │        │  Ontology       │                │    │
│  │   │  Match          │        │  Guided         │                │    │
│  │   │                 │        │                 │                │    │
│  │   │ "Find perturbs  │        │ "Find cell      │                │    │
│  │   │  sharing        │        │  types in       │                │    │
│  │   │  targets with   │        │  same lineage"  │                │    │
│  │   │  TGF-beta"      │        │                 │                │    │
│  │   └────────┬────────┘        └────────┬────────┘                │    │
│  │            │                          │                         │    │
│  │            ▼                          ▼                         │    │
│  │   perturbation_filter         cell_type_filter                  │    │
│  │   = ["BMP4", "Activin A"]     = ["CL:0000057", "CL:0002553"]    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              RETRIEVAL STRATEGIES (fetch cells)                 │    │
│  │                                                                 │    │
│  │   ┌─────────────────┐        ┌─────────────────┐                │    │
│  │   │  Direct         │        │  Semantic       │                │    │
│  │   │  Match          │        │  Match          │                │    │
│  │   │                 │        │                 │                │    │
│  │   │ SQL filters on  │        │ Vector          │                │    │
│  │   │ PostgreSQL      │        │ similarity      │                │    │
│  │   │                 │        │ (pgvector)      │                │    │
│  │   └─────────────────┘        └─────────────────┘                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Atlas Data Model

### 2.1 Harmonized Metadata Schema

All three atlases must be harmonized to a common metadata schema before indexing. This happens at index build time.

```python
class HarmonizedCellMetadata(BaseModel):
    """Harmonized metadata schema for all cells across atlases."""
    
    # === Identity ===
    cell_id: str = Field(description="Globally unique cell ID: {dataset}_{original_index}")
    dataset: Literal["parse_pbmc", "openproblems", "tabula_sapiens"]
    atlas_index: int = Field(description="Original index in source H5AD file")
    
    # === Cell Type (harmonized to Cell Ontology) ===
    cell_type_original: str = Field(description="Original annotation from dataset")
    cell_type_harmonized: str = Field(description="Harmonized cell type name")
    cell_type_cl_id: str | None = Field(description="Cell Ontology ID (e.g., CL:0000235)")
    cell_type_lineage: list[str] = Field(
        default_factory=list,
        description="Lineage path from root (e.g., ['cell', 'native cell', 'leukocyte', 'macrophage'])"
    )
    
    # === Tissue ===
    tissue_original: str | None = Field(description="Original tissue annotation")
    tissue_harmonized: str | None = Field(description="Harmonized tissue name")
    tissue_uberon_id: str | None = Field(description="UBERON ontology ID")
    
    # === Donor ===
    donor_id: str = Field(description="Donor identifier (dataset-specific)")
    
    # === Perturbation Status ===
    is_control: bool = Field(description="True if this is a control/unperturbed cell")
    
    # === Perturbation Details (if perturbed) ===
    perturbation_name: str | None = Field(description="Harmonized perturbation name")
    perturbation_original: str | None = Field(description="Original perturbation annotation")
    perturbation_type: Literal["drug", "cytokine", "genetic", None] = None
    perturbation_external_ids: dict[str, str] = Field(
        default_factory=dict,
        description="External IDs: {'chebi': 'CHEBI:xxxxx', 'drugbank': 'DB00001', ...}"
    )
    perturbation_targets: list[str] = Field(
        default_factory=list,
        description="Known target genes (HGNC symbols)"
    )
    perturbation_pathways: list[str] = Field(
        default_factory=list,
        description="Associated pathway IDs (KEGG, Reactome)"
    )
    
    # === Quality Metrics ===
    n_genes_detected: int
    total_counts: float
    
    # === Precomputed for Search ===
    perturbation_description: str | None = Field(
        description="Natural language description for text embedding"
    )
```

### 2.2 Cell Group Concept

Rather than retrieving individual cells, HAYSTACK retrieves **cell groups** - sets of cells that share key attributes and can serve as a coherent prompt.

```python
class CellGroup(BaseModel):
    """A group of cells that can serve as a prompt unit."""
    
    group_id: str = Field(description="Unique group identifier")
    
    # Defining attributes (cells in group share these)
    dataset: str
    perturbation_name: str | None
    cell_type_cl_id: str | None
    donor_id: str
    
    # Group statistics
    n_cells: int
    cell_indices: list[int] = Field(description="Indices into atlas H5AD file")
    
    # Aggregated metadata
    mean_n_genes: float
    mean_total_counts: float
    
    # For retrieval
    has_control: bool = Field(description="Whether matching control cells exist")
    control_group_id: str | None = Field(description="ID of corresponding control group")
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

## 3. Database Schema (PostgreSQL + pgvector)

### 3.1 Overview

HAYSTACK uses **PostgreSQL + pgvector** as its unified database, providing:
- Full SQL for flexible agent queries
- HNSW indexes for vector similarity search
- Proven scale to 10M+ vectors
- Read-only role enforcement for agents

The complete schema definition and setup is in [Section 11](#11-unified-database-postgresql--pgvector). This section covers the conceptual data model.

### 3.2 Multi-Embedding Design

Each cell record stores two text embedding vectors:

| Column | Model | Dimension | Purpose |
|--------|-------|-----------|---------|
| `perturbation_embedding` | text-embedding-3-large | 1536 | Semantic perturbation search |
| `cell_type_embedding` | text-embedding-3-large | 1536 | Semantic cell type search |

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
    parts = [metadata.cell_type_harmonized]
    
    if metadata.tissue_harmonized:
        parts.append(f"from {metadata.tissue_harmonized}")
    
    if metadata.cell_type_lineage:
        lineage_str = " > ".join(metadata.cell_type_lineage[-3:])
        parts.append(f"(lineage: {lineage_str})")
    
    return " ".join(parts)
    # Example: "alveolar macrophage from lung (lineage: mononuclear phagocyte > macrophage > alveolar macrophage)"
```

### 3.4 Index Structure

PostgreSQL HNSW indexes are created on each embedding column:

```sql
-- HNSW indexes for vector similarity search
CREATE INDEX idx_perturbation_embedding ON cells 
    USING hnsw (perturbation_embedding vector_cosine_ops);
CREATE INDEX idx_cell_type_embedding ON cells 
    USING hnsw (cell_type_embedding vector_cosine_ops);

-- Scalar indexes for filtered queries
CREATE INDEX idx_cells_dataset ON cells(dataset);
CREATE INDEX idx_cells_cell_type ON cells(cell_type_cl_id);
CREATE INDEX idx_cells_perturbation ON cells(perturbation_name);
```

Combined with scalar indexes on metadata columns, this enables efficient filtered vector search.

---

## 4. Retrieval Strategy Architecture

### 4.1 Strategy Interface

```python
from abc import ABC, abstractmethod

class RetrievalStrategy(ABC):
    """Base class for all retrieval strategies."""
    
    @abstractmethod
    def retrieve(
        self,
        query: StructuredQuery,
        db: HaystackDatabase,
        filters: dict | None = None,
        k: int = 10,
    ) -> list[CellGroupCandidate]:
        """
        Retrieve cell group candidates.
        
        Args:
            query: Parsed user query with resolved entities
            db: PostgreSQL database connection
            filters: Optional pre-filters from upstream strategies
            k: Maximum number of candidates to return
        
        Returns:
            List of cell group candidates with scores and rationale
        """
        pass


class FilterStrategy(ABC):
    """Base class for strategies that produce filters rather than cells."""
    
    @abstractmethod
    def generate_filters(
        self,
        query: StructuredQuery,
        knowledge_base: BiologicalKnowledgeBase,
    ) -> dict:
        """
        Generate filters for downstream retrieval.
        
        Args:
            query: Parsed user query
            knowledge_base: Access to KEGG, Reactome, etc.
        
        Returns:
            Dictionary of filters to apply in retrieval
        """
        pass
```

### 4.2 Candidate Model

```python
class CellGroupCandidate(BaseModel):
    """A candidate cell group for use as STACK prompt."""
    
    # Identity
    candidate_id: str
    group_id: str
    
    # Source
    dataset: str
    perturbation_name: str | None
    cell_type_cl_id: str | None
    cell_type_name: str
    donor_id: str
    n_cells: int
    
    # How it was found
    strategy: Literal["direct", "mechanistic", "semantic_perturbation", "semantic_cell_type", "ontology"]
    
    # Scoring
    relevance_score: float = Field(ge=0.0, le=1.0, description="Normalized relevance score")
    
    # Rationale
    rationale: str
    match_details: dict = Field(default_factory=dict)
    
    # Biological justification
    shared_targets: list[str] = Field(default_factory=list)
    shared_pathways: list[str] = Field(default_factory=list)
    ontology_distance: int | None = None  # Hops in cell ontology
    
    # For retrieval
    cell_indices: list[int]
    control_indices: list[int] | None = None
```

---

## 5. Direct Match Strategy

### 5.1 Overview

Direct Match performs **metadata-based filtering** on the PostgreSQL database. It searches for exact or fuzzy matches on perturbation name and cell type.

### 5.2 Implementation

```python
class DirectMatchStrategy(RetrievalStrategy):
    """Find cells by direct metadata matching."""
    
    def retrieve(
        self,
        query: StructuredQuery,
        db: HaystackDatabase,
        filters: dict | None = None,
        k: int = 10,
    ) -> list[CellGroupCandidate]:
        """
        Retrieve by metadata filtering.
        
        Search hierarchy:
        1. Exact perturbation + exact cell type
        2. Exact perturbation + any cell type
        3. Fuzzy perturbation match + exact cell type
        4. Apply any upstream filters (from Mechanistic/Ontology)
        """
        candidates = []
        
        # Build base filter
        base_filter = self._build_filter(filters)
        
        # Level 1: Exact perturbation + exact cell type
        if query.perturbation and query.query_cell_type_cl_id:
            exact_matches = self._search_exact(
                index,
                perturbation=query.perturbation,
                cell_type_cl_id=query.query_cell_type_cl_id,
                additional_filters=base_filter,
            )
            candidates.extend(self._to_candidates(exact_matches, strategy="direct", match_type="exact"))
        
        # Level 2: Exact perturbation + any cell type (if we need more candidates)
        if len(candidates) < k and query.perturbation:
            perturb_matches = self._search_by_perturbation(
                index,
                perturbation=query.perturbation,
                additional_filters=base_filter,
                exclude_groups=[c.group_id for c in candidates],
            )
            candidates.extend(self._to_candidates(perturb_matches, strategy="direct", match_type="perturbation_only"))
        
        # Level 3: Synonym/alias matching
        if len(candidates) < k and query.perturbation:
            synonyms = self._get_perturbation_synonyms(query.perturbation)
            for synonym in synonyms:
                syn_matches = self._search_by_perturbation(
                    index,
                    perturbation=synonym,
                    additional_filters=base_filter,
                    exclude_groups=[c.group_id for c in candidates],
                )
                candidates.extend(self._to_candidates(syn_matches, strategy="direct", match_type="synonym"))
        
        return candidates[:k]
    
    def _search_exact(
        self,
        db: HaystackDatabase,
        perturbation: str,
        cell_type_cl_id: str,
        additional_filters: str | None,
    ) -> list[dict]:
        """Search for exact match on perturbation and cell type."""
        filter_conditions = [
            f"perturbation_name = '{perturbation}'",
            f"cell_type_cl_id = '{cell_type_cl_id}'",
        ]
        if additional_filters:
            filter_conditions.append(additional_filters)
        
        return index.table.search().where(" AND ".join(filter_conditions)).limit(100).to_list()
    
    def _get_perturbation_synonyms(self, perturbation: str) -> list[str]:
        """Get synonyms for a perturbation name."""
        # Uses pre-built synonym table from PubChem, ChEMBL, etc.
        return PERTURBATION_SYNONYMS.get(perturbation.lower(), [])
```

### 5.3 Perturbation Name Normalization

A key challenge is that the same perturbation may have different names across datasets:

| Parse PBMC | OpenProblems | Harmonized |
|------------|--------------|------------|
| "IFNg" | "Interferon gamma" | "IFN-gamma" |
| "TNFa" | "TNF-alpha" | "TNF-alpha" |
| "TGFb" | "TGF-beta-1" | "TGF-beta" |

```python
# Pre-built at index time
PERTURBATION_SYNONYMS = {
    "ifn-gamma": ["ifng", "interferon gamma", "interferon-gamma", "ifn-γ"],
    "tnf-alpha": ["tnfa", "tnf", "tumor necrosis factor alpha", "tnf-α"],
    "tgf-beta": ["tgfb", "tgfb1", "tgf-beta-1", "transforming growth factor beta"],
    # ... etc
}
```

---

## 6. Mechanistic Match Strategy

### 6.1 Overview

Mechanistic Match is a **filter strategy** that identifies perturbations sharing biological mechanisms (targets, pathways) with the query perturbation. It produces a list of candidate perturbation names that are then passed to Direct Match for actual cell retrieval.

### 6.2 Implementation

```python
class MechanisticMatchStrategy(FilterStrategy):
    """Find perturbations sharing targets/pathways with query."""
    
    def __init__(self, knowledge_base: BiologicalKnowledgeBase):
        self.kb = knowledge_base
    
    def generate_filters(
        self,
        query: StructuredQuery,
        knowledge_base: BiologicalKnowledgeBase,
    ) -> dict:
        """
        Generate perturbation filter based on mechanistic similarity.
        
        Returns:
            {"perturbation_names": [...], "rationale": {...}}
        """
        if not query.target_genes and not query.expected_pathways:
            return {"perturbation_names": [], "rationale": "No target information available"}
        
        # Get all perturbations in our index
        available_perturbations = self._get_indexed_perturbations()
        
        # Score each by overlap with query targets/pathways
        scored_perturbations = []
        for perturb in available_perturbations:
            score, details = self._compute_mechanistic_similarity(
                query_targets=query.target_genes,
                query_pathways=query.expected_pathways,
                candidate_targets=perturb["targets"],
                candidate_pathways=perturb["pathways"],
            )
            if score > 0:
                scored_perturbations.append({
                    "perturbation_name": perturb["name"],
                    "score": score,
                    "shared_targets": details["shared_targets"],
                    "shared_pathways": details["shared_pathways"],
                })
        
        # Sort by score and return top candidates
        scored_perturbations.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored_perturbations[:10]
        
        return {
            "perturbation_names": [p["perturbation_name"] for p in top_candidates],
            "rationale": top_candidates,
        }
    
    def _compute_mechanistic_similarity(
        self,
        query_targets: list[str],
        query_pathways: list[str],
        candidate_targets: list[str],
        candidate_pathways: list[str],
    ) -> tuple[float, dict]:
        """
        Compute Jaccard-like similarity between query and candidate.
        
        Returns:
            (score, details_dict)
        """
        # Target overlap (weighted higher)
        shared_targets = set(query_targets) & set(candidate_targets)
        target_jaccard = len(shared_targets) / max(1, len(set(query_targets) | set(candidate_targets)))
        
        # Pathway overlap
        shared_pathways = set(query_pathways) & set(candidate_pathways)
        pathway_jaccard = len(shared_pathways) / max(1, len(set(query_pathways) | set(candidate_pathways)))
        
        # Combined score (targets weighted 2x)
        score = (2 * target_jaccard + pathway_jaccard) / 3
        
        return score, {
            "shared_targets": list(shared_targets),
            "shared_pathways": list(shared_pathways),
            "target_jaccard": target_jaccard,
            "pathway_jaccard": pathway_jaccard,
        }
    
    def _get_indexed_perturbations(self) -> list[dict]:
        """Get all unique perturbations in the index with their targets/pathways."""
        # This is pre-computed at index build time and stored as a separate table
        return self.kb.get_indexed_perturbation_metadata()
```

### 6.3 Chaining to Direct Match

```python
def retrieve_with_mechanistic_match(
    query: StructuredQuery,
    db: HaystackDatabase,
    kb: BiologicalKnowledgeBase,
) -> list[CellGroupCandidate]:
    """
    Full mechanistic match pipeline.
    
    1. Mechanistic strategy generates perturbation filters
    2. Direct strategy retrieves cells matching those perturbations
    """
    # Step 1: Get mechanistically similar perturbations
    mechanistic = MechanisticMatchStrategy(kb)
    filters = mechanistic.generate_filters(query, kb)
    
    if not filters["perturbation_names"]:
        return []
    
    # Step 2: Use Direct Match to retrieve cells
    direct = DirectMatchStrategy()
    candidates = []
    
    for perturb_info in filters["rationale"]:
        perturb_name = perturb_info["perturbation_name"]
        
        # Search for this perturbation
        matches = direct.retrieve(
            query=query,
            index=index,
            filters={"perturbation_name": perturb_name},
            k=5,
        )
        
        # Annotate with mechanistic rationale
        for match in matches:
            match.strategy = "mechanistic"
            match.shared_targets = perturb_info["shared_targets"]
            match.shared_pathways = perturb_info["shared_pathways"]
            match.rationale = (
                f"Shares targets ({', '.join(perturb_info['shared_targets'][:3])}) "
                f"and pathways with query perturbation"
            )
        
        candidates.extend(matches)
    
    return candidates
```

---

## 7. Semantic Match Strategy

### 7.1 Overview

Semantic Match uses **vector similarity search** in PostgreSQL via pgvector. There are two sub-strategies:

1. **Perturbation Semantic Match**: Embed the query perturbation description → search `perturbation_embedding` column
2. **Cell Type Semantic Match**: Embed the query cell type description → search `cell_type_embedding` column

**Note**: STACK cell embeddings are excluded from MVP. Generating a query embedding would require gene expression data, which we don't have for hypothetical/query cells. Text embeddings of perturbation and cell type descriptions are sufficient for retrieval.

### 7.2 Implementation

```python
class SemanticMatchStrategy(RetrievalStrategy):
    """Find cells by semantic similarity of text descriptions."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        self.embed_model = embedding_model
    
    def retrieve(
        self,
        query: StructuredQuery,
        db: HaystackDatabase,
        filters: dict | None = None,
        k: int = 10,
        search_type: Literal["perturbation", "cell_type", "both"] = "both",
    ) -> list[CellGroupCandidate]:
        """
        Retrieve by semantic similarity.
        
        Args:
            search_type: Which embedding column(s) to search
        """
        candidates = []
        
        if search_type in ["perturbation", "both"]:
            perturb_candidates = self._search_perturbation_semantic(query, index, filters, k)
            candidates.extend(perturb_candidates)
        
        if search_type in ["cell_type", "both"]:
            cell_type_candidates = self._search_cell_type_semantic(query, index, filters, k)
            candidates.extend(cell_type_candidates)
        
        # Deduplicate and re-rank
        return self._deduplicate_and_rank(candidates, k)
    
    def _search_perturbation_semantic(
        self,
        query: StructuredQuery,
        db: HaystackDatabase,
        filters: dict | None,
        k: int,
    ) -> list[CellGroupCandidate]:
        """Search by perturbation description similarity."""
        # Generate query text
        query_text = self._generate_perturbation_query_text(query)
        
        # Embed query
        query_embedding = self._embed_text(query_text)
        
        # Vector search on perturbation_embedding column
        results = index.table.search(
            query_embedding,
            vector_column_name="perturbation_embedding",
        ).limit(k * 3)  # Get more, then filter
        
        if filters:
            results = results.where(self._build_filter_string(filters))
        
        results = results.to_list()
        
        return self._to_candidates(
            results,
            strategy="semantic_perturbation",
            query_text=query_text,
        )
    
    def _generate_perturbation_query_text(self, query: StructuredQuery) -> str:
        """Generate text query for perturbation semantic search."""
        parts = [query.perturbation]
        
        if query.perturbation_type != PerturbationType.UNKNOWN:
            parts.append(f"({query.perturbation_type.value})")
        
        if query.target_genes:
            targets = ", ".join(query.target_genes[:5])
            parts.append(f"targeting {targets}")
        
        if query.expected_pathways:
            pathways = ", ".join(query.expected_pathways[:3])
            parts.append(f"affecting {pathways}")
        
        return " ".join(parts)
    
    def _search_cell_type_semantic(
        self,
        query: StructuredQuery,
        db: HaystackDatabase,
        filters: dict | None,
        k: int,
    ) -> list[CellGroupCandidate]:
        """Search by cell type description similarity."""
        # Generate query text
        query_text = self._generate_cell_type_query_text(query)
        
        # Embed query
        query_embedding = self._embed_text(query_text)
        
        # Vector search on cell_type_embedding column
        results = index.table.search(
            query_embedding,
            vector_column_name="cell_type_embedding",
        ).limit(k * 3)
        
        if filters:
            results = results.where(self._build_filter_string(filters))
        
        results = results.to_list()
        
        return self._to_candidates(
            results,
            strategy="semantic_cell_type",
            query_text=query_text,
        )
    
    def _generate_cell_type_query_text(self, query: StructuredQuery) -> str:
        """Generate text query for cell type semantic search."""
        parts = [query.query_cell_type]
        
        if query.query_tissue:
            parts.append(f"from {query.query_tissue}")
        
        return " ".join(parts)
    
    def _embed_text(self, text: str) -> list[float]:
        """Embed text using the configured model."""
        # Uses OpenAI, Anthropic, or other embedding API
        return get_embedding(text, model=self.embed_model)
```

---

## 8. Ontology-Guided Strategy

### 8.1 Overview

Ontology-Guided is a **filter strategy** that uses Cell Ontology relationships to find related cell types. This is useful when:
- The exact query cell type doesn't exist in any atlas
- We want to expand context cells to include related cell types

### 8.2 Implementation

```python
class OntologyGuidedStrategy(FilterStrategy):
    """Use Cell Ontology to find related cell types."""
    
    def __init__(self, cell_ontology: CellOntology):
        self.co = cell_ontology
    
    def generate_filters(
        self,
        query: StructuredQuery,
        knowledge_base: BiologicalKnowledgeBase,
    ) -> dict:
        """
        Generate cell type filters based on ontology relationships.
        
        Returns:
            {"cell_type_cl_ids": [...], "rationale": {...}}
        """
        if not query.query_cell_type_cl_id:
            # Try to resolve cell type to CL ID first
            resolved = self.co.resolve_name(query.query_cell_type)
            if not resolved:
                return {"cell_type_cl_ids": [], "rationale": "Could not resolve cell type"}
            query_cl_id = resolved["cl_id"]
        else:
            query_cl_id = query.query_cell_type_cl_id
        
        # Get related cell types
        related = self._get_related_cell_types(query_cl_id)
        
        # Filter to those present in our atlases
        available_cl_ids = self._get_indexed_cell_types()
        filtered_related = [
            r for r in related
            if r["cl_id"] in available_cl_ids
        ]
        
        return {
            "cell_type_cl_ids": [r["cl_id"] for r in filtered_related],
            "rationale": filtered_related,
        }
    
    def _get_related_cell_types(
        self,
        cl_id: str,
        max_distance: int = 2,
    ) -> list[dict]:
        """
        Get cell types related to query via ontology relationships.
        
        Relationships considered:
        - is_a (parent/child)
        - develops_from (developmental lineage)
        - part_of (tissue relationships)
        """
        related = []
        
        # Get parents (more general cell types)
        parents = self.co.get_ancestors(cl_id, max_distance=max_distance)
        for parent in parents:
            related.append({
                "cl_id": parent["cl_id"],
                "name": parent["name"],
                "relationship": "is_a",
                "distance": parent["distance"],
                "direction": "parent",
            })
        
        # Get children (more specific cell types)
        children = self.co.get_descendants(cl_id, max_distance=max_distance)
        for child in children:
            related.append({
                "cl_id": child["cl_id"],
                "name": child["name"],
                "relationship": "is_a",
                "distance": child["distance"],
                "direction": "child",
            })
        
        # Get siblings (same parent)
        siblings = self.co.get_siblings(cl_id)
        for sibling in siblings:
            related.append({
                "cl_id": sibling["cl_id"],
                "name": sibling["name"],
                "relationship": "sibling",
                "distance": 2,
                "direction": "sibling",
            })
        
        # Get developmental relatives
        dev_relatives = self.co.get_developmental_relatives(cl_id, max_distance=max_distance)
        for dev in dev_relatives:
            related.append({
                "cl_id": dev["cl_id"],
                "name": dev["name"],
                "relationship": "develops_from",
                "distance": dev["distance"],
                "direction": dev["direction"],
            })
        
        # Sort by distance (prefer closer relatives)
        related.sort(key=lambda x: x["distance"])
        
        return related
    
    def _get_indexed_cell_types(self) -> set[str]:
        """Get all CL IDs present in the index."""
        # Pre-computed at index build time
        return INDEXED_CELL_TYPE_CL_IDS
```

### 8.3 Chaining to Direct Match

```python
def retrieve_with_ontology_guidance(
    query: StructuredQuery,
    db: HaystackDatabase,
    cell_ontology: CellOntology,
) -> list[CellGroupCandidate]:
    """
    Full ontology-guided pipeline.
    
    1. Ontology strategy generates cell type filters
    2. Direct strategy retrieves cells matching those cell types
    """
    # Step 1: Get related cell types
    ontology = OntologyGuidedStrategy(cell_ontology)
    filters = ontology.generate_filters(query, None)
    
    if not filters["cell_type_cl_ids"]:
        return []
    
    # Step 2: Use Direct Match for each related cell type
    direct = DirectMatchStrategy()
    candidates = []
    
    for cl_info in filters["rationale"]:
        cl_id = cl_info["cl_id"]
        
        # Search for cells of this type with the query perturbation
        matches = direct.retrieve(
            query=query,
            index=index,
            filters={"cell_type_cl_id": cl_id},
            k=5,
        )
        
        # Annotate with ontology rationale
        for match in matches:
            match.strategy = "ontology"
            match.ontology_distance = cl_info["distance"]
            match.rationale = (
                f"Cell type '{cl_info['name']}' is a {cl_info['relationship']} "
                f"of query cell type (distance={cl_info['distance']})"
            )
        
        candidates.extend(matches)
    
    # Sort by ontology distance
    candidates.sort(key=lambda x: x.ontology_distance or 999)
    
    return candidates
```

---

## 9. Candidate Ranking and Selection

### 9.1 Multi-Strategy Aggregation

After running all strategies in parallel, we need to aggregate and rank candidates:

```python
class CandidateRanker:
    """Rank and select final prompt candidates from multiple strategies."""
    
    def __init__(self, weights: dict | None = None):
        self.weights = weights or {
            "direct": 1.0,           # Exact matches are gold standard
            "mechanistic": 0.8,      # Shared targets/pathways is strong signal
            "semantic_perturbation": 0.6,
            "semantic_cell_type": 0.5,
            "ontology": 0.7,
        }
    
    def rank(
        self,
        candidates: list[CellGroupCandidate],
        query: StructuredQuery,
    ) -> list[CellGroupCandidate]:
        """
        Rank all candidates and return sorted list.
        
        Scoring factors:
        1. Strategy weight
        2. Match quality (exact vs fuzzy)
        3. Cell type match (exact CL ID vs related)
        4. Number of cells (prefer larger groups)
        5. Has matching control cells
        """
        for candidate in candidates:
            candidate.final_score = self._compute_final_score(candidate, query)
        
        # Sort by final score descending
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Remove duplicates (same group from different strategies)
        seen_groups = set()
        unique_candidates = []
        for c in candidates:
            if c.group_id not in seen_groups:
                seen_groups.add(c.group_id)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def _compute_final_score(
        self,
        candidate: CellGroupCandidate,
        query: StructuredQuery,
    ) -> float:
        """Compute final ranking score."""
        score = 0.0
        
        # Strategy weight
        score += self.weights.get(candidate.strategy, 0.5) * 0.3
        
        # Relevance score from strategy
        score += candidate.relevance_score * 0.3
        
        # Cell type match bonus
        if candidate.cell_type_cl_id == query.query_cell_type_cl_id:
            score += 0.2
        elif candidate.ontology_distance and candidate.ontology_distance <= 1:
            score += 0.1
        
        # Group size bonus (log scale)
        score += min(0.1, np.log10(candidate.n_cells + 1) / 10)
        
        # Control availability bonus
        if candidate.control_indices:
            score += 0.1
        
        return score
    
    def select_top_k(
        self,
        candidates: list[CellGroupCandidate],
        k: int = 3,
        diversity: bool = True,
    ) -> list[CellGroupCandidate]:
        """
        Select top k candidates, optionally with diversity.
        
        If diversity=True, ensure selected candidates come from
        different strategies/datasets when possible.
        """
        if not diversity:
            return candidates[:k]
        
        selected = []
        seen_strategies = set()
        seen_datasets = set()
        
        for candidate in candidates:
            # First pass: prioritize diversity
            if len(selected) < k:
                if (candidate.strategy not in seen_strategies or 
                    candidate.dataset not in seen_datasets):
                    selected.append(candidate)
                    seen_strategies.add(candidate.strategy)
                    seen_datasets.add(candidate.dataset)
        
        # Second pass: fill remaining with best scores
        for candidate in candidates:
            if len(selected) >= k:
                break
            if candidate not in selected:
                selected.append(candidate)
        
        return selected
```

---

## 10. Index Building Pipeline

### 10.1 Overview

The index is built once, offline, before HAYSTACK can run queries. This involves:

1. Loading and harmonizing all three atlases
2. Generating text descriptions for perturbations and cell types
3. Computing text embeddings
4. Loading data into PostgreSQL with pgvector
5. Building HNSW indexes on vector columns
6. Building auxiliary lookup tables

### 10.2 Pipeline Steps

```python
def build_haystack_index(
    parse_pbmc_path: str,
    openproblems_path: str,
    tabula_sapiens_path: str,
    db_connection_string: str,
    text_embedding_model: str = "text-embedding-3-large",
):
    """
    Build complete HAYSTACK PostgreSQL database.
    
    Estimated time: 1-2 hours depending on text embedding API throughput.
    
    Args:
        parse_pbmc_path: Path to Parse PBMC H5AD
        openproblems_path: Path to OpenProblems H5AD
        tabula_sapiens_path: Path to Tabula Sapiens H5AD
        db_connection_string: PostgreSQL connection string
        text_embedding_model: Text embedding model name
    """
    # Step 1: Load and harmonize metadata
    print("Step 1: Harmonizing atlas metadata...")
    parse_metadata = harmonize_parse_pbmc(parse_pbmc_path)
    op_metadata = harmonize_openproblems(openproblems_path)
    ts_metadata = harmonize_tabula_sapiens(tabula_sapiens_path)
    
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
    
    # Step 4: Compute text embeddings
    print("Step 4: Computing text embeddings...")
    perturbation_embeddings = batch_embed_texts(perturbation_texts, text_embedding_model)
    cell_type_embeddings = batch_embed_texts(cell_type_texts, text_embedding_model)
    
    # Step 5: Load into PostgreSQL
    print("Step 5: Loading into PostgreSQL...")
    load_cells_to_postgres(
        db_connection_string,
        metadata=all_metadata,
        perturbation_embeddings=perturbation_embeddings,
        cell_type_embeddings=cell_type_embeddings,
    )
    
    # Step 6: Build HNSW indexes
    print("Step 6: Building HNSW indexes...")
    build_hnsw_indexes(db_connection_string)
    
    # Step 7: Build auxiliary tables
    print("Step 7: Building auxiliary tables...")
    build_perturbation_lookup_table(all_metadata, db_connection_string)
    build_cell_type_lookup_table(all_metadata, db_connection_string)
    build_synonym_table(db_connection_string)
    
    print("Done!")
```

### 10.3 Auxiliary Tables

```python
# Perturbation lookup table (for Mechanistic Match)
PERTURBATION_METADATA_TABLE = {
    "perturbation_name": str,      # Harmonized name
    "perturbation_type": str,
    "external_ids": dict,          # ChEBI, DrugBank, etc.
    "targets": list[str],          # Known target genes
    "pathways": list[str],         # Associated pathways
    "datasets_present": list[str], # Which atlases have this perturbation
    "total_cells": int,
    "cell_types_present": list[str],
}

# Cell type lookup table (for Ontology-Guided)
CELL_TYPE_METADATA_TABLE = {
    "cell_type_cl_id": str,
    "cell_type_name": str,
    "lineage": list[str],
    "datasets_present": list[str],
    "total_cells": int,
    "perturbations_present": list[str],
}

# Synonym table (for Direct Match fuzzy matching)
SYNONYM_TABLE = {
    "canonical_name": str,
    "synonyms": list[str],
    "entity_type": str,  # "perturbation" or "cell_type"
}
```

---

## 11. Unified Database: PostgreSQL + pgvector

### 11.1 Why PostgreSQL + pgvector?

HAYSTACK requires a database that provides:

1. **Full SQL support** - Agents need flexible introspection (GROUP BY, JOINs, aggregations)
2. **Vector similarity search** - HNSW indexes for semantic matching
3. **Scale to 10M+ cells** - Production-ready at this scale
4. **Local hosting** - No cloud dependencies
5. **Single database** - No dual-database complexity

**Comparison of Options:**

| Requirement | LanceDB | DuckDB + vss | PostgreSQL + pgvector |
|-------------|---------|--------------|----------------------|
| Full SQL | ❌ WHERE only | ✓ Complete | ✓ Complete |
| Vector search (HNSW) | ✓ | ✓ | ✓ |
| Scale to 10M+ | ✓ | ⚠️ Index must fit in RAM | ✓ Proven |
| Persistence | ✓ | ⚠️ Experimental, can corrupt | ✓ Production-ready |
| Single database | ✓ | ✓ | ✓ |

**Why not DuckDB + vss?**

From DuckDB documentation: *"The index itself is not buffer managed and must be able to fit into RAM memory... persistence is experimental and WAL recovery is not yet properly implemented... you can end up with data loss or corruption."*

For 10M cells with two 1536-dimensional text embeddings, this would require the HNSW index (~150GB+) to fit entirely in RAM, with no disk-based fallback.

**Why PostgreSQL + pgvector?**

- Proven at 10M+ scale (AWS benchmarks demonstrate good performance)
- HNSW indexes with iterative scanning for filtered queries (pgvector 0.8+)
- Full SQL for agent introspection
- Read-only access via database roles
- Mature, production-ready persistence

### 11.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED DATABASE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Agent                                                                 │
│     │                                                                   │
│     ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    PostgreSQL + pgvector                        │   │
│   │                                                                 │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │                    SQL Interface                        │   │   │
│   │   │                                                         │   │   │
│   │   │  • Full SQL (SELECT, JOIN, GROUP BY, aggregations)      │   │   │
│   │   │  • Read-only role for agents                            │   │   │
│   │   │  • Query timeout limits                                 │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                           │                                     │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │                 Vector Search (pgvector)                │   │   │
│   │   │                                                         │   │   │
│   │   │  • HNSW indexes on embedding columns                    │   │   │
│   │   │  • Cosine, L2, inner product distances                  │   │   │
│   │   │  • Filtered vector search with iterative scanning       │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                           │                                     │   │
│   │   ┌─────────────────────────────────────────────────────────┐   │   │
│   │   │                      Tables                             │   │   │
│   │   │                                                         │   │   │
│   │   │  • cells (main table with vectors + metadata)           │   │   │
│   │   │  • cell_groups (aggregated prompt units)                │   │   │
│   │   │  • perturbation_metadata (targets, pathways)            │   │   │
│   │   │  • cell_type_metadata (ontology, lineage)               │   │   │
│   │   └─────────────────────────────────────────────────────────┘   │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Schema Definition

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main cells table with multiple embedding columns
CREATE TABLE cells (
    -- Identity
    cell_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    atlas_index INTEGER NOT NULL,
    
    -- Cell type (harmonized)
    cell_type_original TEXT,
    cell_type_harmonized TEXT NOT NULL,
    cell_type_cl_id TEXT,
    
    -- Tissue
    tissue_original TEXT,
    tissue_harmonized TEXT,
    tissue_uberon_id TEXT,
    
    -- Donor
    donor_id TEXT NOT NULL,
    
    -- Perturbation status
    is_control BOOLEAN NOT NULL,
    perturbation_name TEXT,
    perturbation_type TEXT,  -- 'drug', 'cytokine', 'genetic', NULL
    
    -- Group membership
    group_id TEXT NOT NULL,
    
    -- Quality metrics
    n_genes_detected INTEGER,
    total_counts REAL,
    
    -- Vector embeddings (pgvector types)
    perturbation_embedding vector(1536),   -- Text embedding of perturbation description
    cell_type_embedding vector(1536)       -- Text embedding of cell type description
);

-- Cell groups (aggregated prompt units)
CREATE TABLE cell_groups (
    group_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    perturbation_name TEXT,
    cell_type_cl_id TEXT,
    cell_type_name TEXT,
    donor_id TEXT NOT NULL,
    n_cells INTEGER NOT NULL,
    has_control BOOLEAN NOT NULL,
    control_group_id TEXT,
    mean_n_genes REAL,
    mean_total_counts REAL
);

-- Perturbation metadata (for Mechanistic Match)
CREATE TABLE perturbation_metadata (
    perturbation_name TEXT PRIMARY KEY,
    perturbation_type TEXT,
    external_ids JSONB,          -- {'chebi': '...', 'drugbank': '...'}
    targets TEXT[],              -- Array of gene symbols
    pathways TEXT[],             -- Array of pathway IDs
    datasets_present TEXT[],
    total_cells INTEGER,
    cell_types_present TEXT[]
);

-- Cell type metadata (for Ontology-Guided)
CREATE TABLE cell_type_metadata (
    cell_type_cl_id TEXT PRIMARY KEY,
    cell_type_name TEXT NOT NULL,
    lineage TEXT[],              -- Array from root to leaf
    datasets_present TEXT[],
    total_cells INTEGER,
    perturbations_present TEXT[]
);

-- Synonym lookup table
CREATE TABLE synonyms (
    canonical_name TEXT NOT NULL,
    synonym TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- 'perturbation' or 'cell_type'
    PRIMARY KEY (canonical_name, synonym)
);
```

### 11.4 Index Configuration

```sql
-- Scalar indexes for filtered queries
CREATE INDEX idx_cells_dataset ON cells(dataset);
CREATE INDEX idx_cells_cell_type ON cells(cell_type_cl_id);
CREATE INDEX idx_cells_perturbation ON cells(perturbation_name);
CREATE INDEX idx_cells_tissue ON cells(tissue_uberon_id);
CREATE INDEX idx_cells_is_control ON cells(is_control);
CREATE INDEX idx_cells_group ON cells(group_id);
CREATE INDEX idx_cells_donor ON cells(donor_id);

-- HNSW vector indexes for similarity search
-- Note: Build these AFTER loading data for best performance

-- Perturbation embedding index (for semantic perturbation search)
CREATE INDEX idx_cells_perturbation_embedding ON cells 
USING hnsw (perturbation_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Cell type embedding index (for semantic cell type search)
CREATE INDEX idx_cells_cell_type_embedding ON cells 
USING hnsw (cell_type_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index on synonym table
CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
CREATE INDEX idx_synonyms_type ON synonyms(entity_type);
```

### 11.5 Read-Only Agent Role

```sql
-- Create read-only role for agent access
CREATE ROLE haystack_agent WITH LOGIN PASSWORD 'secure_password';

-- Grant read-only access to all tables
GRANT CONNECT ON DATABASE haystack TO haystack_agent;
GRANT USAGE ON SCHEMA public TO haystack_agent;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO haystack_agent;

-- Prevent any writes
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT ON TABLES TO haystack_agent;

-- Set statement timeout to prevent runaway queries (30 seconds)
ALTER ROLE haystack_agent SET statement_timeout = '30s';

-- Set work_mem for efficient vector operations
ALTER ROLE haystack_agent SET work_mem = '256MB';
```

### 11.6 Python Database Client

```python
import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager


class HaystackDatabase:
    """Unified database client for HAYSTACK."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "haystack",
        user: str = "haystack_agent",
        password: str = "",
    ):
        """
        Initialize database connection.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username (use haystack_agent for read-only access)
            password: Password
        """
        self.conninfo = f"host={host} port={port} dbname={database} user={user} password={password}"
    
    @contextmanager
    def connection(self):
        """Get a database connection."""
        conn = psycopg.connect(self.conninfo, row_factory=dict_row)
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, sql: str, params: dict = None, max_rows: int = 1000) -> dict:
        """
        Execute a read-only SQL query.
        
        Args:
            sql: SQL query (SELECT only)
            params: Query parameters
            max_rows: Maximum rows to return
        
        Returns:
            Dictionary with columns, rows, row_count, truncated
        """
        # Validate read-only
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
            raise ValueError("Only SELECT queries are allowed")
        
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchmany(max_rows + 1)
                columns = [desc.name for desc in cur.description]
                
                truncated = len(rows) > max_rows
                return {
                    "columns": columns,
                    "rows": rows[:max_rows],
                    "row_count": len(rows[:max_rows]),
                    "truncated": truncated,
                }
    
    def vector_search(
        self,
        query_vector: list[float],
        vector_column: str = "perturbation_embedding",
        k: int = 100,
        filters: dict = None,
        ef_search: int = 100,
    ) -> list[dict]:
        """
        Perform vector similarity search.
        
        Args:
            query_vector: Query embedding
            vector_column: Which embedding column to search 
                          ('perturbation_embedding' or 'cell_type_embedding')
            k: Number of results
            filters: Optional metadata filters
            ef_search: HNSW search parameter (higher = better recall, slower)
        
        Returns:
            List of matching cells with distances
        """
        # Build filter clause
        filter_clause = ""
        params = {"query_vec": query_vector, "k": k}
        
        if filters:
            conditions = []
            for i, (col, val) in enumerate(filters.items()):
                param_name = f"filter_{i}"
                if isinstance(val, list):
                    conditions.append(f"{col} = ANY(%({param_name})s)")
                else:
                    conditions.append(f"{col} = %({param_name})s")
                params[param_name] = val
            filter_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
            SET LOCAL hnsw.ef_search = {ef_search};
            
            SELECT 
                cell_id, dataset, cell_type_harmonized, cell_type_cl_id,
                perturbation_name, perturbation_type, donor_id, group_id,
                {vector_column} <=> %(query_vec)s::vector AS distance
            FROM cells
            {filter_clause}
            ORDER BY {vector_column} <=> %(query_vec)s::vector
            LIMIT %(k)s;
        """
        
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
    
    def get_schema(self) -> dict:
        """Get schema information for all tables."""
        sql = """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """
        result = self.execute_query(sql)
        
        schema = {}
        for row in result["rows"]:
            table = row["table_name"]
            if table not in schema:
                schema[table] = []
            schema[table].append({
                "column": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
            })
        return schema
    
    def get_statistics(self) -> dict:
        """Get high-level index statistics."""
        stats = {}
        
        # Cells per dataset
        result = self.execute_query("""
            SELECT dataset, COUNT(*) as n_cells
            FROM cells GROUP BY dataset
        """)
        stats["cells_per_dataset"] = {r["dataset"]: r["n_cells"] for r in result["rows"]}
        
        # Perturbation counts by type
        result = self.execute_query("""
            SELECT perturbation_type, COUNT(DISTINCT perturbation_name) as n
            FROM cells WHERE NOT is_control
            GROUP BY perturbation_type
        """)
        stats["perturbations_by_type"] = {r["perturbation_type"]: r["n"] for r in result["rows"]}
        
        # Cell type count
        result = self.execute_query("""
            SELECT COUNT(DISTINCT cell_type_cl_id) as n FROM cells
            WHERE cell_type_cl_id IS NOT NULL
        """)
        stats["unique_cell_types"] = result["rows"][0]["n"]
        
        # Cell group count
        result = self.execute_query("SELECT COUNT(*) as n FROM cell_groups")
        stats["cell_groups"] = result["rows"][0]["n"]
        
        return stats
```

### 11.7 Agent Query Tool

```python
@tool
def query_index(
    sql: str,
    max_rows: int = 1000,
) -> dict:
    """
    Execute a read-only SQL query against the HAYSTACK database.
    
    Use this tool to explore available data, compute statistics, 
    and validate assumptions before retrieval.
    
    Args:
        sql: SQL query (SELECT only - no INSERT/UPDATE/DELETE)
        max_rows: Maximum rows to return (default 1000, max 10000)
    
    Returns:
        Dictionary with:
        - columns: List of column names
        - rows: List of row dictionaries
        - row_count: Number of rows returned
        - truncated: Whether results were truncated
    
    Available tables:
        - cells: All cells with metadata and embeddings
        - cell_groups: Aggregated cell groups (prompt units)
        - perturbation_metadata: Unique perturbations with targets/pathways
        - cell_type_metadata: Unique cell types with ontology lineage
        - synonyms: Perturbation and cell type synonyms
    
    Example queries:
        -- What perturbations are available?
        SELECT DISTINCT perturbation_name, perturbation_type, dataset
        FROM cells WHERE NOT is_control
        
        -- How many cells per cell type in Parse PBMC?
        SELECT cell_type_harmonized, COUNT(*) as n_cells
        FROM cells WHERE dataset = 'parse_pbmc'
        GROUP BY cell_type_harmonized ORDER BY n_cells DESC
        
        -- Does TGF-beta exist? What cell types have it?
        SELECT cell_type_harmonized, COUNT(*) as n_cells
        FROM cells 
        WHERE perturbation_name ILIKE '%tgf%'
        GROUP BY cell_type_harmonized
        
        -- What perturbations target SMAD genes?
        SELECT perturbation_name, targets
        FROM perturbation_metadata
        WHERE 'SMAD2' = ANY(targets) OR 'SMAD3' = ANY(targets)
        
        -- Find synonyms for a perturbation
        SELECT canonical_name FROM synonyms
        WHERE synonym ILIKE '%interferon%' AND entity_type = 'perturbation'
    
    Notes:
        - Query timeout is 30 seconds
        - Results truncated at max_rows
        - Use ILIKE for case-insensitive pattern matching
        - Use ANY() for array containment checks
    """
    return DATABASE.execute_query(sql, max_rows=min(max_rows, 10000))


@tool
def vector_search(
    query_text: str,
    search_type: Literal["perturbation", "cell_type"] = "perturbation",
    k: int = 50,
    filters: dict = None,
) -> list[dict]:
    """
    Search for similar cells using semantic text embeddings.
    
    This embeds your query text and finds cells with similar
    perturbation or cell type descriptions.
    
    Args:
        query_text: Natural language description to search for
        search_type: "perturbation" or "cell_type" - which embedding to search
        k: Number of results to return
        filters: Optional metadata filters, e.g. {"dataset": "parse_pbmc"}
    
    Returns:
        List of matching cells with similarity scores
    
    Examples:
        # Find cells with perturbations similar to TGF-beta signaling
        vector_search(
            "TGF-beta cytokine targeting SMAD pathway EMT",
            search_type="perturbation"
        )
        
        # Find cells similar to lung fibroblasts
        vector_search(
            "lung fibroblast stromal cell respiratory tissue",
            search_type="cell_type"
        )
        
        # Find with filters
        vector_search(
            "interferon signaling JAK-STAT",
            search_type="perturbation",
            filters={"dataset": "parse_pbmc"}
        )
    """
    # Generate embedding for query text
    query_embedding = EMBEDDING_MODEL.embed(query_text)
    
    # Select embedding column
    vector_column = {
        "perturbation": "perturbation_embedding",
        "cell_type": "cell_type_embedding",
    }[search_type]
    
    return DATABASE.vector_search(
        query_vector=query_embedding,
        vector_column=vector_column,
        k=k,
        filters=filters,
    )


@tool
def describe_index_schema() -> dict:
    """
    Get schema information for all tables in the HAYSTACK database.
    
    Use this to understand what columns are available before writing queries.
    
    Returns:
        Dictionary mapping table names to column definitions
    """
    return DATABASE.get_schema()


@tool
def get_index_statistics() -> dict:
    """
    Get high-level statistics about the HAYSTACK database.
    
    Returns summary counts and distributions without needing to write SQL.
    
    Returns:
        Dictionary with:
        - cells_per_dataset: Count of cells in each atlas
        - perturbations_by_type: Count of unique perturbations by type
        - unique_cell_types: Total unique cell types (by CL ID)
        - cell_groups: Total number of cell groups
    """
    return DATABASE.get_statistics()
```

### 11.8 Example Agent Introspection Flow

```
Agent thinking: "User asked about TGF-beta in lung fibroblasts. 
                Let me check what's available before retrieval."

Step 1: Check if TGF-beta exists
────────────────────────────────
query_index("""
    SELECT DISTINCT perturbation_name, dataset, COUNT(*) as n_cells
    FROM cells 
    WHERE perturbation_name ILIKE '%tgf%beta%'
       OR perturbation_name ILIKE '%tgfb%'
    GROUP BY perturbation_name, dataset
""")

Result:
| perturbation_name | dataset      | n_cells |
|-------------------|--------------|---------|
| TGF-beta-1        | openproblems | 12,450  |

Step 2: Check if lung fibroblasts exist
───────────────────────────────────────
query_index("""
    SELECT DISTINCT cell_type_harmonized, cell_type_cl_id, dataset, COUNT(*) as n_cells
    FROM cells
    WHERE cell_type_harmonized ILIKE '%fibroblast%'
    GROUP BY cell_type_harmonized, cell_type_cl_id, dataset
""")

Result:
| cell_type_harmonized | cell_type_cl_id | dataset         | n_cells |
|----------------------|-----------------|-----------------|---------|
| fibroblast           | CL:0000057      | tabula_sapiens  | 45,230  |
| lung fibroblast      | CL:0002553      | tabula_sapiens  | 8,120   |
| cardiac fibroblast   | CL:0002548      | tabula_sapiens  | 3,450   |

Step 3: Check for TGF-beta + fibroblast combination
───────────────────────────────────────────────────
query_index("""
    SELECT perturbation_name, cell_type_harmonized, dataset, COUNT(*) as n_cells
    FROM cells
    WHERE perturbation_name ILIKE '%tgf%'
      AND cell_type_cl_id IN ('CL:0000057', 'CL:0002553')
    GROUP BY perturbation_name, cell_type_harmonized, dataset
""")

Result: (empty - no TGF-beta treated fibroblasts exist)

Step 4: Find mechanistically similar perturbations
──────────────────────────────────────────────────
query_index("""
    SELECT perturbation_name, targets, pathways
    FROM perturbation_metadata
    WHERE 'SMAD2' = ANY(targets) OR 'SMAD3' = ANY(targets)
       OR 'TGFBR1' = ANY(targets)
""")

Result:
| perturbation_name | targets                    | pathways              |
|-------------------|----------------------------|-----------------------|
| BMP4              | [BMPR1A, SMAD1, SMAD5]     | [KEGG:hsa04350, ...]  |
| Activin A         | [ACVR1, SMAD2, SMAD3]      | [KEGG:hsa04350, ...]  |

Agent conclusion: "No direct match for TGF-beta + fibroblasts. TGF-beta-1 exists 
                  in OpenProblems but only in non-fibroblast cell types. Lung 
                  fibroblasts exist in Tabula Sapiens but unperturbed. 
                  
                  Found mechanistically similar perturbations: BMP4 and Activin A 
                  share SMAD pathway targets. Will use these for Mechanistic Match."
```

### 11.9 Local PostgreSQL Setup

For local development, use Docker:

```bash
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: haystack_db
    environment:
      POSTGRES_DB: haystack
      POSTGRES_USER: haystack_admin
      POSTGRES_PASSWORD: admin_password
    ports:
      - "5432:5432"
    volumes:
      - haystack_data:/var/lib/postgresql/data
    shm_size: '2gb'  # Required for HNSW index builds

volumes:
  haystack_data:
```

Or install locally:

```bash
# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# macOS with Homebrew
brew install postgresql@16
brew install pgvector
```

### 11.10 Memory and Performance Tuning

For 10M cells with two 1536-dimensional text embeddings:

```sql
-- postgresql.conf settings for HAYSTACK workload

-- Memory (adjust based on available RAM)
shared_buffers = '8GB'           -- 25% of RAM
effective_cache_size = '24GB'    -- 75% of RAM
work_mem = '256MB'               -- Per-operation memory
maintenance_work_mem = '2GB'     -- For index builds

-- HNSW-specific
-- Higher ef_search = better recall, slower queries
-- Default: 40, recommend: 100 for research use
-- Set per-session: SET hnsw.ef_search = 100;

-- Parallel query
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4  -- For index builds
```

---

## 12. Open Questions

### 12.1 Resolved in This Document

| Question | Resolution |
|----------|------------|
| How does Direct Match search atlas metadata? | PostgreSQL SQL queries with scalar indexes |
| What metadata must be available? | Defined in `HarmonizedCellMetadata` schema |
| How is metadata harmonized? | At index build time using dataset-specific rules |
| Can the database have multiple embeddings per record? | Yes - PostgreSQL + pgvector supports multiple vector columns |
| Does Mechanistic Match chain to Direct Match? | Yes - it generates perturbation filters |
| How does Semantic Match generate queries? | Text embeddings of perturbation/cell type descriptions (not STACK) |
| Does Ontology-Guided chain to other strategies? | Yes - it generates cell type filters for Direct Match |
| How can agents flexibly query the index? | Full SQL via PostgreSQL with read-only role (Section 11) |
| What database supports both SQL and vector search at scale? | PostgreSQL + pgvector (proven at 10M+ vectors) |

### 12.2 Remaining Open Questions

| Question | Options | Recommendation |
|----------|---------|----------------|
| **How to handle novel perturbations not in any atlas?** | (a) Mechanistic only, (b) Semantic only, (c) Fail gracefully | Combine Mechanistic + Semantic; if both fail, use unperturbed cells from same cell type as control baseline |
| **Should we pre-compute mechanistic similarity matrix?** | (a) Pre-compute all pairs, (b) Compute on demand | Pre-compute at index time for perturbations with >100 cells; compute on demand for rare perturbations |
| **How to weight ontology distance in ranking?** | Linear, exponential, or threshold-based decay | Exponential decay: `weight = 0.9 ** distance` |
| **What if query cell type exists in Tabula Sapiens but not perturbed atlases?** | (a) Use TS cells as context only, (b) Find closest perturbed cell type | Use TS cells as context; find perturbed cells from closest ontology relative |
| **How to handle donor effects?** | (a) Ignore donors, (b) Prefer same donor for prompt+context, (c) Prefer diversity | Context: prefer same donor; Prompt: allow cross-donor if needed |
| **Maximum cells per prompt group?** | Fixed limit vs adaptive | Adaptive based on STACK context window; recommend max 500 cells per group |
| **How to validate index quality?** | Manual spot checks, automated tests | Both: manual review of 50 random groups + automated consistency checks |

### 12.3 Questions for Domain Expert Review

1. **Biological validity of mechanistic similarity**: Is Jaccard similarity over targets/pathways a reasonable proxy for functional similarity between perturbations?

2. **Cell Ontology distance interpretation**: Does an ontology distance of 2 (e.g., "macrophage" → "myeloid cell" → "monocyte") represent biologically similar enough cells for prompt transfer?

3. **Text embedding quality**: Will text embeddings of perturbation descriptions (e.g., "TGF-beta (cytokine) targeting TGFBR1, SMAD2 affecting TGF-beta signaling") capture meaningful biological similarity?

4. **Missing perturbation handling**: When a query perturbation has no exact match AND no mechanistically similar perturbation, is it better to:
   - Use the closest semantic match (potentially unrelated mechanism)?
   - Use unperturbed cells as a "null prompt"?
   - Refuse to make predictions?

---

## Appendix A: Example Query Walkthrough

### Query: "How would lung fibroblasts respond to TGF-beta treatment?"

#### Step 1: Query Understanding
```python
StructuredQuery(
    query_cell_type="lung fibroblast",
    query_cell_type_cl_id="CL:0002553",  # Resolved via Cell Ontology
    query_tissue="lung",
    perturbation="TGF-beta",
    perturbation_type=PerturbationType.CYTOKINE,
    perturbation_id="CHEBI:79226",
    target_genes=["TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4"],
    expected_pathways=["KEGG:hsa04350", "REACTOME:R-HSA-2173789"],
)
```

#### Step 2: Strategy Execution (Parallel)

**Direct Match:**
- Search: `perturbation_name = 'TGF-beta' AND cell_type_cl_id = 'CL:0002553'`
- Result: No exact match (Parse PBMC doesn't have fibroblasts)

**Mechanistic Match:**
- Find perturbations sharing TGFBR1/SMAD targets
- Results: BMP4 (shares SMAD pathway), Activin A (shares SMAD pathway)
- Chain to Direct Match: Find BMP4-treated cells in Parse PBMC
- Result: BMP4-treated monocytes (CL:0000576)

**Semantic Match (Perturbation):**
- Query: "TGF-beta (cytokine) targeting TGFBR1, SMAD2 affecting TGF-beta signaling, EMT"
- Search `perturbation_embedding`
- Results: BMP4, Activin A, TGF-beta-1 (OpenProblems)

**Semantic Match (Cell Type):**
- Query: "lung fibroblast from lung (lineage: stromal cell > fibroblast)"
- Search `cell_type_embedding`
- Results: Fibroblasts from Tabula Sapiens (unperturbed)

**Ontology-Guided:**
- Query CL ID: CL:0002553 (lung fibroblast)
- Related: CL:0000057 (fibroblast, parent), CL:0000192 (smooth muscle cell, sibling)
- Chain to Direct Match with cell type filter
- Result: Find any perturbation applied to fibroblasts or smooth muscle cells

#### Step 3: Candidate Ranking

| Rank | Group | Strategy | Score | Rationale |
|------|-------|----------|-------|-----------|
| 1 | TGF-beta-1 + T cells (OP) | semantic_perturbation | 0.82 | Exact perturbation, different cell type |
| 2 | BMP4 + monocytes (Parse) | mechanistic | 0.78 | Shares SMAD pathway |
| 3 | Activin A + monocytes (Parse) | mechanistic | 0.75 | Shares SMAD pathway |
| 4 | Unperturbed fibroblasts (TS) | ontology | 0.65 | Exact cell type, no perturbation |

#### Step 4: Selection

Select top 2 diverse candidates:
1. TGF-beta-1 + T cells (exact perturbation match)
2. Unperturbed fibroblasts (exact cell type match for context)

---

## Appendix B: PostgreSQL Query Examples

```sql
-- Direct Match: exact perturbation + cell type
SELECT * FROM cells
WHERE perturbation_name = 'TGF-beta' 
  AND cell_type_cl_id = 'CL:0002553'
LIMIT 100;

-- Mechanistic Match: filter by perturbation list
SELECT * FROM cells
WHERE perturbation_name IN ('BMP4', 'Activin A', 'TGF-beta-1')
LIMIT 100;

-- Semantic Match: vector search on perturbation embedding
-- (query_embedding is a parameter with 1536 dimensions)
SELECT *, perturbation_embedding <=> $1 AS distance
FROM cells
ORDER BY perturbation_embedding <=> $1
LIMIT 50;

-- Combined: semantic search with cell type filter
SELECT *, perturbation_embedding <=> $1 AS distance
FROM cells
WHERE cell_type_cl_id IN ('CL:0000057', 'CL:0002553')
ORDER BY perturbation_embedding <=> $1
LIMIT 50;

-- Ontology-guided: cell type filter with any perturbation
SELECT * FROM cells
WHERE cell_type_cl_id IN ('CL:0000057', 'CL:0000192') 
  AND NOT is_control
LIMIT 100;

-- Aggregation queries (not possible in LanceDB, easy in PostgreSQL)
SELECT cell_type_harmonized, perturbation_name, COUNT(*) as n_cells
FROM cells
WHERE dataset = 'parse_pbmc' AND NOT is_control
GROUP BY cell_type_harmonized, perturbation_name
ORDER BY n_cells DESC;

-- Find perturbations sharing targets
SELECT perturbation_name, targets
FROM perturbation_metadata
WHERE 'SMAD2' = ANY(targets) OR 'TGFBR1' = ANY(targets);
```