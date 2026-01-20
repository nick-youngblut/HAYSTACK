# Sprint 02: Core Backend Services

**Duration**: 1-2 weeks  
**Dependencies**: Sprint 01 (Foundation & Infrastructure)  
**Goal**: Implement shared data models, database client, GCS service, and external API integrations.

---

## Overview

This sprint implements the core backend services:
- Shared Pydantic data models
- Async database client with connection pooling
- GCS service for artifact storage
- External API clients (biological databases, literature search)

---

## Phase 1: Shared Data Models

> **Spec Reference**: `./specification/data-models.md`

### Task 1.1: Implement Core Enums

- [ ] **1.1.1** Create `shared/models/enums.py`:

```python
from enum import Enum

class PerturbationType(str, Enum):
    DRUG = "drug"
    CYTOKINE = "cytokine"
    GENETIC = "genetic"
    UNKNOWN = "unknown"

class ICLTaskType(str, Enum):
    PERTURBATION_NOVEL_CELL_TYPES = "perturbation_novel_cell_types"
    PERTURBATION_NOVEL_SAMPLES = "perturbation_novel_samples"
    CELL_TYPE_IMPUTATION = "cell_type_imputation"
    DONOR_EXPRESSION_PREDICTION = "donor_expression_prediction"
    CROSS_DATASET_GENERATION = "cross_dataset_generation"

class ControlStrategy(str, Enum):
    QUERY_AS_CONTROL = "query_as_control"
    SYNTHETIC_CONTROL = "synthetic_control"

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RunPhase(str, Enum):
    PENDING = "pending"
    QUERY_ANALYSIS = "query_analysis"
    PROMPT_GENERATION = "prompt_generation"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    OUTPUT_GENERATION = "output_generation"
```

**Acceptance Criteria**: All enums match specification exactly

---

### Task 1.2: Implement Query Models

- [ ] **1.2.1** Create `shared/models/queries.py` with `StructuredQuery`:
  - Raw query, task type, cell type resolution
  - Perturbation resolution with external IDs
  - Observational context (donor, tissue, disease)
  - Control strategy fields

- [ ] **1.2.2** Add validation for field combinations per task type

---

### Task 1.3: Implement Cell Set Models

- [ ] **1.3.1** Create `CellSetCandidate` model with:
  - Dataset, perturbation, cell type, donor, tissue fields
  - Cell indices array, quality metrics
  - Strategy type and relevance score
  - `selection_key()` method for deduplication

- [ ] **1.3.2** Create `PromptCandidate` model with:
  - Cell set reference
  - Paired control indices for synthetic control
  - Scoring metadata

---

### Task 1.4: Implement Grounding Score Models

> **Spec Reference**: `./specification/data-models.md` (GroundingScore section)

- [ ] **1.4.1** Create `GroundingScore` for perturbational tasks:
  - pathway_coherence, target_activation, literature_support, network_coherence (1-10)
  - composite_score, enriched_pathways, DE genes, improvement suggestions

- [ ] **1.4.2** Create `ObservationalGroundingScore` for observational tasks:
  - marker_gene_expression, tissue_signature_match, donor_effect_capture, cell_type_coherence (1-10)
  - Detected markers, tissue genes, donor signature

---

### Task 1.5: Implement Run Models

- [ ] **1.5.1** Create `IterationRecord` with control strategy tracking
- [ ] **1.5.2** Create `HaystackRun` with full run state
- [ ] **1.5.3** Create API request/response models:
  - `CreateRunRequest`, `RunStatusResponse`, `RunResultResponse`
  - `ControlStrategyValidationRequest`, `ControlStrategyValidation`

---

## Phase 2: Database Client

> **Spec Reference**: `./specification/database.md` (Section 8.4)

### Task 2.1: Implement HaystackDatabase Class

- [ ] **2.1.1** Create `orchestrator/services/database.py`:

```python
class HaystackDatabase:
    """Async database client using Cloud SQL Python Connector."""
    
    async def connect(self) -> None
    async def close(self) -> None
    async def execute_query(self, sql: str, params: tuple) -> list[dict]
    
    # Run management
    async def create_run(self, run: CreateRunRequest) -> str
    async def get_run(self, run_id: str) -> dict | None
    async def update_run(self, run_id: str, **kwargs) -> None
    async def list_runs(self, user_email: str, **filters) -> list[dict]
    
    # Cell queries
    async def search_cells_by_perturbation(self, name: str, **filters) -> list[dict]
    async def search_cells_by_cell_type(self, cl_id: str, **filters) -> list[dict]
    async def search_cells_by_donor(self, donor_id: str, **filters) -> list[dict]
    
    # Vector search
    async def semantic_search_cells(self, embedding: list, search_type: str, top_k: int) -> list[dict]
    
    # Metadata
    async def list_perturbations(self, dataset: str = None) -> list[dict]
    async def list_cell_types(self, dataset: str = None) -> list[dict]
```

- [ ] **2.1.2** Implement connection pooling with asyncpg
- [ ] **2.1.3** Use Cloud SQL Python Connector for secure connections
- [ ] **2.1.4** Add query timeout handling (30s for agent queries)
- [ ] **2.1.5** Add structured logging with run_id context

---

### Task 2.2: Implement Vector Search Methods

- [ ] **2.2.1** Implement cosine similarity search:

```python
async def semantic_search_cells(
    self,
    query_embedding: list[float],
    search_type: Literal["perturbation", "cell_type", "sample_context"],
    top_k: int = 50,
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """Vector similarity search using pgvector."""
    column_map = {
        "perturbation": "perturbation_embedding",
        "cell_type": "cell_type_embedding",
        "sample_context": "sample_context_embedding",
    }
    column = column_map[search_type]
    
    sql = f"""
        SELECT cell_index, dataset, perturbation_name, cell_type_name,
               1 - ({column} <=> $1::vector) as similarity
        FROM cells
        WHERE {column} IS NOT NULL
          AND 1 - ({column} <=> $1::vector) >= $2
        ORDER BY {column} <=> $1::vector
        LIMIT $3
    """
    return await self.execute_query(sql, (query_embedding, similarity_threshold, top_k))
```

- [ ] **2.2.2** Add index hints for HNSW usage
- [ ] **2.2.3** Test query performance with EXPLAIN ANALYZE

---

## Phase 3: GCS Service

> **Spec Reference**: `./specification/architecture.md` (Data Layer)

### Task 3.1: Implement GCSService Class

- [ ] **3.1.1** Create `orchestrator/services/gcs.py`:

```python
class GCSService:
    """Async GCS client for artifact storage."""
    
    async def upload_anndata(self, adata: AnnData, path: str) -> str
    async def download_anndata(self, path: str) -> AnnData
    async def generate_signed_url(self, path: str, expiration: int = 3600) -> str
    async def write_json(self, data: dict, path: str) -> None
    async def read_json(self, path: str) -> dict
    async def write_markdown(self, content: str, path: str) -> None
    async def delete_path(self, path: str) -> None
    async def list_objects(self, prefix: str) -> list[str]
```

- [ ] **3.1.2** Handle H5AD file serialization/deserialization
- [ ] **3.1.3** Implement retry logic for transient failures
- [ ] **3.1.4** Add progress callbacks for large uploads

---

### Task 3.2: Implement Signed URL Generation

- [ ] **3.2.1** Generate signed URLs for result downloads:
  - Default expiration: 1 hour
  - Support for AnnData, report, and log files

---

## Phase 4: External API Clients

### Task 4.1: Implement Biological Knowledge APIs

> **Spec Reference**: `./specification/tools.md` (Section 6.3)

- [ ] **4.1.1** Create `orchestrator/services/biological_apis.py`:

```python
class BiologicalKnowledgeService:
    """Client for biological knowledge databases."""
    
    # KEGG
    async def get_kegg_pathway(self, pathway_id: str) -> dict
    async def get_kegg_drug_targets(self, drug_id: str) -> list[str]
    
    # Reactome
    async def get_reactome_pathway(self, pathway_id: str) -> dict
    async def run_pathway_enrichment(self, gene_list: list[str]) -> list[dict]
    
    # UniProt
    async def get_protein_info(self, uniprot_id: str) -> dict
    async def get_protein_targets(self, drug_name: str) -> list[dict]
    
    # PubChem
    async def resolve_compound(self, name: str) -> dict
    async def get_compound_targets(self, cid: int) -> list[str]
    
    # DrugBank
    async def get_drug_info(self, drugbank_id: str) -> dict
```

- [ ] **4.1.2** Implement rate limiting for each API
- [ ] **4.1.3** Add caching for repeated lookups
- [ ] **4.1.4** Handle API failures gracefully

---

### Task 4.2: Implement Literature Search Service

> **Spec Reference**: `./specification/literature-search.md`

- [ ] **4.2.1** Create `orchestrator/services/literature.py`:

```python
class LiteratureSearchService:
    """Multi-source literature search with PDF acquisition."""
    
    # Search
    async def search_pubmed(self, query: str, max_results: int) -> list[PaperRecord]
    async def search_semantic_scholar(self, query: str, max_results: int) -> list[PaperRecord]
    async def search_biorxiv(self, query: str, max_results: int) -> list[PaperRecord]
    async def search_all(self, query: str, max_results: int = 10) -> list[PaperRecord]
    
    # Acquisition
    async def acquire_full_text(self, doi: str) -> str | None
    async def get_pdf_from_preprint(self, arxiv_id: str) -> bytes | None
    async def get_pdf_from_core(self, doi: str) -> bytes | None
    async def get_pdf_from_unpaywall(self, doi: str) -> bytes | None
    
    # Processing
    async def convert_pdf_to_markdown(self, pdf_bytes: bytes) -> str
```

- [ ] **4.2.2** Implement PubMed E-utilities client
- [ ] **4.2.3** Implement Semantic Scholar Graph API client
- [ ] **4.2.4** Implement bioRxiv/medRxiv API client
- [ ] **4.2.5** Implement PDF acquisition pipeline (CORE → Europe PMC → Unpaywall)
- [ ] **4.2.6** Integrate docling for PDF-to-markdown conversion

---

### Task 4.3: Implement Pathway Enrichment

- [ ] **4.3.1** Integrate gseapy for enrichment analysis:

```python
async def run_enrichment_analysis(
    gene_list: list[str],
    gene_sets: list[str] = ["GO_Biological_Process", "KEGG_2021", "Reactome_2022"],
    organism: str = "human",
) -> list[EnrichmentResult]:
    """Run gene set enrichment analysis."""
    import gseapy as gp
    
    results = []
    for gene_set in gene_sets:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_set,
            organism=organism,
        )
        for _, row in enr.results.iterrows():
            if row["Adjusted P-value"] < 0.05:
                results.append(EnrichmentResult(
                    term=row["Term"],
                    gene_set=gene_set,
                    p_value=row["P-value"],
                    adjusted_p_value=row["Adjusted P-value"],
                    genes=row["Genes"].split(";"),
                ))
    return results
```

---

## Phase 5: Cell Ontology Service

> **Spec Reference**: `./specification/ontology-resolution.md`

### Task 5.1: Implement CellOntologyService

- [ ] **5.1.1** Create `orchestrator/services/ontology.py`:

```python
class CellOntologyService:
    """Native Cell Ontology service using database + OLS fallback."""
    
    _instance: Optional["CellOntologyService"] = None
    
    @classmethod
    def get_instance(cls) -> "CellOntologyService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def semantic_search(
        self,
        labels: list[str],
        k: int = 3,
        distance_threshold: float = 0.7,
    ) -> dict[str, list[dict]]:
        """Search for CL terms using semantic similarity."""
        # Generate embedding for each label
        # Query ontology_terms table with vector similarity
        # Return top-k matches per label
    
    async def get_neighbors(
        self,
        term_ids: list[str],
        relationship_types: list[str] = ["is_a", "part_of", "develops_from"],
        max_distance: int = 2,
    ) -> list[dict]:
        """Get related terms via ontology graph traversal."""
        # Query ontology_relationships table
        # BFS/DFS to specified depth
        # Return neighbors with relationship type and distance
    
    async def query_ols(
        self,
        search_terms: list[str],
    ) -> dict[str, list[dict]]:
        """Fallback to EBI OLS API for term lookup."""
        # Call https://www.ebi.ac.uk/ols4/api/search
        # Parse and return CL matches
```

- [ ] **5.1.2** Implement embedding generation for ontology queries
- [ ] **5.1.3** Implement OLS API client with rate limiting
- [ ] **5.1.4** Add caching for frequently accessed terms

---

## Phase 6: Configuration Management

> **Spec Reference**: `./specification/configuration.md`

### Task 5.1: Set Up Dynaconf

- [ ] **5.1.1** Create `shared/config.py`:

```python
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="HAYSTACK",
    settings_files=["settings.yaml", ".secrets.yaml"],
    environments=True,
    env_switcher="HAYSTACK_ENV",
)
```

- [ ] **5.1.2** Create `settings.yaml` with defaults:
  - Database connection settings
  - GCS bucket names
  - LLM provider settings
  - API timeouts and rate limits

- [ ] **5.1.3** Create environment-specific overrides (dev, staging, prod)
- [ ] **5.1.4** Document all configuration options

---

## Definition of Done

- [ ] All Pydantic models implemented and tested
- [ ] Database client with connection pooling operational
- [ ] Vector search methods verified with EXPLAIN ANALYZE
- [ ] GCS service handles all artifact types
- [ ] External API clients with rate limiting
- [ ] Literature search pipeline functional
- [ ] Configuration management via Dynaconf
- [ ] Unit tests for all services (>80% coverage)

---

## Next Sprint

**Sprint 03: Prompt Retrieval Strategies** - Implement the retrieval strategy framework and all six strategies.
