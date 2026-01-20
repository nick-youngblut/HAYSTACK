# HAYSTACK Implementation Plan

**H**euristic **A**gent for **Y**ielding **S**TACK-**T**uned **A**ssessments with **C**losed-loop **K**nowledge

*Finding the optimal prompt in a haystack of possibilities*

---

## Executive Summary

HAYSTACK transforms STACK (single-cell foundation model) from open-loop inference into closed-loop optimization. Given natural language queries, it iteratively generates biologically-informed prompts, executes STACK inference, and evaluates predictions against pathway databases and literature until convergence.

**Architecture**: Two-tier GCP Batch
- **Cloud Run**: FastAPI API + Next.js frontend (static export)
- **CPU Batch**: Orchestrator agent (LangChain/DeepAgents) running iterative workflow
- **GPU Batch**: STACK inference (A100 80GB)

**Data**: ~10M cells across Parse PBMC, OpenProblems, Tabula Sapiens atlases in Cloud SQL (PostgreSQL + pgvector)

---

## Specification Alignment Notes

This implementation plan has been validated against the HAYSTACK specification documents. Key alignment points:

### Task Types
- **5 ICL task types** (per `specification/data-models.md`):
  1. `perturbation_novel_cell_types` - Predict perturbation effects on new cell types
  2. `perturbation_novel_samples` - Predict perturbation effects in novel samples
  3. `cell_type_imputation` - Impute cell type expression
  4. `donor_expression_prediction` - Predict donor-specific expression
  5. `cross_dataset_generation` - Generate cell types across datasets

### Grounding Evaluation
- **Dual scoring models** (per `specification/data-models.md`):
  - `GroundingScore` for perturbational tasks (pathway_coherence, target_activation, literature_support, network_coherence)
  - `ObservationalGroundingScore` for observational tasks (marker_gene_expression, tissue_signature_match, donor_effect_capture, cell_type_coherence)

### Retrieval Strategy Priority
- **Task-specific priority lists** (per `specification/prompt-retrieval.md`):
  - Perturbational: DirectMatch → Mechanistic → Semantic → Ontology
  - Observational: DonorContext → TissueAtlas → Ontology → Semantic

### Ranking Weights
- **Corrected weights** (per `specification/prompt-retrieval.md`):
  - Relevance: 0.4 (not 0.5)
  - Quality: 0.3 (not 0.2)
  - Diversity: 0.3 (unchanged)

### Control Strategy
- **Two strategies** (per `specification/data-models.md`):
  - `synthetic_control`: Paired prompts, 2x inference, higher confidence
  - `query_as_control`: Single inference, mild confidence penalty
- **Fallback logic**: If synthetic control unavailable, fallback to query-as-control

### Embeddings
- **Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Stored in**: `ontology_terms.embedding`, `cells.perturbation_embedding`, etc.

---

## Technology Stack Summary

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, LangChain, DeepAgents, asyncpg, Pydantic, Dynaconf |
| **Frontend** | Next.js 14+, TypeScript, TanStack Query, Zustand, Tailwind CSS |
| **Database** | PostgreSQL 15 + pgvector (Cloud SQL) |
| **Storage** | Google Cloud Storage |
| **Compute** | Cloud Run (API), GCP Batch (CPU orchestrator, GPU inference) |
| **ML** | STACK foundation model, OpenAI embeddings (text-embedding-3-small) |
| **External APIs** | PubMed, Semantic Scholar, KEGG, Reactome, SendGrid |

---

## PHASE 1: Foundation & Infrastructure

### 1.1 Database Setup

> **Spec Reference**: `./specification/database.md`

- [ ] **Set up GCP Cloud SQL (PostgreSQL 15 + pgvector)**
  - Instance: `db-custom-4-15360` (4 vCPU, 15 GB RAM)
  - Region: `us-east1`
  - Storage: 100 GB SSD with auto-resize
  - Private IP via VPC connector
  - High availability enabled for production

- [ ] **Create schema from specifications**
  ```sql
  -- Enable pgvector extension
  CREATE EXTENSION IF NOT EXISTS vector;

  -- Core cells table (~10M rows)
  CREATE TABLE cells (
      id SERIAL PRIMARY KEY,
      cell_index INT NOT NULL,
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
      
      -- External IDs
      perturbation_external_ids JSONB DEFAULT '{}',
      perturbation_targets TEXT[],
      perturbation_pathways TEXT[],
      
      -- Quality metrics
      n_genes INT,
      total_counts FLOAT,
      
      -- Text embeddings (OpenAI text-embedding-3-small, 1536 dim)
      perturbation_embedding vector(1536),
      cell_type_embedding vector(1536),
      sample_context_embedding vector(1536),
      
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  CREATE TABLE donors (
      donor_id VARCHAR(64) PRIMARY KEY,
      dataset VARCHAR(32) NOT NULL,
      age_category VARCHAR(32),
      sex VARCHAR(16),
      disease_states TEXT[],
      disease_names TEXT[],
      tissue_types TEXT[],
      n_cells INT,
      cell_types_present TEXT[],
      clinical_embedding vector(1536)
  );
  
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
  
  CREATE TABLE cell_types (
      cell_type_cl_id VARCHAR(32) PRIMARY KEY,
      cell_type_name VARCHAR(256) NOT NULL,
      lineage_cl_ids TEXT[],
      lineage_names TEXT[],
      datasets_present TEXT[],
      total_cells INT,
      perturbations_present TEXT[]
  );
  
  CREATE TABLE conditions (
      condition_id VARCHAR(64) PRIMARY KEY,
      condition_type VARCHAR(32) NOT NULL,
      perturbation_name VARCHAR(256),
      perturbation_type VARCHAR(32),
      targets TEXT[],
      pathways TEXT[],
      disease_mondo_id VARCHAR(32),
      tissue_uberon_id VARCHAR(32),
      clinical_attributes JSONB DEFAULT '{}',
      condition_embedding vector(1536)
  );
  
  CREATE TABLE synonyms (
      id SERIAL PRIMARY KEY,
      canonical_name VARCHAR(256) NOT NULL,
      synonym VARCHAR(256) NOT NULL,
      entity_type VARCHAR(32) NOT NULL  -- 'perturbation' or 'cell_type'
  );
  
  -- Ontology tables
  CREATE TABLE ontology_terms (
      id SERIAL PRIMARY KEY,
      term_id VARCHAR(32) NOT NULL,
      name VARCHAR(512) NOT NULL,
      definition TEXT,
      ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
      version VARCHAR(16) NOT NULL,
      embedding vector(1536),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      CONSTRAINT uq_ontology_terms_id_version UNIQUE (term_id, ontology_type, version)
  );
  
  CREATE TABLE ontology_relationships (
      id SERIAL PRIMARY KEY,
      subject_term_id VARCHAR(32) NOT NULL,
      object_term_id VARCHAR(32) NOT NULL,
      relationship_type VARCHAR(64) NOT NULL,  -- is_a, part_of, develops_from
      ontology_type VARCHAR(32) NOT NULL DEFAULT 'cell',
      version VARCHAR(16) NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      CONSTRAINT uq_ontology_rels UNIQUE (subject_term_id, object_term_id, relationship_type, ontology_type, version)
  );
  
  -- Run history table
  CREATE TABLE runs (
      run_id VARCHAR(64) PRIMARY KEY,
      user_email VARCHAR(256),
      status VARCHAR(32) NOT NULL,  -- pending, running, completed, failed, cancelled
      batch_job_name VARCHAR(256),
      control_strategy VARCHAR(32) DEFAULT 'synthetic_control',
      control_strategy_effective VARCHAR(32),
      control_cells_available BOOLEAN DEFAULT FALSE,
      
      raw_query TEXT NOT NULL,
      structured_query JSONB,
      config JSONB NOT NULL,
      random_seed INT,
      
      start_time TIMESTAMP NOT NULL,
      end_time TIMESTAMP,
      current_phase VARCHAR(32),
      current_iteration INT DEFAULT 0,
      
      iterations JSONB DEFAULT '[]',
      final_score INT,
      termination_reason VARCHAR(256),
      error_message TEXT,
      
      output_anndata_path VARCHAR(512),
      output_report_path VARCHAR(512),
      output_log_path VARCHAR(512),
      
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

- [ ] **Add vector columns and HNSW indexes**
  ```sql
  -- Cell table indexes
  CREATE INDEX idx_cells_dataset ON cells(dataset);
  CREATE INDEX idx_cells_cell_type ON cells(cell_type_cl_id);
  CREATE INDEX idx_cells_perturbation ON cells(perturbation_name);
  CREATE INDEX idx_cells_tissue ON cells(tissue_uberon_id);
  CREATE INDEX idx_cells_is_control ON cells(is_control);
  CREATE INDEX idx_cells_donor ON cells(donor_id);
  CREATE INDEX idx_cells_disease ON cells(disease_mondo_id);
  CREATE INDEX idx_cells_condition ON cells(sample_condition);
  CREATE INDEX idx_cells_pert_ct_donor ON cells(perturbation_name, cell_type_cl_id, donor_id);
  CREATE INDEX idx_cells_tissue_ct ON cells(tissue_uberon_id, cell_type_cl_id);
  CREATE INDEX idx_cells_donor_ct ON cells(donor_id, cell_type_cl_id);
  
  -- HNSW vector indexes (build AFTER loading data)
  CREATE INDEX idx_cells_perturbation_embedding ON cells 
  USING hnsw (perturbation_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  
  CREATE INDEX idx_cells_cell_type_embedding ON cells 
  USING hnsw (cell_type_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  
  CREATE INDEX idx_cells_sample_context_embedding ON cells 
  USING hnsw (sample_context_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  
  -- Ontology indexes
  CREATE INDEX idx_ontology_terms_term_id ON ontology_terms(term_id);
  CREATE INDEX idx_ontology_terms_name ON ontology_terms(name);
  CREATE INDEX idx_ontology_terms_name_lower ON ontology_terms(LOWER(name));
  CREATE INDEX idx_ontology_terms_type_version ON ontology_terms(ontology_type, version);
  
  CREATE INDEX idx_ontology_terms_embedding ON ontology_terms
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
  
  -- Relationship indexes for graph traversal
  CREATE INDEX idx_ontology_rels_subject ON ontology_relationships(subject_term_id);
  CREATE INDEX idx_ontology_rels_object ON ontology_relationships(object_term_id);
  CREATE INDEX idx_ontology_rels_type ON ontology_relationships(relationship_type);
  CREATE INDEX idx_ontology_rels_subject_version ON ontology_relationships(
      subject_term_id, ontology_type, version);
  CREATE INDEX idx_ontology_rels_object_version ON ontology_relationships(
      object_term_id, ontology_type, version);
  
  -- Other indexes
  CREATE INDEX idx_synonyms_synonym ON synonyms(synonym);
  CREATE INDEX idx_synonyms_synonym_lower ON synonyms(LOWER(synonym));
  CREATE INDEX idx_synonyms_type ON synonyms(entity_type);
  CREATE INDEX idx_perturbations_targets ON perturbations USING GIN(targets);
  CREATE INDEX idx_perturbations_pathways ON perturbations USING GIN(pathways);
  CREATE INDEX idx_cell_types_lineage ON cell_types USING GIN(lineage_cl_ids);
  CREATE INDEX idx_donors_dataset ON donors(dataset);
  CREATE INDEX idx_conditions_type ON conditions(condition_type);
  CREATE INDEX idx_runs_user ON runs(user_email);
  CREATE INDEX idx_runs_status ON runs(status);
  CREATE INDEX idx_runs_created ON runs(created_at DESC);
  ```

- [ ] **Create database roles**
  - `haystack_app`: Read-write for runs, read-only for cells
  - `haystack_agent`: Read-only for all tables (30s statement timeout)
  - `haystack_admin`: Full access

- [ ] **Create GCS buckets**
  - `haystack-atlases`: Atlas H5AD files
  - `haystack-models`: STACK model checkpoints
  - `haystack-batch-io`: Batch job input/output
  - `haystack-results`: Final output artifacts

### 1.2 Atlas Data Processing

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 2: Atlas Data Model)

- [ ] **Download atlas datasets**
  - Parse PBMC (~10M cells, 90 cytokines, 12 donors)
  - OpenProblems (~500K cells, 147 drugs, 3 donors)
  - Tabula Sapiens (~500K cells, unperturbed, 25 tissues, 24 donors)

- [ ] **Implement harmonization pipeline**
  
  | Dataset | Cell Type | Perturbation | Donor ID |
  |---------|-----------|--------------|----------|
  | Parse PBMC | Lookup → CL ID | Normalize (e.g., "IFNg" → "IFN-gamma") | Prefix "parse_" |
  | OpenProblems | Direct copy | PubChem/ChEMBL resolution | Prefix "op_" |
  | Tabula Sapiens | Already has CL IDs | Set to None (unperturbed) | Prefix "ts_" |

- [ ] **Build index population script**
  - Generate text descriptions for perturbations and cell types
  - Compute text embeddings via OpenAI API (text-embedding-3-small, 1536 dim)
  - Batch insert to Cloud SQL (~10K rows/batch)
  - Build HNSW indexes after data load completes
  - Estimated time: 2-4 hours

### 1.3 Cell Ontology Integration

> **Spec Reference**: `./specification/ontology-resolution.md`

- [ ] **Implement CellOntologyService**
  - Load CL OBO file into ontology_terms/relationships tables
  - Generate embeddings for all term names + definitions
  - Build HNSW index for semantic search
  - Implement graph traversal methods (get_parents, get_children, get_ancestors)

- [ ] **Create resolution tools**
  - `resolve_cell_type_semantic`: Semantic search with distance threshold
  - `get_cell_type_neighbors`: Get related types via hierarchy
  - `query_cell_ontology_ols`: OLS API fallback for edge cases

---

## PHASE 2: Core Backend Services

### 2.1 Shared Data Models

> **Spec Reference**: `./specification/data-models.md`

- [ ] **Implement Pydantic schemas** (shared/models/)
  ```python
  class StructuredQuery(BaseModel):
      """Parsed representation of a user query."""
      raw_query: str = Field(description="Original user query text")
      task_type: ICLTaskType = Field(description="Resolved ICL task type")
      
      # Cell type resolution
      cell_type_query: str = Field(description="Extracted cell type from query")
      cell_type_cl_id: Optional[str] = Field(description="Resolved Cell Ontology ID")
      cell_type_synonyms: list[str] = Field(default_factory=list)
      
      # Perturbation resolution (optional for observational tasks)
      perturbation_query: Optional[str] = None
      perturbation_type: Optional[PerturbationType] = None
      perturbation_resolved: Optional[str] = None  # Canonical name
      perturbation_external_ids: dict[str, str] = Field(default_factory=dict)
      expected_targets: list[str] = Field(default_factory=list)
      expected_pathways: list[str] = Field(default_factory=list)
      
      # Observational context (optional for perturbational tasks)
      target_donor_id: Optional[str] = None
      target_tissue: Optional[str] = None  # UBERON ID
      target_disease_state: Optional[str] = None  # MONDO ID
      target_condition: Optional[str] = None
      reference_donor_id: Optional[str] = None
      reference_dataset: Optional[str] = None
      
      # Biological context (task-agnostic)
      expected_marker_genes: list[str] = Field(default_factory=list)
      expected_tissue_genes: list[str] = Field(default_factory=list)
      literature_context: Optional[str] = None
      
      # Control strategy (perturbational tasks)
      control_strategy: ControlStrategy = ControlStrategy.SYNTHETIC_CONTROL
      control_cells_available: bool = False
      control_cell_info: Optional[dict] = None
      control_strategy_fallback: Optional[ControlStrategy] = None
  
  class CellSetCandidate(BaseModel):
      """A candidate cell set for prompt selection."""
      dataset: str
      perturbation_name: Optional[str] = None
      cell_type_cl_id: str
      cell_type_name: str
      donor_id: Optional[str] = None
      tissue_uberon_id: Optional[str] = None
      disease_mondo_id: Optional[str] = None
      sample_condition: Optional[str] = None
      sample_metadata: Optional[dict] = None
      cell_indices: list[int]
      n_cells: int
      mean_n_genes: Optional[float] = None
      mean_total_counts: Optional[float] = None
      strategy: Literal["direct", "mechanistic", "semantic", "ontology", "donor_context", "tissue_atlas"]
      relevance_score: float
      rationale: str
      
      # For ontology-guided strategy
      ontology_distance: Optional[int] = None
      
      def selection_key(self) -> tuple:
          """Return unique key for deduplication."""
          return (self.dataset, self.perturbation_name, self.cell_type_cl_id, 
                  self.donor_id, self.tissue_uberon_id, self.sample_condition)
  
  class PromptCandidate(BaseModel):
      """A candidate prompt configuration."""
      cell_set: CellSetCandidate
      strategy: str
      prompt_cell_indices: list[int]
      
      # Paired control for synthetic control strategy
      paired_control_indices: Optional[list[int]] = None
      paired_control_metadata: Optional[dict] = None
      
      # Scoring metadata
      similarity_score: Optional[float] = None
      mechanistic_score: Optional[float] = None
      ontology_distance: Optional[int] = None
      rationale: str
  
  class GroundingScore(BaseModel):
      """Biological grounding evaluation for perturbational tasks."""
      # Component scores (1-10)
      pathway_coherence: int = Field(ge=1, le=10)
      target_activation: int = Field(ge=1, le=10)
      literature_support: int = Field(ge=1, le=10)
      network_coherence: int = Field(ge=1, le=10)
      
      # Composite score
      composite_score: int = Field(ge=1, le=10)
      
      # Details
      enriched_pathways: list[dict]
      de_genes_up: list[str]
      de_genes_down: list[str]
      literature_evidence: list[dict]
      
      # Feedback for next iteration
      improvement_suggestions: list[str]
  
  class ObservationalGroundingScore(BaseModel):
      """Biological grounding evaluation for observational ICL tasks."""
      # Component scores (1-10)
      marker_gene_expression: int = Field(ge=1, le=10)
      tissue_signature_match: int = Field(ge=1, le=10)
      donor_effect_capture: int = Field(ge=1, le=10)
      cell_type_coherence: int = Field(ge=1, le=10)
      
      # Composite score
      composite_score: int = Field(ge=1, le=10)
      
      # Details
      marker_genes_detected: dict[str, float]
      tissue_genes_detected: dict[str, float]
      donor_signature_genes: list[str]
      
      # Feedback
      improvement_suggestions: list[str]

  class IterationRecord(BaseModel):
      """Record of a single iteration."""
      iteration_number: int
      prompt_candidates: list[PromptCandidate]
      selected_prompt: PromptCandidate
      grounding_score: Union[GroundingScore, ObservationalGroundingScore]
      duration_seconds: float
      
      # Control strategy used
      control_strategy: ControlStrategy = ControlStrategy.SYNTHETIC_CONTROL
      
      # Artifacts
      prediction_gcs_path: Optional[str] = None
      control_prediction_gcs_path: Optional[str] = None  # Synthetic control only
      query_cells_gcs_path: Optional[str] = None  # Query-as-control only
      de_analysis_metadata: Optional[dict] = None

  class HaystackRun(BaseModel):
      """Complete run record."""
      run_id: str
      user_email: Optional[str] = None
      start_time: datetime
      end_time: Optional[datetime] = None
      status: Literal["pending", "running", "completed", "failed", "cancelled"]
      
      # Control strategy tracking
      control_strategy: ControlStrategy = ControlStrategy.SYNTHETIC_CONTROL
      control_strategy_effective: Optional[ControlStrategy] = None
      control_cells_available: bool = False
      
      # Configuration
      config: dict
      random_seed: int
      
      # Query
      raw_query: str
      structured_query: Optional[StructuredQuery] = None
      
      # Iterations
      iterations: list[IterationRecord] = Field(default_factory=list)
      
      # Final result
      final_score: Optional[int] = None
      termination_reason: Optional[str] = None
      
      # Output paths (GCS)
      output_anndata_path: Optional[str] = None
      output_report_path: Optional[str] = None
      output_log_path: Optional[str] = None
  ```

- [ ] **Implement enums**
  ```python
  class PerturbationType(str, Enum):
      """Types of perturbations supported."""
      DRUG = "drug"
      CYTOKINE = "cytokine"
      GENETIC = "genetic"
      UNKNOWN = "unknown"

  class ICLTaskType(str, Enum):
      """Types of ICL tasks supported by STACK."""
      PERTURBATION_NOVEL_CELL_TYPES = "perturbation_novel_cell_types"
      PERTURBATION_NOVEL_SAMPLES = "perturbation_novel_samples"
      CELL_TYPE_IMPUTATION = "cell_type_imputation"
      DONOR_EXPRESSION_PREDICTION = "donor_expression_prediction"
      CROSS_DATASET_GENERATION = "cross_dataset_generation"
  
  class ControlStrategy(str, Enum):
      """Strategy for computing differential expression in perturbational tasks."""
      QUERY_AS_CONTROL = "query_as_control"
      SYNTHETIC_CONTROL = "synthetic_control"

  class RunPhase(str, Enum):
      """Current phase of a run for status reporting."""
      PENDING = "pending"
      QUERY_ANALYSIS = "query_analysis"
      PROMPT_GENERATION = "prompt_generation"
      INFERENCE = "inference"
      EVALUATION = "evaluation"
      OUTPUT_GENERATION = "output_generation"
  ```

### 2.2 Database Client

> **Spec Reference**: `./specification/backend-api.md` (Section 9.1: Project Structure)

- [ ] **Implement HaystackDatabase** (orchestrator/services/database.py)
  ```python
  class HaystackDatabase:
      async def connect(self)
      async def close(self)
      async def execute_query(self, sql: str, params: tuple) -> list[dict]
      async def get_run(self, run_id: str) -> dict
      async def update_run_status(self, run_id: str, status: str, **kwargs)
      async def create_run(self, run: CreateRunRequest) -> str
      async def list_perturbations(self, dataset: str = None) -> list[dict]
      async def list_cell_types(self, dataset: str = None) -> list[dict]
  ```
  - Use asyncpg with connection pooling
  - Cloud SQL Python Connector for secure connections

### 2.3 GCS Service

- [ ] **Implement GCS client** (orchestrator/services/gcs.py)
  ```python
  class GCSService:
      async def upload_anndata(self, adata: AnnData, path: str)
      async def download_anndata(self, path: str) -> AnnData
      async def generate_signed_url(self, path: str, expiration: int = 3600) -> str
      async def write_json(self, data: dict, path: str)
      async def read_json(self, path: str) -> dict
  ```

### 2.4 External API Clients

> **Spec References**: 
> - `./specification/tools.md` (Section 6.3: Drug-Target Knowledge Tools)
> - `./specification/literature-search.md`

- [ ] **Implement biological knowledge APIs** (orchestrator/services/biological_apis.py)
  - KEGG: Pathways, drug targets
  - Reactome: Pathway analysis
  - UniProt: Protein information
  - gseapy: GO/KEGG/Reactome enrichment analysis

- [ ] **Implement literature search** (orchestrator/services/literature.py)
  - **Search tools**:
    - `search_literature`: Multi-database search (pubmed, semantic_scholar, biorxiv)
    - Returns formatted list with titles, authors, abstracts
    - max_results parameter per database
  - **Acquisition tools**:
    - `acquire_full_text_paper`: Get paper content by DOI
    - Tries: preprint servers → CORE API → Europe PMC → Unpaywall
    - Uses docling for PDF-to-markdown conversion
    - Falls back to abstract if full text unavailable
  - **Search databases**:
    - PubMed (E-utilities API)
    - Semantic Scholar (Graph API)
    - bioRxiv/medRxiv (preprint servers)
  - **PDF acquisition pipeline**: CORE → Europe PMC → Unpaywall
  ```python
  @tool
  async def search_literature(
      query: str,
      max_results: int = 10,
      databases: list[str] | None = None,  # pubmed, semantic_scholar, biorxiv
  ) -> str:
      """Search scientific literature databases for relevant papers."""
      ...
  
  @tool
  async def acquire_full_text_paper(doi: str) -> str:
      """Acquire full-text paper content and convert to markdown."""
      ...
  ```

---

## PHASE 3: Prompt Retrieval Strategies

> **Spec Reference**: `./specification/prompt-retrieval.md`

### 3.1 Strategy Framework

- [ ] **Implement base RetrievalStrategy**
  ```python
  class RetrievalStrategy(ABC):
      def __init__(self, db: HaystackDatabase):
          self.db = db
      
      @abstractmethod
      async def retrieve(
          self,
          query: StructuredQuery,
          max_results: int = 50,
          filters: Optional[dict] = None,
      ) -> list[CellSetCandidate]
      
      @property
      @abstractmethod
      def strategy_name(self) -> str:
          """Return the strategy name for logging."""
          pass
  ```

- [ ] **Implement strategy orchestration with task-specific priority**
  ```python
  from .direct_match import DirectMatchStrategy
  from .mechanistic_match import MechanisticMatchStrategy
  from .semantic_match import SemanticMatchStrategy
  from .ontology_guided import OntologyGuidedStrategy
  from .donor_context import DonorContextStrategy
  from .tissue_atlas import TissueAtlasStrategy
  
  # Perturbational tasks: predict drug/cytokine effects
  PERTURBATIONAL_STRATEGY_PRIORITY = [
      DirectMatchStrategy,      # Exact perturbation + cell type
      MechanisticMatchStrategy, # Same target genes/pathways
      SemanticMatchStrategy,    # Similar perturbation description
      OntologyGuidedStrategy,   # Related cell types via CL
  ]
  
  # Observational tasks: impute/predict cell types for donors
  OBSERVATIONAL_STRATEGY_PRIORITY = [
      DonorContextStrategy,     # Same donor, different cell types
      TissueAtlasStrategy,      # Same tissue, different donors
      OntologyGuidedStrategy,   # Related cell types via CL
      SemanticMatchStrategy,    # Similar sample context
  ]
  
  async def execute_strategy_pipeline(
      query: StructuredQuery,
      db: HaystackDatabase,
      max_results: int = 50,
  ) -> list[CellSetCandidate]:
      """Execute retrieval strategies in priority order based on task type."""
      is_perturbational = query.task_type in [
          ICLTaskType.PERTURBATION_NOVEL_CELL_TYPES,
          ICLTaskType.PERTURBATION_NOVEL_SAMPLES,
      ]
      
      strategies = (PERTURBATIONAL_STRATEGY_PRIORITY 
                   if is_perturbational 
                   else OBSERVATIONAL_STRATEGY_PRIORITY)
      
      candidates = []
      seen_keys = set()
      
      for strategy_class in strategies:
          if len(candidates) >= max_results:
              break
          
          strategy = strategy_class(db)
          results = await strategy.retrieve(
              query, 
              max_results=max_results - len(candidates),
              filters={"exclude_keys": seen_keys}
          )
          
          for c in results:
              key = c.selection_key()
              if key not in seen_keys:
                  seen_keys.add(key)
                  candidates.append(c)
      
      return candidates[:max_results]
  ```

### 3.2 Direct Match Strategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 5: Direct Match Strategy)

- [ ] **Implement DirectMatchStrategy**
  - Exact perturbation + cell type matching
  - Fuzzy matching via synonym lookup table
  - Priority: Highest for perturbational tasks

### 3.3 Mechanistic Match Strategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 6: Mechanistic Match Strategy)

- [ ] **Implement MechanisticMatchStrategy**
  - Input: Target genes and pathways from query
  - Find perturbations sharing targets/pathways
  - Score by overlap (Jaccard similarity)

### 3.4 Semantic Match Strategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 7: Semantic Match Strategy)

- [ ] **Implement SemanticMatchStrategy**
  - Vector similarity search on perturbation/cell_type embeddings
  - Configurable similarity threshold (default: 0.7)
  - Uses HNSW indexes for efficient search
  - Generate query text from perturbation name + type + targets
  - Returns candidates sorted by cosine distance

### 3.5 Ontology-Guided Strategy

> **Spec References**: 
> - `./specification/prompt-retrieval.md` (Section 8: Ontology-Guided Strategy)
> - `./specification/ontology-resolution.md`

- [ ] **Implement OntologyGuidedStrategy**
  - **Key design**: Uses native `CellOntologyService` (no external MCP server)
  - Leverages `ontology_terms` and `ontology_relationships` tables in Cloud SQL
  - CL hierarchy traversal for related cell types when exact match unavailable
  - Supports bidirectional graph traversal (`is_a`, `part_of`, `develops_from`)
  - Score by ontology distance: relevance_score = 1.0 / (distance + 1)
  - Priority: Parents > children > siblings
  ```python
  class OntologyGuidedStrategy(RetrievalStrategy):
      """Find cells via Cell Ontology hierarchy using native CL tools."""
      
      def __init__(self, db: HaystackDatabase):
          super().__init__(db)
          self.ontology_service = CellOntologyService.get_instance()
      
      @property
      def strategy_name(self) -> str:
          return "ontology_guided"
      
      async def retrieve(
          self,
          query: StructuredQuery,
          max_results: int = 50,
          filters: Optional[dict] = None,
      ) -> list[CellSetCandidate]:
          """
          Find cell sets with related cell types via Cell Ontology.
          
          Strategy:
          1. Get neighbors of the query cell type via CL hierarchy
          2. Group neighbors by relationship type and distance
          3. Search for cells of related types with query perturbation/context
          4. Score inversely with ontology distance
          """
          if not query.cell_type_cl_id:
              return []
          
          # Get neighbors from ontology service
          neighbors = await self.ontology_service.get_neighbors(
              term_ids=[query.cell_type_cl_id],
              relationship_types=["is_a", "part_of", "develops_from"],
              max_distance=2,
          )
          
          # Prioritize: is_a parents first, then children, then others
          # Search for cells of each related type
          ...
  ```

### 3.6 Donor Context Strategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 9: Donor Context Strategy)

- [ ] **Implement DonorContextStrategy**
  - For observational tasks (donor imputation, expression prediction)
  - Find same donor, different cell types for context
  - Priority in OBSERVATIONAL_STRATEGY_PRIORITY
  ```python
  class DonorContextStrategy(RetrievalStrategy):
      """Find cells from same donor with different cell types."""
      
      @property
      def strategy_name(self) -> str:
          return "donor_context"
      
      async def retrieve(
          self,
          query: StructuredQuery,
          max_results: int = 50,
          filters: Optional[dict] = None,
      ) -> list[CellSetCandidate]:
          # Find other cell types from the target donor
          sql = """
              SELECT 
                  c.dataset, c.cell_type_name, c.cell_type_cl_id,
                  c.donor_id, c.tissue_name, c.tissue_uberon_id,
                  c.sample_condition, c.sample_metadata,
                  ARRAY_AGG(c.cell_index) AS cell_indices,
                  COUNT(*) AS n_cells,
                  AVG(c.n_genes) AS mean_n_genes
              FROM cells c
              WHERE c.donor_id = $1
                AND c.cell_type_cl_id != $2
                AND c.is_control = TRUE
              GROUP BY c.dataset, c.cell_type_name, c.cell_type_cl_id,
                       c.donor_id, c.tissue_name, c.tissue_uberon_id,
                       c.sample_condition, c.sample_metadata
              ORDER BY n_cells DESC
              LIMIT $3
          """
          ...
  ```

### 3.7 Tissue Atlas Strategy

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 10: Tissue Atlas Strategy)

- [ ] **Implement TissueAtlasStrategy**
  - For observational tasks (cross-tissue cell type retrieval)
  - Prioritizes comprehensive atlases (Tabula Sapiens)
  - Selects high-quality reference populations with large cell counts
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
      ) -> list[CellSetCandidate]:
          sql = """
              SELECT 
                  c.dataset, c.cell_type_name, c.cell_type_cl_id,
                  c.donor_id, c.tissue_name, c.tissue_uberon_id,
                  c.sample_condition, c.sample_metadata,
                  ARRAY_AGG(c.cell_index) AS cell_indices,
                  COUNT(*) AS n_cells,
                  AVG(c.n_genes) AS mean_n_genes
              FROM cells c
              WHERE c.cell_type_cl_id = $1
                AND c.is_control = TRUE
                AND ($2::text IS NULL OR c.tissue_uberon_id = $2)
              GROUP BY c.dataset, c.cell_type_name, c.cell_type_cl_id,
                       c.donor_id, c.tissue_name, c.tissue_uberon_id,
                       c.sample_condition, c.sample_metadata
              ORDER BY 
                  CASE c.dataset 
                      WHEN 'tabula_sapiens' THEN 1 
                      ELSE 2 
                  END,
                  n_cells DESC
              LIMIT $3
          """
          ...
  ```

### 3.8 Candidate Ranking & Selection

> **Spec Reference**: `./specification/prompt-retrieval.md` (Section 11: Candidate Ranking and Selection)

- [ ] **Implement CandidateRanker**
  ```python
  class CandidateRanker:
      """Rank and select final prompt candidates."""
      
      def __init__(
          self,
          relevance_weight: float = 0.4,
          diversity_weight: float = 0.3,
          quality_weight: float = 0.3,
      ):
          self.relevance_weight = relevance_weight
          self.diversity_weight = diversity_weight
          self.quality_weight = quality_weight
      
      def rank_candidates(
          self,
          candidates: list[CellSetCandidate],
          top_k: int = 10,
      ) -> list[CellSetCandidate]:
          """
          Rank candidates and select top K.
          
          Scoring factors:
          1. Relevance: Strategy-specific relevance score
          2. Diversity: Penalize redundant selections
          3. Quality: Cell count, data quality metrics
          """
          selected = []
          remaining = list(candidates)
          
          while remaining and len(selected) < top_k:
              best_score = -1
              best_idx = -1
              
              for i, c in enumerate(remaining):
                  score = self._compute_final_score(c, selected)
                  if score > best_score:
                      best_score = score
                      best_idx = i
              
              if best_idx >= 0:
                  selected.append(remaining.pop(best_idx))
          
          return selected
      
      def _compute_final_score(self, candidate: CellSetCandidate, selected: list) -> float:
          relevance = candidate.relevance_score
          quality = self._compute_quality_score(candidate)
          diversity = self._compute_diversity_score(candidate, selected)
          
          return (self.relevance_weight * relevance + 
                  self.quality_weight * quality + 
                  self.diversity_weight * diversity)
      
      def _compute_diversity_score(self, candidate: CellSetCandidate, selected: list) -> float:
          if not selected:
              return 1.0
          
          same_pert = sum(1 for s in selected if s.perturbation_name == candidate.perturbation_name)
          same_ct = sum(1 for s in selected if s.cell_type_cl_id == candidate.cell_type_cl_id)
          same_dataset = sum(1 for s in selected if s.dataset == candidate.dataset)
          
          diversity = 1.0
          diversity -= 0.3 * (same_pert / len(selected))
          diversity -= 0.2 * (same_ct / len(selected))
          diversity -= 0.1 * (same_dataset / len(selected))
          
          return max(0.0, diversity)
  ```

### 3.9 Control Strategy Implementation

> **Spec Reference**: `./specification/data-models.md` (ControlStrategy enum and related models)

- [ ] **Implement synthetic control matching**
  - Find matched control cells (same donor/sample, unperturbed)
  - Validate availability before run (POST /api/v1/runs/validate-control-strategy)
  - Set `paired_control_indices` and `paired_control_metadata` in PromptCandidate
  - Run STACK twice: once with perturbed prompt, once with control prompt
  - Store `control_prediction_gcs_path` in IterationRecord
  - Higher confidence in grounding evaluation

- [ ] **Implement query-as-control baseline**
  - Use original query cells as control reference for DE analysis
  - Single STACK inference run
  - Store `query_cells_gcs_path` in IterationRecord
  - Faster but may include prompting artifacts
  - Mild confidence penalty in grounding

- [ ] **Control strategy availability check**
  ```python
  @router.post("/validate-control-strategy", response_model=ControlStrategyValidation)
  async def validate_control_strategy(request: ControlStrategyValidationRequest):
      """
      Validate whether synthetic control strategy is possible for a query.
      
      Checks if matched control cells are available in the atlas for the
      expected prompt cells.
      """
      # Resolve query to structured form
      # Check for matched unperturbed cells from same donor/sample
      # Return ControlStrategyValidation with:
      #   - synthetic_control_available: bool
      #   - control_cells_found: int
      #   - control_donors: list[str]
      #   - recommendation: ControlStrategy
      #   - warning: Optional[str]
  ```

- [ ] **Fallback logic in orchestrator**
  ```python
  # In OrchestratorAgent.run()
  if self.control_strategy == "synthetic_control":
      control_info = await self.prompt_agent.find_matched_controls(structured_query)
      structured_query.control_cells_available = bool(control_info)
      structured_query.control_cell_info = control_info
      if not control_info:
          structured_query.control_strategy_fallback = "query_as_control"
          self.control_strategy = "query_as_control"  # Actual fallback
  ```

---

## PHASE 4: Agent Framework

> **Spec References**: 
> - `./specification/agents.md`
> - `./specification/tools.md`

### 4.1 Orchestrator Agent

- [ ] **Implement OrchestratorAgent** (orchestrator/agents/orchestrator.py)
  ```python
  class OrchestratorAgent:
      def __init__(self, run_id: str, query: str, user_email: str, config: dict)
      
      async def run(self) -> OrchestratorResult:
          # Phase 1: Query understanding
          structured_query = await self.query_understanding_agent.run(self.query)
          
          # Phase 2: Iteration loop (max 5 iterations)
          for iteration in range(self.config["max_iterations"]):
              # Check cancellation
              if await self.check_cancellation():
                  break
              
              # Generate prompts
              prompt_config = await self.prompt_generation_agent.run(structured_query)
              
              # Run STACK inference
              predictions = await self.run_inference(prompt_config)
              
              # Evaluate grounding
              score = await self.grounding_evaluation_agent.run(predictions)
              
              # Check convergence
              if score.composite_score >= self.config["score_threshold"]:
                  break
          
          # Phase 3: Output generation
          return await self.generate_outputs()
  ```

### 4.2 Query Understanding Subagent

> **Spec Reference**: `./specification/agents.md` (Query Understanding Agent section)

- [ ] **Implement QueryUnderstandingAgent**
  - Parse natural language to StructuredQuery
  - Resolve cell types via Cell Ontology
  - Resolve perturbations via DrugBank/PubChem
  - Retrieve biological priors from literature
  - Classify task type (perturbational vs observational)

- [ ] **Tools**:
  - `parse_query_to_structured`
  - `resolve_cell_type_semantic`
  - `resolve_perturbation`
  - `get_drug_targets`
  - `search_literature`

### 4.3 Prompt Generation Subagent

> **Spec Reference**: `./specification/agents.md` (Prompt Generation Agent section)

- [ ] **Implement PromptGenerationAgent**
  - Execute retrieval strategies in parallel
  - Rank and select candidates
  - Match control prompts (for synthetic control)
  - Validate prompt quality

- [ ] **Tools**:
  - `execute_retrieval_strategies`
  - `rank_and_select_prompts`
  - `find_matched_controls`
  - `validate_prompt_quality`

### 4.4 Grounding Evaluation Subagent

> **Spec References**: 
> - `./specification/agents.md` (Grounding Evaluation Agent section)
> - `./specification/tools.md` (Section 6.5: Enrichment and Evaluation Tools)

- [ ] **Implement GroundingEvaluationAgent**
  - Unified evaluator that handles both perturbational and observational tasks
  - Extract DE genes from predictions vs control/reference
  - Run pathway enrichment (GO/KEGG/Reactome) for perturbational tasks
  - Check marker gene expression for observational tasks
  - Compute composite score (1-10)
  - Generate improvement suggestions

- [ ] **Implement GroundingEvaluator class**
  ```python
  class GroundingEvaluator:
      """Unified evaluator that handles both perturbational and observational tasks."""
      
      async def evaluate(
          self,
          query: StructuredQuery,
          predictions: AnnData,
          control_or_reference: AnnData,
          control_strategy: ControlStrategy | None = None,
      ) -> Union[GroundingScore, ObservationalGroundingScore]:
          """Evaluate predictions based on task type."""
          if query.task_type in [
              ICLTaskType.PERTURBATION_NOVEL_CELL_TYPES,
              ICLTaskType.PERTURBATION_NOVEL_SAMPLES,
          ]:
              return await self._evaluate_perturbational(
                  query, predictions, control_or_reference, control_strategy
              )
          return await self._evaluate_observational(
              query, predictions, control_or_reference
          )
  ```

- [ ] **Perturbational scoring criteria (each 1-10)**:
  - **Pathway coherence**: Do enriched pathways match expected biology?
  - **Target activation**: Are known targets differentially expressed?
  - **Literature support**: Do predictions have published evidence?
  - **Network coherence**: Do DE genes form functional modules?

- [ ] **Observational scoring criteria (each 1-10)**:
  - **Marker gene expression**: Are canonical cell type markers expressed?
  - **Tissue signature match**: Does expression match tissue-specific patterns?
  - **Donor effect capture**: Are donor-specific effects preserved?
  - **Cell type coherence**: Is expression consistent with cell identity?

- [ ] **Control strategy confidence adjustment**
  - Synthetic control: Higher confidence (paired comparison)
  - Query-as-control: Mild confidence penalty (potential prompting artifacts)

- [ ] **Tools**:
  - `extract_de_genes`: DE analysis with control_strategy parameter
  - `run_pathway_enrichment`: GO/KEGG/Reactome for perturbational
  - `get_cell_type_markers`: Marker genes for observational
  - `get_tissue_signature`: Tissue-specific genes for observational
  - `identify_donor_signature`: Donor effects for observational
  - `search_literature_evidence`: Literature validation
  - `compute_grounding_score`: Unified scoring tool

  ```python
  @tool
  async def compute_grounding_score(
      query: StructuredQuery,
      predictions: AnnData,
      control_or_reference: AnnData,
      control_strategy: ControlStrategy | None = None,
  ) -> Union[GroundingScore, ObservationalGroundingScore]:
      """Compute composite biological grounding score."""
      ...
  ```

### 4.5 LangChain Tool Definitions

> **Spec Reference**: `./specification/tools.md`

- [ ] **Implement all tools** (orchestrator/tools/)
  - Proper Pydantic schemas for inputs/outputs
  - Async execution
  - Error handling with retries
  - Comprehensive docstrings for LLM

---

## PHASE 5: STACK Inference Integration

> **Spec References**: 
> - `./specification/tools.md` (Section 6.4: STACK Inference Tools)
> - `./specification/architecture.md` (Section 3.4: Two-Tier Batch Job Architecture)

### 5.1 GPU Batch Job

- [ ] **Create STACK inference container** (docker/Dockerfile.inference)
  ```dockerfile
  FROM nvcr.io/nvidia/pytorch:24.01-py3
  RUN git clone https://github.com/arcinstitute/STACK.git /app/stack
  RUN pip install -e /app/stack
  RUN pip install google-cloud-storage scanpy anndata
  COPY inference /app/inference
  ENTRYPOINT ["python", "-m", "inference.run_inference"]
  ```

- [ ] **Implement run_inference.py**
  ```python
  async def main():
      # Load STACK model from GCS
      model = load_stack_model(os.environ["MODEL_PATH"])
      
      # Read prompt and query cells from GCS
      prompt_cells = read_anndata(os.environ["PROMPT_CELLS_PATH"])
      query_cells = read_anndata(os.environ["QUERY_CELLS_PATH"])
      
      # Execute mask-diffusion inference
      predictions = model.predict(
          prompt_cells=prompt_cells,
          query_cells=query_cells,
          num_samples=5,  # T=5 as in paper
      )
      
      # Write predictions to GCS
      write_anndata(predictions, os.environ["OUTPUT_PATH"])
  ```

### 5.2 GPU Batch Client

- [ ] **Implement GPUBatchClient** (orchestrator/services/batch.py)
  ```python
  class GPUBatchClient:
      async def submit_inference_job(
          self,
          run_id: str,
          iteration: int,
          prompt_cells_path: str,
          query_cells_path: str,
          control_strategy: ControlStrategy,
      ) -> str:  # Returns job ID
      
      async def wait_for_completion(self, job_id: str, timeout: int = 1800) -> JobResult
      
      async def cancel_job(self, job_id: str)
  ```
  - A100 80GB GPU
  - Handle synthetic control (2x inference)
  - Retry logic for transient failures

### 5.3 Inference Tool

- [ ] **Implement run_stack_inference_tool**
  - Prepare input data (write cells to GCS)
  - Submit GPU Batch job
  - Poll for completion with timeout
  - Read predictions from GCS
  - Update run status

---

## PHASE 6: Cloud Run API

> **Spec Reference**: `./specification/backend-api.md`

### 6.1 FastAPI Application

- [ ] **Implement FastAPI app** (api/main.py)
  ```python
  from fastapi import FastAPI
  from fastapi.staticfiles import StaticFiles
  
  app = FastAPI(title="HAYSTACK API", version="1.0.0")
  
  # Mount static frontend
  app.mount("/", StaticFiles(directory="/app/frontend/out", html=True))
  
  # Include routers
  app.include_router(runs_router, prefix="/api/v1/runs")
  app.include_router(metadata_router, prefix="/api/v1/metadata")
  app.include_router(health_router, prefix="/api/v1/health")
  ```

### 6.2 Run Management Endpoints

> **Spec Reference**: `./specification/backend-api.md` (Section 9.3: Run Management Endpoints)

- [ ] **POST /api/v1/runs/**
  - Validate CreateRunRequest
  - Create run in Cloud SQL (status: "pending")
  - Submit CPU Batch job
  - Return run_id immediately (async job pattern)
  ```python
  class CreateRunRequest(BaseModel):
      query: str = Field(description="Natural language query", min_length=10)
      control_strategy: ControlStrategy = ControlStrategy.SYNTHETIC_CONTROL
      max_iterations: Optional[int] = Field(default=None, ge=1, le=10)
      score_threshold: Optional[int] = Field(default=None, ge=1, le=10)
      llm_provider: Optional[Literal["anthropic", "openai", "google_genai"]] = None
      llm_model: Optional[str] = None
      enable_literature_search: Optional[bool] = True
      enable_pathway_enrichment: Optional[bool] = True
      random_seed: Optional[int] = None
  ```

- [ ] **GET /api/v1/runs/{id}**
  - Read run status from Cloud SQL
  - Return RunStatusResponse with full status info
  ```python
  class RunStatusResponse(BaseModel):
      run_id: str
      status: Literal["pending", "running", "completed", "failed", "cancelled"]
      current_iteration: int = 0
      max_iterations: int
      current_phase: Optional[RunPhase] = None
      grounding_scores: list[int] = Field(default_factory=list)  # Score per iteration
      created_at: datetime
      updated_at: datetime
      error_message: Optional[str] = None
      
      # Control strategy info
      control_strategy: Optional[ControlStrategy] = None
      control_strategy_effective: Optional[ControlStrategy] = None
      control_cells_available: bool = False
      
      # User info
      user_email: str
  ```

- [ ] **GET /api/v1/runs/{id}/result**
  - Generate signed GCS URLs (1 hour expiration)
  - Return RunResultResponse (AnnData, report, logs)
  ```python
  class RunResultResponse(BaseModel):
      run_id: str
      success: bool
      grounding_score: int
      termination_reason: str
      
      # Predictions summary
      num_de_genes: int
      top_upregulated: list[str]
      top_downregulated: list[str]
      
      # Interpretation
      enriched_pathways: list[str]
      activated_tfs: list[str]
      
      # Download URLs (signed GCS URLs)
      anndata_url: str
      report_url: str
      log_url: str
  ```

- [ ] **GET /api/v1/runs/**
  - List user's runs with pagination
  - Filter by status
  ```python
  class RunListResponse(BaseModel):
      runs: list[HaystackRun]
      total: int
      page: int
      page_size: int
  ```

- [ ] **POST /api/v1/runs/{id}/cancel**
  - Set cancellation flag in Cloud SQL
  - Cancel CPU Batch job via GCP API
  - GPU job (if running) is also cancelled

### 6.3 Metadata Endpoints

- [ ] **GET /api/v1/metadata/perturbations**
- [ ] **GET /api/v1/metadata/cell-types**
- [ ] **GET /api/v1/metadata/datasets**

### 6.4 Control Strategy Validation

- [ ] **POST /api/v1/runs/validate-control-strategy**
  - Check synthetic control availability for query
  - Return ControlStrategyValidation with recommendation

### 6.5 IAP Integration

- [ ] **Implement get_current_user dependency**
  - Extract user email from `X-Goog-Authenticated-User-Email` header
  - Development fallback for local testing

### 6.6 CPU Batch Client

- [ ] **Implement BatchClient** (api/services/batch.py)
  - Submit orchestrator jobs (e2-standard-4, 4 vCPU, 16 GB)
  - Configure VPC connector for Cloud SQL access
  - Set environment variables (RUN_ID, USER_EMAIL, etc.)

---

## PHASE 7: Frontend

> **Spec Reference**: `./specification/frontend.md`

### 7.1 Project Setup

- [ ] **Initialize Next.js 14+** with App Router
  - TypeScript configuration
  - Tailwind CSS + Headless UI
  - TanStack Query for data fetching
  - Zustand for state management

- [ ] **Configure static export**
  ```javascript
  // next.config.js
  module.exports = {
    output: 'export',
    trailingSlash: true,
  }
  ```

### 7.2 Core Components

- [ ] **Layout components**
  - Sidebar navigation
  - Header with user info
  - PageLayout wrapper

- [ ] **RunForm component**
  - Query textarea with validation (min 10 chars)
  - Control strategy selection (radio group)
  - Pros/cons display for each strategy
  - Control strategy validation with warning
  - Submit button with loading state

- [ ] **RunStatus component**
  - Status header with icon
  - Progress bar with iteration count
  - Current phase display
  - Grounding scores history
  - Control strategy display with fallback indicator
  - Cancel button
  - Error message display
  - Email notification note

- [ ] **RunResults component**
  - Download links (AnnData, report, logs)
  - Grounding score visualization
  - Iteration history table
  - Pathway enrichment results
  - Literature citations

### 7.3 Hooks & State Management

- [ ] **useRunPolling hook**
  ```typescript
  export function useRunPolling(runId: string, options = {}) {
    const { pollInterval = 15000 } = options;
    
    return useQuery({
      queryKey: ["run", runId],
      queryFn: () => getRunStatus(runId),
      refetchInterval: (query) => {
        const status = query.state.data?.status;
        if (["completed", "failed", "cancelled"].includes(status)) {
          return false;
        }
        return pollInterval;
      },
      refetchIntervalInBackground: true,
    });
  }
  ```

- [ ] **Zustand stores**
  - `runStore`: Current run state
  - `uiStore`: UI preferences

- [ ] **API client** (lib/api/)
  - Axios instance with base URL
  - createRun, getRunStatus, cancelRun, getRunResult
  - validateControlStrategy
  - Error handling

### 7.4 Pages

- [ ] **app/page.tsx**: Recent runs list, quick start button
- [ ] **app/runs/new/page.tsx**: RunForm, redirect on submission
- [ ] **app/runs/[id]/page.tsx**: RunStatus with polling, RunResults when completed

### 7.5 Styling

- [ ] **Tailwind theme configuration**
- [ ] **UI components** (components/ui/): Button, Modal, LoadingSpinner, ProgressBar, Tooltip

---

## PHASE 8: Orchestrator Batch Job

> **Spec References**: 
> - `./specification/orchestrator.md`
> - `./specification/architecture.md` (Section 3: System Architecture)

### 8.1 Entrypoint

- [ ] **Implement orchestrator/main.py**
  ```python
  async def main():
      run_id = os.environ["RUN_ID"]
      user_email = os.environ["USER_EMAIL"]
      control_strategy = os.environ.get("CONTROL_STRATEGY", "synthetic_control")
      
      db = await HaystackDatabase.connect()
      email_service = SendGridService()
      
      try:
          await db.update_run_status(run_id, "running")
          
          agent = OrchestratorAgent(run_id, query, user_email, config)
          result = await agent.run()
          
          await db.update_run_status(run_id, "completed", final_score=result.score)
          await email_service.send_run_completed(user_email, run_id, result)
          
      except Exception as e:
          await db.update_run_status(run_id, "failed", error_message=str(e))
          await email_service.send_run_failed(user_email, run_id, str(e))
          raise
  ```

### 8.2 Email Service

- [ ] **Implement SendGrid email client**
  - `send_run_completed`: Success notification with score
  - `send_run_failed`: Failure notification with error
  - HTML email templates

### 8.3 Orchestrator Workflow

> **Spec Reference**: `./specification/orchestrator.md` (Main Workflow section)

- [ ] **Main orchestration loop**
  1. **Phase 1: Query Understanding**
     - Update status: phase = "query_analysis"
     - Call QueryUnderstandingAgent
     - Resolve entities (CL, DrugBank/PubChem, UBERON, MONDO)
     - Classify task type (perturbational vs observational)
     - Store StructuredQuery in database
     - **Validate control strategy feasibility**:
       ```python
       if control_strategy == "synthetic_control":
           control_info = await prompt_agent.find_matched_controls(structured_query)
           structured_query.control_cells_available = bool(control_info)
           structured_query.control_cell_info = control_info
           if not control_info:
               structured_query.control_strategy_fallback = "query_as_control"
               control_strategy = "query_as_control"  # Fallback
               await db.update_run(run_id, control_strategy_effective="query_as_control")
       ```
  
  2. **Phase 2: Iteration Loop** (max 5 iterations)
     - **Prompt Generation**
       - Update status: phase = "prompt_generation"
       - Run parallel retrieval strategies (task-specific priority)
       - Match control cells if synthetic control
       - Rank and select candidates
     - **Inference**
       - Update status: phase = "inference"
       - Submit GPU Batch job (1x or 2x based on control strategy)
         - Query-as-control: Single inference run
         - Synthetic control: Two inference runs (perturbed + control prompts)
       - Poll for completion (10s intervals, 30min timeout)
     - **Evaluation**
       - Update status: phase = "evaluation"
       - Run GroundingEvaluator with task-specific routing
       - Adjust confidence based on control strategy
     - Store IterationRecord with control metadata
     - Check cancellation flag from database
     - Check convergence (score ≥ 7 or max iterations)
  
  3. **Phase 3: Output Generation**
     - Update status: phase = "output_generation"
     - Write final AnnData to GCS
     - Generate interpretation report (Markdown)
     - Write structured log (JSON)
     - Update status: "completed"
     - Send email notification with control strategy info

---

## PHASE 9: Docker Containers

> **Spec Reference**: `./specification/deployment.md`

### 9.1 API Container

- [ ] **Create Dockerfile.api**
  ```dockerfile
  # Stage 1: Build frontend
  FROM node:20-alpine AS frontend-build
  WORKDIR /app/frontend
  COPY frontend/package*.json ./
  RUN npm ci
  COPY frontend ./
  RUN npm run build
  
  # Stage 2: Python API
  FROM python:3.11-slim
  WORKDIR /app
  RUN apt-get update && apt-get install -y libpq-dev
  COPY api /app/api
  COPY shared /app/shared
  COPY pyproject.toml /app/
  RUN pip install -e ".[api]"
  COPY --from=frontend-build /app/frontend/out /app/frontend/out
  EXPOSE 8080
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
  ```

### 9.2 Orchestrator Container

- [ ] **Create Dockerfile.orchestrator**
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  RUN apt-get update && apt-get install -y git libpq-dev
  COPY orchestrator /app/orchestrator
  COPY shared /app/shared
  COPY pyproject.toml /app/
  RUN pip install -e ".[orchestrator]"
  CMD ["python", "-m", "orchestrator.main"]
  ```

### 9.3 Inference Container

- [ ] **Create Dockerfile.inference**
  ```dockerfile
  FROM nvcr.io/nvidia/pytorch:24.01-py3
  WORKDIR /app
  RUN git clone https://github.com/arcinstitute/STACK.git /app/stack
  RUN pip install -e /app/stack
  RUN pip install google-cloud-storage scanpy anndata
  COPY inference /app/inference
  ENTRYPOINT ["python", "-m", "inference.run_inference"]
  ```

### 9.4 Build & Push

- [ ] **Set up Cloud Build** (cloudbuild.yaml)
  - Build all three containers
  - Push to Artifact Registry
  - Tag with commit SHA and 'latest'

---

## PHASE 10: Deployment & Configuration

> **Spec References**: 
> - `./specification/deployment.md`
> - `./specification/configuration.md`

### 10.1 GCP Infrastructure

- [ ] **Set up VPC network**
  - Create VPC connector for Cloud SQL private IP
  - Configure firewall rules

- [ ] **Set up service accounts**
  - `cloud-run-sa`: Cloud SQL Client, Storage Admin, Batch Job Editor
  - `orchestrator-sa`: Same permissions
  - `inference-sa`: Storage Admin

- [ ] **Set up Secret Manager**
  - Database passwords
  - API keys (OpenAI, SendGrid, etc.)
  - Grant secret access to service accounts

### 10.2 Cloud Run Deployment

- [ ] **Deploy API container**
  - Region: us-east1
  - Min instances: 0, Max instances: 10
  - CPU: 2, Memory: 4 GB
  - Timeout: 60s
  - Service account: cloud-run-sa
  - Enable IAP authentication

### 10.3 Configuration Management

- [ ] **Set up Dynaconf configuration**
  - `settings.yaml`: Default settings
  - `.secrets.yaml`: Local secrets (gitignored)
  - Environment-specific overrides (dev, staging, prod)
  - LLM provider settings
  - Database connection strings
  - GCS bucket names

### 10.4 Monitoring & Logging

- [ ] **Set up Cloud Logging**
  - Structured logging with run_id
  - Log aggregation for all components

- [ ] **Set up Cloud Monitoring**
  - Uptime checks for Cloud Run
  - Batch job success/failure metrics
  - Alert policies for failures

---

## PHASE 11: Testing

> **Spec Reference**: `./specification/testing.md`

### 11.1 Backend Unit Tests

- [ ] **Set up pytest configuration**
  - Async test fixtures
  - Mock database client
  - Mock external APIs

- [ ] **Test API endpoints**
  - test_create_run
  - test_get_run_status
  - test_cancel_run
  - test_validate_control_strategy

- [ ] **Test retrieval strategies**
  - Test each strategy with mock database
  - Test ranking algorithm
  - Test diversity scoring

- [ ] **Test ontology resolution**
  - Test semantic search
  - Test graph traversal
  - Test OLS fallback

- [ ] **Test grounding evaluation**
  - Test DE analysis
  - Test pathway enrichment
  - Test score calculation

### 11.2 Frontend Unit Tests

- [ ] **Set up Jest + React Testing Library**
- [ ] **Test components**: RunForm, RunStatus, RunResults
- [ ] **Test hooks**: useRunPolling

### 11.3 Integration Tests

- [ ] **Test full orchestrator workflow**
  - Mock LLM responses
  - Mock STACK inference
  - Test iteration loop
  - Test convergence
  - Test cancellation

### 11.4 Load Testing

- [ ] **Test concurrent runs**
- [ ] **Test large dataset retrieval** (10M cells)
- [ ] **Test vector similarity search latency**

---

## PHASE 12: Documentation & Launch Prep

> **Spec References**: 
> - `./specification/README.md`
> - `./specification/dependencies.md`

### 12.1 Documentation

- [ ] **API documentation**: OpenAPI/Swagger spec
- [ ] **User guide**:
  - How to formulate queries
  - Control strategy selection guide
  - Interpreting results
  - Troubleshooting

- [ ] **Developer guide**:
  - Local development setup
  - Adding new retrieval strategies
  - Extending grounding evaluation
  - Deployment procedures

### 12.2 Data Validation

- [ ] **Validate atlas data quality**
  - Check cell counts match expected
  - Verify embeddings generated correctly
  - Spot-check ontology mappings

- [ ] **Validate index completeness**
  - All cells indexed
  - All embeddings present
  - HNSW indexes built correctly

### 12.3 Performance Optimization

- [ ] **Optimize database queries**
  - Add missing indexes
  - Tune HNSW parameters
  - Optimize connection pooling

- [ ] **Optimize LLM calls**
  - Batch where possible
  - Cache repeated queries
  - Use appropriate temperature/max_tokens

- [ ] **Optimize STACK inference**
  - Batch size tuning
  - GPU memory optimization

### 12.4 Launch Checklist

- [ ] **Security review**
  - IAP configuration verified
  - Service account permissions minimal
  - Secrets properly managed
  - No credentials in code

- [ ] **Cost estimation**
  
  | Control Strategy | 1 Iteration | 5 Iterations |
  |-----------------|-------------|--------------|
  | Query-as-Control | ~$0.30 | ~$1.25 |
  | Synthetic Control | ~$0.55 | ~$2.25 |

- [ ] **Backup strategy**
  - Database automated backups enabled
  - GCS versioning enabled
  - Disaster recovery plan documented

- [ ] **Monitoring dashboards**
  - Run success/failure rates
  - Average run duration
  - Grounding score distributions
  - Error rates by component

---

## Specification Documents Reference

All specification documents are located in the `./specification/` directory. Refer to these as the authoritative source for implementation details.

| Document | Path | Description |
|----------|------|-------------|
| **README** | `./specification/README.md` | Project overview and quick start |
| **Architecture** | `./specification/architecture.md` | System architecture, component responsibilities, data flow |
| **Backend API** | `./specification/backend-api.md` | FastAPI routes, request/response models, project structure |
| **Orchestrator** | `./specification/orchestrator.md` | CPU Batch job workflow, iteration loop, status updates |
| **Frontend** | `./specification/frontend.md` | Next.js pages, components, hooks, state management |
| **Database** | `./specification/database.md` | PostgreSQL schema, indexes, Cloud SQL configuration |
| **Prompt Retrieval** | `./specification/prompt-retrieval.md` | Retrieval strategies, ranking algorithm, cell selection |
| **Ontology Resolution** | `./specification/ontology-resolution.md` | Cell Ontology integration, semantic search, graph traversal |
| **Literature Search** | `./specification/literature-search.md` | PubMed/Semantic Scholar APIs, PDF acquisition, docling |
| **Data Models** | `./specification/data-models.md` | Pydantic schemas, enums, request/response models |
| **Agents** | `./specification/agents.md` | LangChain agents, subagents, system prompts |
| **Tools** | `./specification/tools.md` | LangChain tool definitions, database/knowledge/inference tools |
| **Testing** | `./specification/testing.md` | Test strategy, fixtures, mocking, integration tests |
| **Deployment** | `./specification/deployment.md` | Docker containers, Cloud Run, GCP Batch configuration |
| **Configuration** | `./specification/configuration.md` | Dynaconf settings, environment variables, secrets |
| **Dependencies** | `./specification/dependencies.md` | Python and Node.js dependencies |

---

## Project Directory Structure

```
haystack/
├── api/                         # Cloud Run API (thin layer)
│   ├── routes/
│   │   ├── runs.py              # Run management endpoints
│   │   ├── metadata.py          # Metadata endpoints
│   │   └── health.py            # Health check
│   ├── services/
│   │   ├── database.py          # Cloud SQL client (read status)
│   │   └── batch.py             # Batch job submission
│   ├── main.py                  # FastAPI app
│   └── config.py                # API configuration
│
├── orchestrator/                # CPU Batch job (agent workflow)
│   ├── agents/
│   │   ├── orchestrator.py      # Main orchestrator loop
│   │   ├── query_understanding.py
│   │   ├── prompt_generation.py
│   │   └── grounding_evaluation.py
│   ├── services/
│   │   ├── database.py          # Cloud SQL client (update status)
│   │   ├── batch.py             # GPU Batch job submission
│   │   ├── email.py             # SendGrid notifications
│   │   ├── ontology.py          # Cell Ontology service
│   │   └── literature.py        # Literature search service
│   ├── tools/                   # LangChain tools
│   │   ├── database_tools.py
│   │   ├── knowledge_tools.py
│   │   ├── inference_tools.py
│   │   └── enrichment_tools.py
│   └── main.py                  # Batch job entrypoint
│
├── inference/                   # GPU Batch job (STACK only)
│   └── run_inference.py         # STACK inference script
│
├── shared/                      # Shared code
│   └── models/                  # Pydantic schemas
│       ├── runs.py
│       ├── cells.py
│       └── queries.py
│
├── frontend/
│   ├── app/                     # Next.js App Router pages
│   ├── components/              # React components
│   ├── hooks/                   # Custom React hooks
│   ├── lib/                     # API client, utilities
│   └── stores/                  # Zustand stores
│
├── docker/
│   ├── Dockerfile.api           # Cloud Run container
│   ├── Dockerfile.orchestrator  # CPU Batch container
│   └── Dockerfile.inference     # GPU Batch container
│
├── scripts/
│   ├── init_schema.py           # Database initialization
│   ├── build_index.py           # Atlas indexing pipeline
│   ├── load_ontology.py         # Cell Ontology loading
│   └── seed_db.py               # Development data seeding
│
├── specification/               # Design documents
├── tests/                       # Test suite
├── docker-compose.yml           # Local development
├── cloudbuild.yaml              # CI/CD configuration
├── pyproject.toml               # Python dependencies
└── README.md
```

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Foundation & Infrastructure | 2-3 weeks | None |
| Phase 2: Core Backend Services | 1-2 weeks | Phase 1 |
| Phase 3: Prompt Retrieval Strategies | 2 weeks | Phase 2 |
| Phase 4: Agent Framework | 2-3 weeks | Phases 2, 3 |
| Phase 5: STACK Inference Integration | 1 week | Phase 4 |
| Phase 6: Cloud Run API | 1 week | Phase 4 |
| Phase 7: Frontend | 2 weeks | Phase 6 |
| Phase 8: Orchestrator Batch Job | 1-2 weeks | Phases 4, 5, 6 |
| Phase 9: Docker Containers | 1 week | Phases 6, 8 |
| Phase 10: Deployment & Configuration | 1 week | Phase 9 |
| Phase 11: Testing | 2 weeks | All phases |
| Phase 12: Documentation & Launch | 1 week | All phases |

**Total estimated duration: 16-20 weeks**

---

## Revision History

| Date | Changes |
|------|---------|
| 2026-01-19 | Initial draft |
| 2026-01-20 | **Major revision**: Updated all data models to match specification. Added dual grounding scores (perturbational vs observational). Added complete database schema with indexes. Fixed ranking weights (0.4/0.3/0.3). Added task-specific strategy priority lists. Added control strategy fallback logic. Fixed embedding model to text-embedding-3-small. Added IterationRecord and HaystackRun models. Enhanced API endpoints with full request/response models. Added detailed orchestrator workflow with control strategy handling. **Added spec references** to all phases and sections pointing to `./specification/` documents. |

