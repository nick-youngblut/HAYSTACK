# Data Models

### 4.1 Core Data Models (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal, Union
from enum import Enum
from datetime import datetime


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


class StructuredQuery(BaseModel):
    """Parsed representation of a user query."""
    
    raw_query: str = Field(description="Original user query text")
    task_type: ICLTaskType = Field(description="Resolved ICL task type")
    
    # Cell type resolution
    cell_type_query: str = Field(description="Extracted cell type from query")
    cell_type_cl_id: Optional[str] = Field(description="Resolved Cell Ontology ID")
    cell_type_synonyms: list[str] = Field(default_factory=list)
    
    # Perturbation resolution (optional for observational tasks)
    perturbation_query: Optional[str] = Field(default=None, description="Extracted perturbation from query")
    perturbation_type: Optional[PerturbationType] = None
    perturbation_resolved: Optional[str] = Field(default=None, description="Canonical name")
    perturbation_external_ids: dict[str, str] = Field(default_factory=dict)
    expected_targets: list[str] = Field(default_factory=list)
    expected_pathways: list[str] = Field(default_factory=list)
    
    # Observational context (optional for perturbational tasks)
    target_donor_id: Optional[str] = Field(default=None, description="Donor to impute for")
    target_tissue: Optional[str] = Field(default=None, description="Target tissue (UBERON ID)")
    target_disease_state: Optional[str] = Field(default=None, description="Disease state (MONDO ID)")
    target_condition: Optional[str] = Field(default=None, description="Free-text condition")
    reference_donor_id: Optional[str] = Field(default=None, description="Reference donor for query cells")
    reference_dataset: Optional[str] = Field(default=None, description="Reference dataset for query cells")
    
    # Biological context (task-agnostic)
    expected_marker_genes: list[str] = Field(default_factory=list)
    expected_tissue_genes: list[str] = Field(default_factory=list)
    literature_context: Optional[str] = None


class PromptCandidate(BaseModel):
    """A candidate prompt configuration."""
    
    strategy: Literal[
        "direct",
        "mechanistic",
        "semantic",
        "ontology",
        "donor_context",
        "tissue_atlas",
    ]
    prompt_cell_indices: list[int] = Field(description="Selected cell indices for prompt")
    
    # Metadata for ranking
    similarity_score: Optional[float] = None
    mechanistic_score: Optional[float] = None
    ontology_distance: Optional[int] = None
    
    # Explanation
    rationale: str = Field(description="Why this prompt was selected")


class GroundingScore(BaseModel):
    """Biological grounding evaluation result."""
    
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

    # Feedback for next iteration
    improvement_suggestions: list[str]

class IterationRecord(BaseModel):
    """Record of a single iteration."""
    
    iteration_number: int
    prompt_candidates: list[PromptCandidate]
    selected_prompt: PromptCandidate
    grounding_score: Union[GroundingScore, ObservationalGroundingScore]
    duration_seconds: float
    
    # Artifacts
    prediction_gcs_path: Optional[str] = None


class HaystackRun(BaseModel):
    """Complete run record."""
    
    # Metadata
    run_id: str
    user_email: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    status: Literal["running", "completed", "failed", "cancelled"]
    
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


class RunListResponse(BaseModel):
    """Response for listing runs."""
    
    runs: list[HaystackRun]
    total: int
    page: int
    page_size: int
```

### 4.2 API Request/Response Models

```python
class CreateRunRequest(BaseModel):
    """Request to create a new HAYSTACK run."""
    
    query: str = Field(description="Natural language query", min_length=10)
    
    # Optional overrides
    max_iterations: Optional[int] = Field(default=None, ge=1, le=10)
    score_threshold: Optional[int] = Field(default=None, ge=1, le=10)
    llm_provider: Optional[Literal["anthropic", "openai", "google_genai"]] = None
    llm_model: Optional[str] = None
    random_seed: Optional[int] = None


class RunStatusResponse(BaseModel):
    """Response with run status."""
    
    run_id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    current_iteration: int
    max_iterations: int
    current_score: Optional[int] = None
    message: str


class RunResultResponse(BaseModel):
    """Response with run results."""
    
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


class RunPhase(str, Enum):
    """Current phase of a run for status reporting."""
    PENDING = "pending"
    QUERY_ANALYSIS = "query_analysis"
    PROMPT_GENERATION = "prompt_generation"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    OUTPUT_GENERATION = "output_generation"


class RunStatusResponse(BaseModel):
    """Response model for run status polling."""
    
    run_id: str
    status: RunStatus
    current_iteration: int = 0
    max_iterations: int
    current_phase: Optional[RunPhase] = None
    grounding_scores: list[int] = Field(default_factory=list)  # Score per iteration
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    
    # User info
    user_email: str


class EmailNotification(BaseModel):
    """Email notification configuration."""
    
    recipient_email: str
    subject: str
    template: Literal["run_completed", "run_failed", "run_cancelled"]
    template_data: dict = Field(default_factory=dict)
```

### 4.3 Configuration Models

```python
class LLMConfig(BaseModel):
    """LLM backend configuration."""
    
    provider: Literal["anthropic", "openai", "google_genai"]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Provider-specific
    api_key_env_var: str = Field(
        description="Environment variable containing API key"
    )


class IterationConfig(BaseModel):
    """Iteration control configuration."""
    
    max_iterations: int = 5
    score_threshold: int = 7  # 1-10 scale
    plateau_window: int = 3  # Stop if no improvement over N iterations
    min_improvement: int = 1  # Minimum score improvement to not count as plateau


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    # Cloud SQL connection
    instance_connection_name: str
    database_name: str = "haystack"
    user: str = "haystack_app"
    
    # Connection pool
    pool_size: int = 5
    max_overflow: int = 10


class GCSConfig(BaseModel):
    """Google Cloud Storage configuration."""
    
    project_id: str
    bucket_name: str
    
    # Paths within bucket
    atlases_prefix: str = "atlases/"
    stack_model_prefix: str = "models/stack/"
    results_prefix: str = "results/"


class DatabaseAPIConfig(BaseModel):
    """Biological database API configuration."""
    
    # Retry settings
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    
    # Rate limiting
    requests_per_minute: int = 30


class HaystackConfig(BaseModel):
    """Complete HAYSTACK configuration."""
    
    # Environment
    environment: Literal["dev", "prod"] = "dev"
    debug: bool = False
    
    # Components
    llm: LLMConfig
    iteration: IterationConfig
    database: DatabaseConfig
    gcs: GCSConfig
    database_apis: DatabaseAPIConfig
    
    # STACK model
    stack_checkpoint_path: str  # GCS path
    stack_genelist_path: str    # GCS path
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
```

---

## Related Specs

- `specification/backend-api.md`
- `specification/database.md`
- `specification/agents.md`
