# Tool Specifications

### 6.0 LangChain Tool Conventions

All tools use the LangChain `@tool` decorator and should be designed to run both in unit tests and inside agents. Prefer async tools for I/O. If a tool needs access to runtime context (config, state, streaming), include a `runtime` parameter and treat it as optional to keep tests simple.

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@tool
async def resolve_cell_type(
    query: str,
    runtime: ToolRuntime | None = None,
) -> dict:
    """Resolve a cell type name to CL ID with provenance."""
    config = (runtime.config or {}) if runtime else {}
    # Use config for thresholds, collection names, etc.
    return {"label": "fibroblast", "cl_id": "CL:0000057"}
```

For tools that stream progress to the UI, emit structured progress events via `runtime.stream_writer` (if present). Failures should raise typed exceptions (see `specification/error-handling.md`) and include enough context to support retries.

### 6.1 Database Query Tools

```python
@tool
async def search_cells_by_perturbation(
    perturbation_name: str,
    cell_type_cl_id: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Search for cells by perturbation name with optional filters.
    
    Args:
        perturbation_name: Canonical perturbation name
        cell_type_cl_id: Optional Cell Ontology ID filter
        dataset: Optional dataset filter (parse_pbmc, openproblems, tabula_sapiens)
        limit: Maximum results to return
    
    Returns:
        List of cell groups matching criteria
    """
    ...


@tool
async def semantic_search_cells(
    query_text: str,
    search_type: Literal["perturbation", "cell_type", "sample_context"],
    top_k: int = 50,
    similarity_threshold: float = 0.7,
) -> list[dict]:
    """
    Vector similarity search for cells using text embeddings.
    
    Args:
        query_text: Natural language description
        search_type: Which embedding to search
        top_k: Number of results
        similarity_threshold: Minimum cosine similarity
    
    Returns:
        List of cell groups with similarity scores
    """
    ...


@tool
async def find_ontology_related_cells(
    cell_type_cl_id: str,
    max_distance: int = 2,
    perturbation_filter: Optional[str] = None,
) -> list[dict]:
    """
    Find cells with related cell types via Cell Ontology hierarchy.
    
    Args:
        cell_type_cl_id: Query cell type CL ID
        max_distance: Maximum ontology distance (parent/child levels)
        perturbation_filter: Optional perturbation filter
    
    Returns:
        List of cell groups with ontology distance
    """
    ...


@tool
async def find_donor_context_cells(
    cell_type_cl_id: str,
    target_donor_id: str,
    target_tissue: Optional[str] = None,
    target_disease_state: Optional[str] = None,
    max_results: int = 50,
) -> list[dict]:
    """
    Find cell groups from donors with similar clinical context.
    
    Args:
        cell_type_cl_id: Target cell type CL ID
        target_donor_id: Donor to exclude from references
        target_tissue: Optional UBERON ID
        target_disease_state: Optional MONDO ID
        max_results: Max results to return
    
    Returns:
        List of cell groups with donor similarity scores
    """
    ...


@tool
async def find_tissue_atlas_cells(
    cell_type_cl_id: str,
    target_tissue: Optional[str] = None,
    max_results: int = 50,
) -> list[dict]:
    """
    Find high-quality reference cell groups from tissue atlases.
    
    Args:
        cell_type_cl_id: Target cell type CL ID
        target_tissue: Optional UBERON ID
        max_results: Max results to return
    
    Returns:
        List of reference cell groups ranked by atlas priority and cell count
    """
    ...
```

### 6.2 Drug-Target Knowledge Tools

```python
@tool
async def get_drug_targets(
    perturbation: str,
    perturbation_type: str,
) -> dict:
    """
    Retrieve known targets for a perturbation.
    
    Args:
        perturbation: Name of drug/cytokine/gene
        perturbation_type: One of 'drug', 'cytokine', 'genetic'
    
    Returns:
        Dictionary with targets, target_types, sources, confidence
    
    Databases:
        - KEGG DRUG (for drugs)
        - UniProt (for receptors/binding partners)
        - Reactome (for signaling components)
    """
    ...


@tool
async def get_pathway_memberships(genes: list[str]) -> dict:
    """
    Get pathway memberships for a list of genes.
    
    Args:
        genes: List of gene symbols
    
    Returns:
        Dictionary with kegg_pathways, reactome_pathways, go_terms
    """
    ...


@tool
async def find_mechanistically_similar_perturbations(
    target_genes: list[str],
    pathways: list[str],
    available_perturbations: list[str],
) -> list[dict]:
    """
    Find perturbations sharing targets or pathways with query.
    
    Args:
        target_genes: Known target genes
        pathways: Associated pathways
        available_perturbations: Perturbations in atlas
    
    Returns:
        List of similar perturbations with overlap scores
    """
    ...
```

### 6.3 STACK Inference Tools

```python
@tool
async def run_stack_inference(
    prompt_cell_group_ids: list[str],
    query_cell_group_ids: list[str],
    run_id: str,
    iteration: int,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run STACK inference with selected prompt and query cells.
    
    Args:
        prompt_cell_group_ids: Cell group IDs for prompt
        query_cell_group_ids: Cell group IDs for query
        run_id: Current run ID
        iteration: Current iteration number
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dictionary with:
        - status: 'completed' or 'failed'
        - prediction_path: GCS path to predictions
        - duration_seconds: Inference time
        - metrics: Model quality metrics
    
    Implementation:
        Runs STACK Large (T=5) using asyncio executor
    """
    ...


@tool
async def extract_de_genes(
    prediction_path: str,
    control_path: str,
    lfc_threshold: float = 0.5,
    pval_threshold: float = 0.05,
) -> list[dict]:
    """
    Extract differentially expressed genes from predictions.
    
    Args:
        prediction_path: GCS path to prediction AnnData
        control_path: GCS path to control AnnData
        lfc_threshold: Minimum log2 fold change
        pval_threshold: Maximum adjusted p-value
    
    Returns:
        List of DE genes with statistics
    """
    ...
```

### 6.4 Enrichment and Evaluation Tools

```python
@tool
async def run_pathway_enrichment(
    genes: list[str],
    background_genes: Optional[list[str]] = None,
    databases: list[str] = ["GO_BP", "KEGG", "Reactome"],
) -> dict:
    """
    Run pathway enrichment analysis using gseapy.
    
    Args:
        genes: List of gene symbols
        background_genes: Optional background gene set
        databases: Databases to query
    
    Returns:
        Dictionary with enrichment results per database
    """
    ...


@tool
async def compute_grounding_score(
    query: StructuredQuery,
    predictions: AnnData,
    control_or_reference: AnnData,
) -> Union[GroundingScore, ObservationalGroundingScore]:
    """
    Compute composite biological grounding score.
    
    Args:
        query: Parsed query with task type and priors
        predictions: Predicted AnnData
        control_or_reference: Control (perturbational) or reference (observational) AnnData
    
    Returns:
        GroundingScore or ObservationalGroundingScore with component and composite scores
    """
    ...


class GroundingEvaluator:
    """Unified evaluator that handles both perturbational and observational tasks."""
    
    async def evaluate(
        self,
        query: StructuredQuery,
        predictions: AnnData,
        control_or_reference: AnnData,
    ) -> Union[GroundingScore, ObservationalGroundingScore]:
        """
        Evaluate predictions based on task type.
        """
        if query.task_type in [
            ICLTaskType.PERTURBATION_NOVEL_CELL_TYPES,
            ICLTaskType.PERTURBATION_NOVEL_SAMPLES,
        ]:
            return await self._evaluate_perturbational(query, predictions, control_or_reference)
        return await self._evaluate_observational(query, predictions, control_or_reference)
    
    async def _evaluate_observational(
        self,
        query: StructuredQuery,
        predictions: AnnData,
        reference: AnnData,
    ) -> ObservationalGroundingScore:
        """Evaluate observational ICL predictions."""
        ...
```

### 6.5 Literature Search Tools

```python
@tool
async def search_literature(
    query: str,
    max_results: int = 10,
    databases: list[str] | None = None,
) -> str:
    """
    Search scientific literature databases for relevant papers.
    
    Args:
        query: Search query (supports standard search syntax)
        max_results: Maximum results per database
        databases: Databases to search (pubmed, semantic_scholar, biorxiv)
        
    Returns:
        Formatted list of papers with titles, authors, abstracts
    """
    ...


@tool
async def acquire_full_text_paper(
    doi: str,
) -> str:
    """
    Acquire full-text paper content and convert to markdown.
    
    Tries preprint servers, CORE API, Europe PMC, and Unpaywall.
    Falls back to abstract if full text unavailable.
    
    Args:
        doi: Paper DOI (e.g., "10.1016/j.cell.2024.01.001")
        
    Returns:
        Paper content as markdown
    """
    ...


@tool
async def search_literature_evidence(
    genes: list[str],
    perturbation: str | None = None,
    cell_type: str | None = None,
    max_papers: int = 5,
) -> str:
    """
    Search for literature evidence supporting gene expression patterns.
    
    Args:
        genes: List of gene symbols to search for
        perturbation: Optional perturbation context
        cell_type: Optional cell type context
        max_papers: Maximum papers to return
        
    Returns:
        Evidence summary with citations
    """
    ...
```

### 6.6 Tool Input/Output Standards

- Inputs should be typed, validated, and described in docstrings to support automatic schema generation.
- Outputs should be structured dictionaries or Pydantic models where possible; avoid free-form strings except for text-heavy content (e.g., full-text paper markdown).
- Tools should be deterministic given fixed inputs; any randomness must be explicit and configurable.
- For heavy I/O or long-running tools, include timeouts and return partial results when safe.

---

## Related Specs

- `specification/agents.md`
- `specification/literature-search.md`
- `specification/biological-database-integration.md`
- `specification/error-handling.md`
- `specification/testing.md`
