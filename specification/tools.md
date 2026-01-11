# Tool Specifications

### 6.0 LangChain Tool Conventions

All tools use the LangChain `@tool` decorator and should be designed to run both in unit tests and inside agents. Prefer async tools for I/O. If a tool needs access to runtime context (config, state, streaming), include a `runtime` parameter and treat it as optional to keep tests simple.

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
        List of cell sets matching criteria
    """
    ...


@tool
async def search_cells_by_cell_type(
    cell_type_cl_id: str,
    dataset: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Search for cells by Cell Ontology ID with optional dataset filters.
    
    Args:
        cell_type_cl_id: Cell Ontology ID (e.g., CL:0000057)
        dataset: Optional dataset filter (parse_pbmc, openproblems, tabula_sapiens)
        limit: Maximum results to return
    
    Returns:
        List of cell sets matching criteria
    """
    ...


@tool
async def search_cells_by_donor(
    donor_id: str,
    cell_type_cl_id: Optional[str] = None,
    dataset: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Search for cells by donor ID with optional filters.
    
    Args:
        donor_id: Donor identifier (prefixed with dataset)
        cell_type_cl_id: Optional Cell Ontology ID filter
        dataset: Optional dataset filter (parse_pbmc, openproblems, tabula_sapiens)
        limit: Maximum results to return
    
    Returns:
        List of cell sets matching criteria
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
        List of cell sets with similarity scores
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
        List of cell sets with ontology distance
    """
    ...


@tool
async def find_donor_cells(
    donor_id: str,
    cell_type_cl_id: Optional[str] = None,
    tissue_uberon_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Find cell sets for a specific donor with optional filters.
    
    Args:
        donor_id: Donor identifier (prefixed with dataset)
        cell_type_cl_id: Optional Cell Ontology ID filter
        tissue_uberon_id: Optional tissue filter (UBERON ID)
        limit: Maximum results to return
    
    Returns:
        List of cell sets for the donor
    """
    ...


@tool
async def find_reference_cells(
    cell_type_cl_id: str,
    tissue_uberon_id: Optional[str] = None,
    disease_mondo_id: Optional[str] = None,
    max_results: int = 50,
) -> list[dict]:
    """
    Find reference cell sets for observational tasks.
    
    Args:
        cell_type_cl_id: Target Cell Ontology ID
        tissue_uberon_id: Optional tissue filter (UBERON ID)
        disease_mondo_id: Optional disease filter (MONDO ID)
        max_results: Max results to return
    
    Returns:
        List of reference cell sets with relevance scores
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
    Find cell sets from donors with similar clinical context.
    
    Args:
        cell_type_cl_id: Target cell type CL ID
        target_donor_id: Donor to exclude from references
        target_tissue: Optional UBERON ID
        target_disease_state: Optional MONDO ID
        max_results: Max results to return
    
    Returns:
        List of cell sets with donor similarity scores
    """
    ...


@tool
async def find_tissue_atlas_cells(
    cell_type_cl_id: str,
    target_tissue: Optional[str] = None,
    max_results: int = 50,
) -> list[dict]:
    """
    Find high-quality reference cell sets from tissue atlases.
    
    Args:
        cell_type_cl_id: Target cell type CL ID
        target_tissue: Optional UBERON ID
        max_results: Max results to return
    
    Returns:
        List of reference cell sets ranked by atlas priority and cell count
    """
    ...
```

### 6.2 Cell Ontology Resolution Tools

These tools enable agents to resolve free-text cell type labels to Cell Ontology (CL) IDs and navigate the CL hierarchy. They replace the previous `resolve_cell_type_tool` stub.

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
    
    Uses OpenAI embeddings (text-embedding-3-small) to find semantically similar
    CL terms. Input labels are automatically deduplicated.
    
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
    import yaml
    
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
    import yaml
    
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

### 6.3 Drug-Target Knowledge Tools

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

### 6.4 STACK Inference Tools

```python
@tool
async def run_stack_inference(
    prompt_cell_indices: list[int],
    query_cell_indices: list[int],
    run_id: str,
    iteration: int,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Run STACK inference with selected prompt and query cells.
    
    Args:
        prompt_cell_indices: Cell indices for prompt
        query_cell_indices: Cell indices for query
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

### 6.5 Enrichment and Evaluation Tools

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

### 6.6 Literature Search Tools

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

### 6.7 Tool Input/Output Standards

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
