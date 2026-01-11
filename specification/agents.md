# Agent Specifications

### 5.1 Orchestrator Agent

The orchestrator is the main entry point, implemented using DeepAgents with FastAPI integration.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import init_chat_model

async def create_orchestrator(config: HaystackConfig, run_id: str):
    """
    Create orchestrator agent for a HAYSTACK run.
    
    Args:
        config: HAYSTACK configuration
        run_id: Unique run identifier
    
    Returns:
        Configured DeepAgent orchestrator
    """
    # Initialize Cell Ontology service (singleton)
    from haystack.orchestrator.services.ontology import CellOntologyService
    await CellOntologyService.initialize(config)

    # Initialize model
    model = init_chat_model(
        f"{config.llm.provider}:{config.llm.model}",
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )

    orchestrator = create_deep_agent(
        model=model,
        tools=[
            # Direct tools
            check_convergence_tool,
            generate_report_tool,
            run_stack_inference_tool,
            save_results_to_gcs_tool,
        ],
        subagents=[
            query_understanding_subagent,
            prompt_generation_subagent,
            grounding_evaluation_subagent,
        ],
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        backend=FilesystemBackend(root_dir=f"/tmp/haystack/{run_id}"),
    )
    
    return orchestrator
```

**Orchestrator System Prompt**:
```
You are the HAYSTACK orchestrator, an AI system that improves single-cell predictions through iterative knowledge-guided prompting. You support both perturbational and observational in-context learning tasks.

TASK TYPES:
1. PERTURBATIONAL ICL: Predict effects of drugs/cytokines/genetic perturbations
   - Prompts: Perturbed cells
   - Queries: Control cells (unperturbed)
   - Goal: Generate perturbed expression profiles

2. OBSERVATIONAL ICL: Predict cell type expression in a specific donor/condition
   - Prompts: Cell types from target donor (the context you want to predict in)
   - Queries: Cell types from reference donors (providing the cell type template)
   - Goal: Generate donor-specific cell type expression

3. HYBRID ICL: Cross-dataset cell type generation
   - Prompts: Cells from dataset A (defining the biological context)
   - Queries: Cell types from dataset B (providing cell type templates)
   - Goal: Generate cell types absent from dataset A

Your workflow:
1. UNDERSTAND: Parse the query to identify task type, cell type(s), and biological context
2. GENERATE: Create prompt candidates using appropriate strategies for the task type
3. INFER: Run STACK inference with selected prompts
4. EVALUATE: Assess biological grounding using task-appropriate metrics
5. DECIDE: If score ≥ threshold OR max iterations reached → finalize; else → refine

Key principles:
- Always explain your reasoning before taking actions
- Use biological knowledge to guide prompt selection
- Consider multiple strategies in parallel
- Learn from evaluation feedback to improve subsequent iterations
- Be conservative with iteration count; stop when predictions are well-grounded

For OBSERVATIONAL tasks, focus on:
- Finding reference donors with similar clinical profiles
- Selecting high-quality reference cell populations
- Evaluating cell type marker expression and tissue signatures
```

### 5.2 Query Understanding Subagent

```python
query_understanding_subagent = create_deep_agent(
    model=model,
    tools=[
        # Cell Ontology tools (primary for cell type resolution)
        resolve_cell_type_semantic,      # Maps free-text → CL ID via embeddings
        get_cell_type_neighbors,          # Navigates CL hierarchy
        query_cell_ontology_ols,          # Fallback to OLS API

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
    ],
    system_prompt=QUERY_UNDERSTANDING_PROMPT,
)
```

**System Prompt**:
```
You are a biological query understanding agent for HAYSTACK. Your job is to parse 
natural language queries and resolve biological entities to standardized identifiers.

## TASK TYPE DETERMINATION

1. If query mentions drug/cytokine/perturbation effects → PERTURBATIONAL
2. If query asks to predict/impute cell types for a donor → OBSERVATIONAL
3. If query involves cross-dataset generation → HYBRID

## CELL TYPE RESOLUTION (CRITICAL)

For ALL tasks, you MUST resolve cell types to Cell Ontology (CL) IDs:

### Step 1: Semantic Search (Primary Method)
Use `resolve_cell_type_semantic` to map free-text cell types to CL IDs:
- Example: "lung fibroblast" → CL:0002553 (fibroblast of lung)
- Example: "activated T cell" → CL:0000911 (activated T cell)
- Example: "CD8+ T cell" → CL:0000625 (CD8-positive, alpha-beta T cell)

Always check the distance score:
- distance < 0.3: High confidence match
- distance 0.3-0.5: Good match, verify name
- distance 0.5-0.7: Weak match, consider alternatives
- distance > 0.7: No reliable match

### Step 2: OLS Fallback
If semantic search returns "No ontology ID found" or only weak matches:
- Use `query_cell_ontology_ols` for keyword-based lookup
- OLS is better for exact term names or abbreviations

### Step 3: Hierarchy Navigation
Use `get_cell_type_neighbors` to:
- Find parent types (more general) if exact match unavailable
- Find child types (more specific) for refinement
- Identify sibling types for alternative matches

Example workflow for "cardiac fibroblast":
1. resolve_cell_type_semantic("cardiac fibroblast") 
   → May return weak match
2. query_cell_ontology_ols("cardiac fibroblast")
   → Finds CL:0002548 (cardiac fibroblast)
3. get_cell_type_neighbors("CL:0002548")
   → Shows parent: CL:0000057 (fibroblast)
   → Shows siblings: CL:0002553 (fibroblast of lung), etc.

### Step 4: Capture Synonyms
From resolution results, extract and store:
- The best matching CL ID in `cell_type_cl_id`
- Alternative names in `cell_type_synonyms`
- The canonical name from CL in `cell_type_name`

## OTHER ENTITY RESOLUTION

For PERTURBATIONAL tasks:
- Resolve perturbation name using `resolve_perturbation_tool`
- Get known targets via `get_drug_targets_tool`
- Identify affected pathways via `get_pathway_priors_tool`

For OBSERVATIONAL tasks:
- Resolve tissue to UBERON ID (if mentioned)
- Resolve disease state to MONDO ID (if mentioned)
- Identify target donor/sample

## BIOLOGICAL CONTEXT GATHERING

When targets or pathways are unclear:
1. Search literature: `search_literature_tool` for mechanism studies
2. Extract findings from papers: `acquire_full_text_paper_tool`
3. Use evidence to populate `expected_targets` and `expected_pathways`

For cell types, gather marker genes:
- Use `get_cell_type_markers_tool` with the resolved CL ID
- Store in `expected_marker_genes`

## OUTPUT

Return a StructuredQuery with ALL resolved information:
- `cell_type_cl_id` MUST be populated for successful retrieval
- `cell_type_synonyms` should include alternative names
- Include confidence notes in `literature_context` if resolution was uncertain

Be thorough in cell type resolution - this information guides prompt selection.
```

**Cell Type Resolution Fallback (Error Handling)**:
```python
async def resolve_cell_type_with_fallback(label: str) -> dict:
    """
    Resolve cell type with graceful fallback through multiple methods.
    
    Resolution order:
    1. Semantic search (embedding-based)
    2. OLS keyword search (exact matching)
    3. Return unresolved with warning
    """
    from haystack.orchestrator.services.ontology import CellOntologyService
    
    service = CellOntologyService.get_instance()
    
    # Try semantic search first
    try:
        results = await service.semantic_search(
            labels=[label],
            k=3,
            distance_threshold=0.7,
        )
        
        if results.get(label) and not isinstance(results[label], str):
            best = results[label][0]
            return {
                "resolved": True,
                "term_id": best["term_id"],
                "term_name": best["name"],
                "confidence": 1.0 - best["distance"],
                "method": "semantic",
                "alternatives": results[label][1:] if len(results[label]) > 1 else [],
            }
    except Exception as e:
        logger.warning(f"Semantic search failed for '{label}': {e}")
    
    # Fall back to OLS
    try:
        ols_results = await service.query_ols(search_terms=[label])
        
        if ols_results.get(label):
            best = ols_results[label][0]
            return {
                "resolved": True,
                "term_id": best["term_id"],
                "term_name": best["name"],
                "confidence": 0.8,  # Lower confidence for OLS
                "method": "ols",
                "alternatives": ols_results[label][1:] if len(ols_results[label]) > 1 else [],
            }
    except Exception as e:
        logger.warning(f"OLS query failed for '{label}': {e}")
    
    # No resolution found
    return {
        "resolved": False,
        "query_label": label,
        "method": "none",
        "warning": "Could not resolve cell type to CL ID",
    }
```

### 5.3 Prompt Generation Subagent

```python
prompt_generation_subagent = create_deep_agent(
    model=model,
    tools=[
        # Database retrieval tools
        search_cells_by_perturbation,
        search_cells_by_cell_type,
        search_cells_by_donor,
        semantic_search_cells,
        find_donor_cells,
        find_reference_cells,

        # Cell Ontology tools (for ontology-guided retrieval)
        get_cell_type_neighbors,
        resolve_cell_type_semantic,

        # Strategy execution
        execute_retrieval_strategy,
        rank_candidates,
        select_prompt_cells,
    ],
    system_prompt=PROMPT_GENERATION_PROMPT,
)
```

**System Prompt**:
```
You are a prompt generation agent for HAYSTACK. Your job is to find optimal 
prompt cells for STACK's in-context learning.

## RETRIEVAL STRATEGIES

Execute strategies in priority order based on task type:

### For PERTURBATIONAL tasks:
1. **Direct Match** (highest priority): Exact perturbation + cell type
2. **Mechanistic Match**: Same target genes/pathways
3. **Semantic Match**: Similar perturbation description
4. **Ontology-Guided**: Related cell types via CL hierarchy

### For OBSERVATIONAL tasks:
1. **Donor Context**: Same donor, different cell types
2. **Tissue Atlas**: Same tissue, different donors
3. **Ontology-Guided**: Related cell types via CL hierarchy
4. **Semantic Match**: Similar sample context

## ONTOLOGY-GUIDED RETRIEVAL

When exact cell type match is unavailable, use the Cell Ontology:

### Step 1: Get Related Types
Example:
    get_cell_type_neighbors("CL:0002553")  # fibroblast of lung

Returns parent, child, and sibling types with relationship info.

### Step 2: Search by Related Types
For each related type (prioritize parents over children):
- Search for cells with the related cell type
- Apply perturbation/donor filters as appropriate
- Score candidates by ontology distance

### Step 3: Score by Hierarchy Distance
- Parent type: distance = 1
- Grandparent: distance = 2
- Child type: distance = 1
- Sibling type: distance = 2

Weight scores inversely with distance:
    relevance_score = 1.0 / (ontology_distance + 1)

## CANDIDATE RANKING

Combine strategy scores:
- Relevance score: How well candidate matches query
- Diversity score: Coverage of different contexts
- Quality score: Cell count, data quality metrics

Final score = 0.5 * relevance + 0.3 * diversity + 0.2 * quality

## OUTPUT

Return a PromptConfiguration with:
- Selected prompt cell sets (selection criteria, cell_indices)
- Strategy used for each group
- Combined relevance score
- Rationale for selection
```

### 5.4 Grounding Evaluation Subagent

```python
grounding_evaluation_subagent = create_deep_agent(
    model=model,
    tools=[
        extract_de_genes_tool,
        run_pathway_enrichment_tool,
        check_target_activation_tool,
        search_literature_evidence_tool,
        acquire_full_text_paper_tool,
        build_gene_network_tool,
        get_cell_type_markers_tool,
        get_tissue_signature_tool,
        identify_donor_signature_tool,
        compute_grounding_score_tool,
    ],
    system_prompt=GROUNDING_EVALUATION_PROMPT,
)
```

**System Prompt**:
```
You are a biological grounding evaluation agent. Your job is to assess how well STACK predictions align with biological knowledge.

Evaluation criteria by task type (each scored 1-10):

PERTURBATIONAL:
1. PATHWAY COHERENCE: Do enriched pathways match expected biology?
2. TARGET ACTIVATION: Are known targets differentially expressed correctly?
3. LITERATURE SUPPORT: Do predictions have published evidence?
   - 9-10: Multiple papers directly support predictions with matching direction
   - 7-8: Some supporting evidence, no contradictions
   - 5-6: Limited evidence, results plausible based on pathway biology
   - 3-4: No direct evidence, results speculative
   - 1-2: Evidence contradicts predictions
4. NETWORK COHERENCE: Do DE genes form connected functional modules?

OBSERVATIONAL:
1. MARKER GENE EXPRESSION: Are canonical markers expressed as expected?
2. TISSUE SIGNATURE MATCH: Do predictions match tissue-specific patterns?
3. DONOR EFFECT CAPTURE: Are donor/disease effects preserved?
4. CELL TYPE COHERENCE: Is the transcriptional state coherent?

Compute a composite score and provide actionable feedback for improvement.

Be critical but fair - novel predictions that make biological sense should not be penalized.
```

---

## Related Specs

- `specification/tools.md`
- `specification/prompt-retrieval.md`
- `specification/literature-search.md`
- `specification/orchestrator.md`
