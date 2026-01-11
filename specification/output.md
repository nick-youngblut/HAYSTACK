# Output Specification

### 13.1 Output Files

| File | Format | Contents |
|------|--------|----------|
| `{run_id}_predictions.h5ad` | AnnData | Predicted expression with metadata |
| `{run_id}_report.html` | HTML | Interactive interpretation report |
| `{run_id}_log.json` | JSON | Complete execution log |

### 13.2 AnnData Structure

```python
predictions.obs:
    - cell_id: str
    - task_type: str
    - cell_type_query: str
    - perturbation_query: Optional[str]
    - target_donor_id: Optional[str]
    - target_tissue: Optional[str]
    - target_disease_state: Optional[str]
    - prompt_strategy: str
    - iteration: int
    - grounding_score: int

predictions.var:
    - gene_symbol: str
    - predicted_lfc: float
    - predicted_pval: float
    - is_de: bool
    - target_gene: bool

predictions.uns:
    - run_id: str
    - query: str
    - config: dict
    - iterations: list[dict]
    - enrichment_results: dict
```

---

## Related Specs

- `specification/backend-api.md`
- `specification/orchestrator.md`
