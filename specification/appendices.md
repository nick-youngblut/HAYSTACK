# Appendices

## Grounding Score Calculation

```python
def compute_composite_score(
    pathway: int,
    target: int, 
    literature: int,
    network: int
) -> int:
    """
    Compute composite grounding score (perturbational tasks).
    
    Args:
        pathway: Pathway coherence score (1-10)
        target: Target activation score (1-10)
        literature: Literature support score (1-10)
        network: Network coherence score (1-10)
    
    Returns:
        Composite score (1-10)
    
    Weights:
    - Pathway coherence: 25%
    - Target activation: 30%
    - Literature support: 25%
    - Network coherence: 20%
    
    Asymmetric penalties:
    - Literature contradictions penalized more heavily than lack of support
    - Novel predictions (no literature) not penalized
    """
    base_score = (
        0.25 * pathway +
        0.30 * target +
        0.25 * literature +
        0.20 * network
    )
    return max(1, min(10, round(base_score)))
```

---

## Retrieval Strategy Hierarchy

See `prompt-retrieval.md` for detailed specification of the cell retrieval strategies:
- Direct Match Strategy
- Mechanistic Match Strategy
- Semantic Match Strategy
- Ontology-Guided Strategy
- Donor Context Strategy
- Tissue Atlas Strategy

## Related Specs

- `specification/prompt-retrieval.md`
- `specification/data-models.md`
