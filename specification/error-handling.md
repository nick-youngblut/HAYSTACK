# Error Handling

### 14.1 Error Types

```python
class HaystackError(Exception):
    """Base exception for HAYSTACK errors."""
    pass


class QueryParsingError(HaystackError):
    """Failed to parse user query."""
    pass


class CellRetrievalError(HaystackError):
    """Failed to retrieve cells from database."""
    pass


class InferenceError(HaystackError):
    """STACK inference failed."""
    pass


class EvaluationError(HaystackError):
    """Grounding evaluation failed."""
    pass


class ExternalAPIError(HaystackError):
    """External biological database API failed."""
    pass
```

### 14.2 Error Response Format

```python
class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str
    error_type: str
    message: str
    details: Optional[dict] = None
    run_id: Optional[str] = None
    
    
# Example error response
{
    "error": "inference_failed",
    "error_type": "InferenceError",
    "message": "STACK inference timed out after 30 minutes",
    "details": {
        "iteration": 2,
        "prompt_strategy": "mechanistic"
    },
    "run_id": "hay_20260110_abc123"
}
```

---

## Related Specs

- `specification/backend-api.md`
- `specification/orchestrator.md`
- `specification/tools.md`
