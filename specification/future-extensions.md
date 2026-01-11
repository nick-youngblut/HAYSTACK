# Future Extensions

### 17.1 Post-MVP Features

| Feature | Priority | Description |
|---------|----------|-------------|
| User custom atlases | High | Allow users to upload their own H5AD files |
| Results caching | High | Cache API responses and enrichments |
| Batch queries | Medium | Process multiple queries efficiently |
| Validation agent | Medium | Compare predictions to experimental data |
| Jupyter integration | Low | Notebook interface for power users |
| Fine-tuning support | Low | Improve STACK with user feedback |
| **Pause/resume with feedback** | **Medium** | **Allow users to provide mid-run guidance to the agent** |

### 17.2 Pause/Resume with User Feedback (Deferred)

A deferred feature that would allow users to pause a running analysis and provide feedback to the agent:

**Concept:**
- User clicks "Pause for Feedback" during a run
- Agent checkpoints its state and presents current results
- User reviews intermediate predictions and provides guidance (e.g., "Focus more on fibroblast markers", "Exclude immune cell prompts")
- Agent incorporates feedback and resumes

**Why Deferred:**
- Requires agent state checkpointing and resume logic
- Needs UI for structured feedback input
- Timeout handling if user never responds
- Adds significant complexity to the iteration loop

**Alternative (MVP):**
- Users can cancel runs and start new ones with refined queries
- Run history allows comparison across attempts
- Email notification enables asynchronous workflow

### 17.3 Scalability Improvements

- Parallel prompt evaluation across multiple workers
- Distributed vector index with read replicas
- Result streaming with Server-Sent Events (SSE) for active monitoring
- Warm VM pools to reduce Batch job startup latency

### 17.4 Advanced Literature Features

- Citation network analysis for discovering related work
- Automatic paper summarization using LLM
- Literature-based hypothesis generation
- Integration with institutional library access for paywalled content
- Semantic search over indexed paper corpus

### 17.5 Integration Opportunities

- Benchling integration for experimental tracking
- GEO integration for validation data retrieval
- Slack notifications for run completion (alternative to email)
- Asana task creation for failed runs

---

## Related Specs

- `specification/architecture.md`
- `specification/literature-search.md`
- `specification/prompt-retrieval.md`
