# Testing Strategy

### 15.1 Backend Tests

```python
# backend/tests/conftest.py

import pytest
import asyncio
from httpx import AsyncClient
from backend.main import app
from backend.services.database import database


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def mock_database(mocker):
    """Mock database for unit tests."""
    mock_db = mocker.MagicMock()
    mocker.patch("backend.services.database.database", mock_db)
    return mock_db
```

```python
# backend/tests/test_api/test_runs.py

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_run(client: AsyncClient, mock_database):
    """Test creating a new run."""
    response = await client.post(
        "/api/v1/runs/",
        json={"query": "How would lung fibroblasts respond to TGF-beta?"},
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_get_run_status(client: AsyncClient, mock_database):
    """Test getting run status."""
    mock_database.get_run.return_value = {
        "run_id": "test_123",
        "status": "running",
        "iterations": [],
        "config": {"max_iterations": 5},
    }
    
    response = await client.get("/api/v1/runs/test_123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "test_123"
    assert data["status"] == "running"
```

### 15.2 Frontend Tests

```typescript
// frontend/__tests__/components/RunForm.test.tsx

import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RunForm } from "@/components/runs/RunForm";

const queryClient = new QueryClient();

describe("RunForm", () => {
  it("renders query input", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    expect(screen.getByPlaceholderText(/lung fibroblasts/i)).toBeInTheDocument();
  });
  
  it("disables submit for short queries", () => {
    render(
      <QueryClientProvider client={queryClient}>
        <RunForm />
      </QueryClientProvider>
    );
    
    const button = screen.getByRole("button", { name: /start analysis/i });
    expect(button).toBeDisabled();
  });
});
```

### 15.3 Agent + Tool Tests

Test tools as pure functions with mocked runtime/context, then add agent-level integration tests that exercise tool selection and orchestration. Separate fast unit tests from optional LLM-backed integration tests.

```python
# backend/tests/agents/tools/test_query_tool.py

import pytest
from backend.agents.tools.query_tool import query_tool


class MockRuntime:
    """Minimal ToolRuntime stand-in for unit tests."""

    def __init__(self, config=None):
        self.config = config or {}


def test_query_tool_returns_ranked_items(mocker):
    runtime = MockRuntime(config={"collections": {"papers": "test_papers"}})
    mocker.patch("backend.agents.tools.query_tool.search_index", return_value=[
        {"id": "p1", "score": 0.91},
        {"id": "p2", "score": 0.84},
    ])

    result = query_tool("tgf-beta fibroblasts", runtime=runtime)

    assert result["items"][0]["id"] == "p1"
    assert result["items"][0]["score"] >= result["items"][1]["score"]
```

```python
# backend/tests/agents/test_agent_integration.py

import pytest
from langchain.chat_models import init_chat_model
from backend.agents.main import build_agent


@pytest.mark.asyncio
async def test_agent_routes_to_tool(mocker):
    # Mock tool execution to make the agent deterministic
    mocker.patch("backend.agents.tools.query_tool.query_tool", return_value={
        "items": [{"id": "p1", "score": 0.91}],
    })
    model = init_chat_model("openai:gpt-5-mini", temperature=0)
    agent = build_agent(model=model)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Find TGF-beta fibroblast papers"}]})

    assert "p1" in str(result)
```

Optional integration tests should be marked (e.g., `@pytest.mark.integration`) and run with real model keys for end-to-end validation.

---

## Related Specs

- `specification/backend-api.md`
- `specification/orchestrator.md`
- `specification/frontend.md`
- `specification/agents.md`
- `specification/tools.md`
