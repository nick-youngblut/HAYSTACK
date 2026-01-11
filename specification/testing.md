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

---

## Related Specs

- `specification/backend-api.md`
- `specification/orchestrator.md`
- `specification/frontend.md`
