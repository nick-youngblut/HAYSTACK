# Sprint 07: Frontend

**Duration**: 2 weeks  
**Dependencies**: Sprint 06 (Cloud Run API)  
**Goal**: Implement the Next.js frontend for user interaction.

---

## Overview

> **Spec Reference**: `./specification/frontend.md`

This sprint implements the frontend:
- Next.js 14+ with App Router
- Run creation form with control strategy selection
- Status polling and display
- Results visualization

---

## Phase 1: Project Setup

### Task 1.1: Initialize Next.js Project

- [ ] **1.1.1** Create Next.js app:
  ```bash
  npx create-next-app@latest frontend --typescript --tailwind --app --src-dir=false
  ```

- [ ] **1.1.2** Install dependencies (per `./specification/dependencies.md`):
  ```bash
  npm install @tanstack/react-query@^5 zustand axios @headlessui/react@^2 \
    @heroicons/react clsx date-fns react-markdown tailwind-merge zod
  ```

- [ ] **1.1.3** Configure static export:
  ```javascript
  // next.config.js
  /** @type {import('next').NextConfig} */
  const nextConfig = {
    output: 'export',
    trailingSlash: true,
    images: { unoptimized: true },
  };
  module.exports = nextConfig;
  ```

---

### Task 1.2: Set Up Project Structure

> **Spec Reference**: `./specification/frontend.md` (Section 10.1)

- [ ] **1.2.1** Create directory structure:
  ```
  frontend/
  ├── app/
  │   ├── layout.tsx
  │   ├── page.tsx
  │   ├── runs/
  │   │   ├── new/page.tsx
  │   │   └── [id]/page.tsx
  ├── components/
  │   ├── runs/
  │   └── ui/
  ├── hooks/
  │   └── queries/
  ├── lib/
  │   └── api/
  └── stores/
  ```

---

### Task 1.3: Configure Providers

- [ ] **1.3.1** Create `components/Providers.tsx`:

```typescript
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
});

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster position="top-right" />
    </QueryClientProvider>
  );
}
```

---

## Phase 2: API Client

### Task 2.1: Implement API Client

- [ ] **2.1.1** Create `lib/api/client.ts`:

```typescript
import axios from "axios";

const api = axios.create({
  baseURL: "/api/v1",
  headers: { "Content-Type": "application/json" },
});

export interface CreateRunRequest {
  query: string;
  control_strategy: "query_as_control" | "synthetic_control";
  max_iterations?: number;
  score_threshold?: number;
}

export interface RunStatus {
  run_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  current_iteration: number;
  max_iterations: number;
  current_phase?: string;
  grounding_scores: number[];
  control_strategy?: string;
  control_strategy_effective?: string;
  error_message?: string;
}

export const createRun = (data: CreateRunRequest) => 
  api.post<RunStatus>("/runs", data);

export const getRunStatus = (runId: string) => 
  api.get<RunStatus>(`/runs/${runId}`);

export const cancelRun = (runId: string) => 
  api.post(`/runs/${runId}/cancel`);

export const getRunResult = (runId: string) => 
  api.get(`/runs/${runId}/result`);

export const validateControlStrategy = (query: string, strategy: string) =>
  api.post("/runs/validate-control-strategy", { query, control_strategy: strategy });
```

---

## Phase 3: Core Components

### Task 3.1: Implement RunForm Component

> **Spec Reference**: `./specification/frontend.md` (Section 10.2)

- [ ] **3.1.1** Create `components/runs/RunForm.tsx`:
  - Query textarea (min 10 chars)
  - Control strategy radio group
  - Pros/cons display for each strategy
  - Real-time control validation
  - Submit button with loading state

- [ ] **3.1.2** Implement control strategy options:

```typescript
const CONTROL_STRATEGIES = [
  {
    value: "synthetic_control",
    label: "Synthetic Control (Recommended)",
    description: "Compare perturbed vs unperturbed prompt predictions",
    pros: ["Higher confidence", "Cleaner DE results"],
    cons: ["Requires matched controls", "2x inference time"],
  },
  {
    value: "query_as_control",
    label: "Query as Control",
    description: "Use original query cells as control reference",
    pros: ["Faster", "Always available"],
    cons: ["May include artifacts", "Less rigorous"],
  },
];
```

- [ ] **3.1.3** Add control strategy warning when matched controls unavailable

---

### Task 3.2: Implement RunStatus Component

- [ ] **3.2.1** Create `components/runs/RunStatus.tsx`:
  - Status header with icon
  - Progress bar showing iteration
  - Current phase display
  - Grounding scores history chart
  - Control strategy indicator
  - Cancel button
  - Error message display

- [ ] **3.2.2** Create status icons:

```typescript
const STATUS_CONFIG = {
  pending: { icon: Clock, color: "text-gray-500", label: "Pending" },
  running: { icon: Loader2, color: "text-blue-500", label: "Running" },
  completed: { icon: CheckCircle, color: "text-green-500", label: "Completed" },
  failed: { icon: XCircle, color: "text-red-500", label: "Failed" },
  cancelled: { icon: Ban, color: "text-gray-500", label: "Cancelled" },
};
```

---

### Task 3.3: Implement RunResults Component

- [ ] **3.3.1** Create `components/runs/RunResults.tsx`:
  - Download buttons (AnnData, report, logs)
  - Final grounding score
  - Iteration history table
  - Top DE genes
  - Enriched pathways

---

## Phase 4: Hooks

### Task 4.1: Implement useRunPolling Hook

- [ ] **4.1.1** Create `hooks/useRunPolling.ts`:

```typescript
export function useRunPolling(runId: string, options = {}) {
  const { pollInterval = 15000 } = options;
  
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRunStatus(runId).then(r => r.data),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (["completed", "failed", "cancelled"].includes(status)) {
        return false;
      }
      return pollInterval;
    },
    refetchIntervalInBackground: true,
  });
}
```

---

### Task 4.2: Implement Query Hooks

- [ ] **4.2.1** Create `hooks/queries/useRuns.ts`:

```typescript
export function useCreateRun() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: createRun,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });
}

export function useRunList(page = 1) {
  return useQuery({
    queryKey: ["runs", page],
    queryFn: () => api.get(`/runs?page=${page}`).then(r => r.data),
  });
}
```

---

## Phase 5: Pages

### Task 5.1: Implement Home Page

- [ ] **5.1.1** Create `app/page.tsx`:
  - Recent runs list
  - Quick start button
  - Empty state for new users

---

### Task 5.2: Implement New Run Page

- [ ] **5.2.1** Create `app/runs/new/page.tsx`:
  - RunForm component
  - Redirect to status page on submission

---

### Task 5.3: Implement Run Detail Page

- [ ] **5.3.1** Create `app/runs/[id]/page.tsx`:
  - Fetch run by ID
  - RunStatus component with polling
  - RunResults when completed

---

## Phase 6: UI Components

### Task 6.1: Create Base Components

- [ ] **6.1.1** Create `components/ui/Button.tsx`
- [ ] **6.1.2** Create `components/ui/Modal.tsx`
- [ ] **6.1.3** Create `components/ui/LoadingSpinner.tsx`
- [ ] **6.1.4** Create `components/ui/ProgressBar.tsx`
- [ ] **6.1.5** Create `components/ui/Tooltip.tsx`

---

### Task 6.2: Create Layout Components

- [ ] **6.2.1** Create Sidebar navigation
- [ ] **6.2.2** Create Header with user info
- [ ] **6.2.3** Create PageLayout wrapper

---

## Phase 7: Styling

### Task 7.1: Configure Tailwind Theme

- [ ] **7.1.1** Set up color palette
- [ ] **7.1.2** Configure typography
- [ ] **7.1.3** Add custom animations

---

## Phase 8: Testing

### Task 8.1: Component Tests

- [ ] **8.1.1** Test RunForm validation
- [ ] **8.1.2** Test RunStatus display
- [ ] **8.1.3** Test polling behavior

---

## Definition of Done

- [ ] All pages implemented
- [ ] Status polling works correctly
- [ ] Control strategy UI complete
- [ ] Results download works
- [ ] Static export builds successfully
- [ ] Responsive design works

---

## Next Sprint

**Sprint 08: Orchestrator Batch Job** - Finalize the orchestrator entrypoint and email notifications.
