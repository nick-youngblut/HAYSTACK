# Frontend Specification

### 10.1 Project Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout with Providers
│   ├── page.tsx                # Home/dashboard
│   ├── globals.css             # Tailwind imports
│   ├── runs/
│   │   ├── page.tsx            # /runs - list runs
│   │   ├── new/page.tsx        # /runs/new - create run
│   │   └── [id]/page.tsx       # /runs/[id] - run detail
│   └── explore/
│       └── page.tsx            # /explore - browse cells
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── PageLayout.tsx
│   ├── runs/
│   │   ├── RunForm.tsx         # Query input form
│   │   ├── RunStatus.tsx       # Status display with polling
│   │   ├── RunResults.tsx      # Results visualization
│   │   └── RunHistory.tsx      # Past runs table
│   ├── explore/
│   │   ├── CellBrowser.tsx     # Atlas browser
│   │   └── SearchFilters.tsx
│   ├── ui/
│   │   ├── Button.tsx
│   │   ├── Modal.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ProgressBar.tsx
│   └── Providers.tsx           # React Query + Toaster
├── hooks/
│   ├── queries/
│   │   ├── useRuns.ts          # Run queries with polling
│   │   ├── useCells.ts
│   │   └── useMetadata.ts
│   └── useRunPolling.ts        # Status polling hook
├── stores/
│   ├── runStore.ts             # Current run state
│   └── uiStore.ts              # UI state
├── lib/
│   ├── api/
│   │   ├── client.ts
│   │   ├── runs.ts
│   │   └── cells.ts
│   └── query-client.ts
├── types/
│   ├── api.ts
│   └── runs.ts
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── next.config.js
```

### 10.2 Key Components

```typescript
// components/runs/RunForm.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { createRun } from "@/lib/api/runs";
import { Button } from "@/components/ui/Button";

export function RunForm() {
  const router = useRouter();
  const [query, setQuery] = useState("");
  
  const mutation = useMutation({
    mutationFn: createRun,
    onSuccess: (data) => {
      router.push(`/runs/${data.run_id}`);
    },
  });
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    mutation.mutate({ query });
  };
  
  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700">
          Describe your perturbation prediction task
        </label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="How would lung fibroblasts respond to TGF-beta treatment?"
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
          rows={4}
        />
      </div>
      
      <Button
        type="submit"
        disabled={mutation.isPending || query.length < 10}
      >
        {mutation.isPending ? "Starting..." : "Start Analysis"}
      </Button>
    </form>
  );
}
```

```typescript
// components/runs/RunProgress.tsx
"use client";

import { useRunProgress } from "@/hooks/useRunProgress";
import { ProgressBar } from "@/components/ui/ProgressBar";

interface RunProgressProps {
  runId: string;
}

export function RunProgress({ runId }: RunProgressProps) {
  const { status, currentIteration, maxIterations, currentScore, messages } = 
    useRunProgress(runId);
  
  const progress = (currentIteration / maxIterations) * 100;
  
  return (
    <div className="space-y-4">
      <div className="flex justify-between text-sm">
        <span>Iteration {currentIteration} of {maxIterations}</span>
        {currentScore && <span>Score: {currentScore}/10</span>}
      </div>
      
      <ProgressBar value={progress} />
      
      <div className="space-y-2">
        {messages.map((msg, i) => (
          <div key={i} className="text-sm text-gray-600">
            {msg.type === "iteration_start" && (
              <span>🔄 Starting iteration {msg.data?.iteration}...</span>
            )}
            {msg.type === "progress" && (
              <span>📊 {msg.data?.message}</span>
            )}
            {msg.type === "iteration_end" && (
              <span>✅ Iteration {msg.data?.iteration} complete (score: {msg.data?.score})</span>
            )}
          </div>
        ))}
      </div>
      
      {status === "completed" && (
        <div className="text-green-600 font-medium">
          ✅ Analysis complete! Final score: {currentScore}/10
        </div>
      )}
      
      {status === "failed" && (
        <div className="text-red-600 font-medium">
          ❌ Analysis failed. Please try again.
        </div>
      )}
    </div>
  );
}
```

```typescript
// hooks/useRunPolling.ts
import { useQuery } from "@tanstack/react-query";
import { getRunStatus } from "@/lib/api/runs";
import { RunStatus } from "@/types/runs";

interface UseRunPollingOptions {
  /** Polling interval in milliseconds (default: 15000 = 15 seconds) */
  pollInterval?: number;
  /** Whether to enable polling (default: true) */
  enabled?: boolean;
}

export function useRunPolling(
  runId: string,
  options: UseRunPollingOptions = {}
) {
  const { pollInterval = 15000, enabled = true } = options;

  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: () => getRunStatus(runId),
    enabled: enabled && !!runId,
    // Poll while run is in progress
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Stop polling when run is finished
      if (status === "completed" || status === "failed" || status === "cancelled") {
        return false;
      }
      return pollInterval;
    },
    // Keep polling even when window is not focused
    refetchIntervalInBackground: true,
    // Don't refetch on window focus - use interval instead
    refetchOnWindowFocus: false,
  });

  const run = query.data;

  return {
    // Query state
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    
    // Run data
    run,
    status: run?.status ?? "pending",
    currentIteration: run?.current_iteration ?? 0,
    maxIterations: run?.max_iterations ?? 5,
    currentPhase: run?.current_phase,
    groundingScores: run?.grounding_scores ?? [],
    errorMessage: run?.error_message,
    
    // Derived state
    isFinished: ["completed", "failed", "cancelled"].includes(run?.status ?? ""),
    isRunning: run?.status === "running",
    latestScore: run?.grounding_scores?.length 
      ? run.grounding_scores[run.grounding_scores.length - 1] 
      : null,
    
    // Actions
    refetch: query.refetch,
  };
}
```

```typescript
// components/runs/RunStatus.tsx
"use client";

import { useRunPolling } from "@/hooks/useRunPolling";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cancelRun } from "@/lib/api/runs";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { LoadingSpinner } from "@/components/ui/LoadingSpinner";

interface RunStatusProps {
  runId: string;
}

const PHASE_LABELS: Record<string, string> = {
  pending: "Queued",
  query_analysis: "Analyzing query",
  prompt_generation: "Generating prompts",
  inference: "Running STACK inference",
  evaluation: "Evaluating predictions",
  output_generation: "Generating outputs",
};

export function RunStatus({ runId }: RunStatusProps) {
  const queryClient = useQueryClient();
  
  const {
    status,
    currentIteration,
    maxIterations,
    currentPhase,
    groundingScores,
    errorMessage,
    isLoading,
    isRunning,
    isFinished,
    latestScore,
  } = useRunPolling(runId);

  const cancelMutation = useMutation({
    mutationFn: () => cancelRun(runId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run", runId] });
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <LoadingSpinner />
        <span className="ml-2">Loading run status...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {isRunning && <LoadingSpinner size="sm" />}
          <span className="text-lg font-medium">
            {status === "completed" && "✅ Completed"}
            {status === "failed" && "❌ Failed"}
            {status === "cancelled" && "⏹️ Cancelled"}
            {status === "running" && PHASE_LABELS[currentPhase ?? ""] ?? "Running"}
            {status === "pending" && "⏳ Queued"}
          </span>
        </div>
        
        {isRunning && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => cancelMutation.mutate()}
            disabled={cancelMutation.isPending}
          >
            {cancelMutation.isPending ? "Cancelling..." : "Cancel Run"}
          </Button>
        )}
      </div>

      {/* Progress Bar */}
      {!isFinished && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm text-gray-600">
            <span>Iteration {currentIteration} of {maxIterations}</span>
            {latestScore !== null && <span>Score: {latestScore}/10</span>}
          </div>
          <ProgressBar 
            value={currentIteration} 
            max={maxIterations}
            phase={currentPhase}
          />
        </div>
      )}

      {/* Grounding Scores History */}
      {groundingScores.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Grounding Scores</h3>
          <div className="flex space-x-2">
            {groundingScores.map((score, idx) => (
              <div
                key={idx}
                className={`
                  px-3 py-1 rounded-full text-sm font-medium
                  ${score >= 7 ? "bg-green-100 text-green-800" : 
                    score >= 5 ? "bg-yellow-100 text-yellow-800" : 
                    "bg-red-100 text-red-800"}
                `}
              >
                Iter {idx + 1}: {score}/10
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {status === "failed" && errorMessage && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium">Error</h3>
          <p className="text-red-700 text-sm mt-1">{errorMessage}</p>
        </div>
      )}

      {/* Email Notification Note */}
      {isRunning && (
        <p className="text-sm text-gray-500">
          💌 You'll receive an email when this run completes. Feel free to close this page.
        </p>
      )}
    </div>
  );
}
```

---

## Related Specs

- `specification/backend-api.md`
- `specification/output.md`
