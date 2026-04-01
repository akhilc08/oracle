/** Typed API client for all Oracle backend endpoints. */

const BASE = "/api/v1";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

// --- Types ---

export interface PortfolioState {
  cash: number;
  positions: Record<string, PositionData>;
  total_value: number;
  pnl: number;
  position_count: number;
}

export interface PositionData {
  direction: string;
  entry_price: number;
  current_price: number;
  quantity: number;
  value: number;
  unrealized_pnl: number;
  conviction_at_entry: number;
  research_trace_id: string;
  category: string;
}

export interface SystemStatus {
  running: boolean;
  agents: Record<string, AgentStatus>;
  bus: { registered_agents: string[]; message_history_count: number };
  portfolio: PortfolioState;
  cache: { size: number; hits: number; misses: number; hit_rate: number };
}

export interface AgentStatus {
  agent_id: string;
  status: string;
  tasks_completed: number;
}

export interface EvalStats {
  total_predictions: number;
  overall_accuracy: number;
  brier_score: number;
  alpha_rate: number;
  by_category: Record<string, { count: number; accuracy: number; brier_score: number }>;
}

export interface PortfolioHistory {
  id: number;
  total_value: number;
  cash: number;
  positions_value: number;
  pnl: number;
  recorded_at: string;
}

export interface GraphSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphNode {
  id: string;
  label: string;
  group: number;
  title: string;
  value: number;
}

export interface GraphEdge {
  from: string;
  to: string;
  label: string;
  color: string;
}

export interface TraceRecord {
  trace_id: string;
  parent_trace_id: string | null;
  agent: string;
  prompt_template: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  model: string;
  market_id: string;
  evaluation_scores: Record<string, number>;
  cost_usd: number;
  created_at: string;
}

export interface CostSummary {
  total_traces: number;
  total_cost_usd: number;
  avg_cost_usd: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  avg_latency_ms: number;
}

export interface SSEEvent {
  id: string;
  seq: number;
  type: string;
  timestamp: string;
  agent: string | null;
  payload: Record<string, unknown>;
}

// --- API calls ---

export const api = {
  getStatus: () => get<SystemStatus>("/agents/status"),
  getEvalStats: () => get<EvalStats>("/evaluation/stats"),
  getGraphSnapshot: () => get<GraphSnapshot>("/knowledge/graph-snapshot"),
  searchGraph: (q: string) => get<GraphSnapshot>(`/knowledge/search?q=${encodeURIComponent(q)}`),
  getTraces: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return get<{ traces: TraceRecord[]; count: number }>(`/observability/traces${qs}`);
  },
  getCosts: () => get<CostSummary>("/observability/costs"),
  getMetricsReport: () => get<Record<string, unknown>>("/reports/metrics"),
};
