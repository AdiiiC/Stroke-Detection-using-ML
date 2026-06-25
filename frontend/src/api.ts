import type {
  DisparityRow,
  FairnessRow,
  FeatureImportance,
  Metrics,
  Patient,
  PredictionResponse,
  SampleSet,
  WhatIfResponse,
} from "./types";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
// In production, set VITE_API_URL to the API origin at build time.
const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "/api";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<{ status: string; model_loaded: boolean }>("/health"),
  metrics: () => get<Metrics>("/metrics"),
  fairness: () => get<{ audit: FairnessRow[]; disparities: DisparityRow[] }>("/fairness"),
  featureImportance: () => get<{ features: FeatureImportance[] }>("/feature-importance"),
  samples: () => get<SampleSet>("/samples"),
  predict: (p: Patient) => post<PredictionResponse>("/predict?explain=true", p),
  predictBatch: (patients: Patient[]) =>
    post<{ predictions: PredictionResponse[] }>("/predict/batch", patients),
  whatif: (p: Patient) => post<WhatIfResponse>("/whatif", p),
};
