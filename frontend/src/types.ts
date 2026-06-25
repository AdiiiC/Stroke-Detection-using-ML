export interface Patient {
  gender: "Male" | "Female" | "Other";
  age: number;
  hypertension: 0 | 1;
  heart_disease: 0 | 1;
  ever_married: "Yes" | "No";
  work_type: "Private" | "Self-employed" | "Govt_job" | "children" | "Never_worked";
  Residence_type: "Urban" | "Rural";
  avg_glucose_level: number;
  bmi: number | null;
  smoking_status: "never smoked" | "formerly smoked" | "smokes" | "Unknown";
}

export interface Contribution {
  feature: string;
  contribution: number;
  direction: string;
}

export interface PredictionResponse {
  stroke_probability: number;
  predicted_stroke: number;
  risk_band: string;
  threshold: number;
  explanation: Contribution[];
}

export interface WhatIfLever {
  label: string;
  new_probability: number;
  delta: number;
  relative_reduction: number;
}

export interface WhatIfResponse {
  base_probability: number;
  risk_band: string;
  levers: WhatIfLever[];
}

export interface Metrics {
  model: string;
  resampling: string;
  class_balance: {
    n_total: number;
    n_positive: number;
    n_negative: number;
    positive_rate: number;
    imbalance_ratio: number;
  };
  cv_roc_auc_mean: number;
  cv_pr_auc_mean: number;
  validation: {
    roc_auc: number;
    pr_auc: number;
    recall: number;
    precision: number;
    brier: number;
    threshold: number;
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  };
}

export interface FairnessRow {
  attribute: string;
  group: string;
  n: number;
  positive_rate: number;
  selection_rate: number;
  recall: number;
  auc: number | null;
}

export interface DisparityRow {
  attribute: string;
  recall_gap: number;
  selection_rate_gap: number;
  auc_gap: number;
  worst_recall_group: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface SampleSet {
  [key: string]: { label: string; patient: Patient };
}
