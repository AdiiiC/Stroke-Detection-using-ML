import { useEffect, useState } from "react";
import { api } from "./api";
import BatchPanel from "./components/BatchPanel";
import Contributions from "./components/Contributions";
import ModelInsights from "./components/ModelInsights";
import PatientForm from "./components/PatientForm";
import RiskGauge from "./components/RiskGauge";
import WhatIf from "./components/WhatIf";
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

const DEFAULT_PATIENT: Patient = {
  gender: "Male",
  age: 67,
  hypertension: 1,
  heart_disease: 1,
  ever_married: "Yes",
  work_type: "Private",
  Residence_type: "Urban",
  avg_glucose_level: 169,
  bmi: 32.5,
  smoking_status: "formerly smoked",
};

export default function App() {
  const [patient, setPatient] = useState<Patient>(DEFAULT_PATIENT);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [whatif, setWhatif] = useState<WhatIfResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [theme, setTheme] = useState<"light" | "dark">(
    () => (localStorage.getItem("ng-theme") as "light" | "dark") || "light"
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ng-theme", theme);
  }, [theme]);

  const [online, setOnline] = useState<boolean | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [importance, setImportance] = useState<FeatureImportance[]>([]);
  const [fairness, setFairness] = useState<{
    audit: FairnessRow[];
    disparities: DisparityRow[];
  } | null>(null);
  const [samples, setSamples] = useState<SampleSet | null>(null);

  useEffect(() => {
    api
      .health()
      .then((h) => setOnline(h.model_loaded))
      .catch(() => setOnline(false));
    api.metrics().then(setMetrics).catch(() => {});
    api.featureImportance().then((d) => setImportance(d.features)).catch(() => {});
    api.fairness().then(setFairness).catch(() => {});
    api.samples().then(setSamples).catch(() => {});
  }, []);

  const assess = async () => {
    setLoading(true);
    setError(null);
    try {
      const [pred, wi] = await Promise.all([api.predict(patient), api.whatif(patient)]);
      setResult(pred);
      setWhatif(wi);
    } catch (e) {
      setError(
        e instanceof Error
          ? `Could not reach the model API (${e.message}). Is it running on :8000?`
          : "Assessment failed."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-mark">
            <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">
              <path
                d="M1 11h4l2-5 3 10 2.5-7 1.5 2h6"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <h1>NeuroGuard</h1>
            <p>Explainable, calibrated stroke-risk intelligence</p>
          </div>
        </div>
        <div className="header-right">
          <span className="pill">
            <span className={`dot ${online === null ? "" : online ? "live" : "down"}`} />
            {online === null ? "connecting…" : online ? "model online" : "API offline"}
          </span>
          <span className="pill">Research use only</span>
          <button
            className="theme-toggle"
            onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
            aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            title={theme === "light" ? "Dark mode" : "Light mode"}
          >
            {theme === "light" ? (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path
                  d="M13 9.2A5.2 5.2 0 016.8 3a5.2 5.2 0 106.2 6.2z"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinejoin="round"
                />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <circle cx="8" cy="8" r="3.1" stroke="currentColor" strokeWidth="1.4" />
                <path
                  d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3 3l1.1 1.1M11.9 11.9L13 13M13 3l-1.1 1.1M4.1 11.9L3 13"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                />
              </svg>
            )}
          </button>
        </div>
      </header>

      <div className="grid cols-2">
        <PatientForm
          patient={patient}
          setPatient={setPatient}
          samples={samples}
          onSubmit={assess}
          loading={loading}
        />

        <div className="card">
          <h2>Risk assessment</h2>
          <p className="sub">Calibrated probability with a recall-first decision threshold.</p>
          {error && <div className="err">{error}</div>}
          {result ? (
            <div className="fade-in">
              <RiskGauge
                probability={result.stroke_probability}
                threshold={result.threshold}
                band={result.risk_band}
              />
              <div className="decision">
                Clinical decision
                <strong>
                  {result.predicted_stroke ? "Flag for review" : "No flag"}
                </strong>
              </div>
            </div>
          ) : (
            <p className="empty">
              Fill in a patient profile and run an assessment to see the calibrated risk score.
            </p>
          )}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <h2>Why this prediction?</h2>
          <p className="sub">Per-patient SHAP attribution — transparent, auditable reasoning.</p>
          <Contributions items={result?.explanation ?? []} />
        </div>
        <div className="card">
          <h2>Actionable levers</h2>
          <p className="sub">What-if simulation over modifiable risk factors.</p>
          <WhatIf data={whatif} />
        </div>
      </div>

      <div className="grid cols-2">
        <ModelInsights metrics={metrics} importance={importance} fairness={fairness} />
        <BatchPanel />
      </div>

      <p className="disclaimer">
        NeuroGuard is a research and educational demonstration built on public competition data.
        It is <strong>not</strong> a medical device and must not be used for clinical decisions.
        Predictions capture correlation, not causation.
      </p>
    </div>
  );
}
