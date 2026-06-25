import { useState } from "react";
import type { DisparityRow, FairnessRow, FeatureImportance, Metrics } from "../types";

interface Props {
  metrics: Metrics | null;
  importance: FeatureImportance[];
  fairness: { audit: FairnessRow[]; disparities: DisparityRow[] } | null;
}

const PRETTY = (name: string) =>
  name
    .replace(/^.*__/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

export default function ModelInsights({ metrics, importance, fairness }: Props) {
  const [tab, setTab] = useState<"perf" | "drivers" | "fairness">("perf");

  return (
    <div className="card">
      <h2>Model intelligence</h2>
      <p className="sub">How the model performs, what drives it, and who it works for.</p>
      <div className="tabs">
        <button className={`tab ${tab === "perf" ? "active" : ""}`} onClick={() => setTab("perf")}>
          Performance
        </button>
        <button
          className={`tab ${tab === "drivers" ? "active" : ""}`}
          onClick={() => setTab("drivers")}
        >
          Global drivers
        </button>
        <button
          className={`tab ${tab === "fairness" ? "active" : ""}`}
          onClick={() => setTab("fairness")}
        >
          Fairness audit
        </button>
      </div>

      {tab === "perf" &&
        (metrics ? (
          <div className="fade-in">
            <div className="kpis">
              <div className="kpi">
                <div className="v">{metrics.validation.roc_auc.toFixed(3)}</div>
                <div className="k">ROC-AUC</div>
              </div>
              <div className="kpi">
                <div className="v">{metrics.validation.pr_auc.toFixed(3)}</div>
                <div className="k">PR-AUC</div>
              </div>
              <div className="kpi">
                <div className="v">{(metrics.validation.recall * 100).toFixed(0)}%</div>
                <div className="k">Recall (strokes caught)</div>
              </div>
              <div className="kpi">
                <div className="v">{metrics.validation.brier.toFixed(3)}</div>
                <div className="k">Brier (calibration)</div>
              </div>
            </div>
            <p className="gauge-label" style={{ marginTop: 14 }}>
              {metrics.model} · trained on {metrics.class_balance.n_total.toLocaleString()} patients
              at a {(metrics.class_balance.positive_rate * 100).toFixed(1)}% stroke rate (
              {metrics.class_balance.imbalance_ratio.toFixed(0)}:1 imbalance). Confusion @ threshold:
              TP {metrics.validation.tp} · FN {metrics.validation.fn} · FP {metrics.validation.fp} ·
              TN {metrics.validation.tn}.
            </p>
          </div>
        ) : (
          <p className="empty">Train a model to populate metrics.</p>
        ))}

      {tab === "drivers" &&
        (importance.length ? (
          <div className="fade-in">
            {(() => {
              const max = Math.max(...importance.map((f) => f.importance)) || 1;
              return importance.map((f) => (
                <div
                  key={f.feature}
                  style={{ display: "grid", gridTemplateColumns: "170px 1fr", gap: 10, alignItems: "center", marginBottom: 9 }}
                >
                  <span className="contrib-name" title={PRETTY(f.feature)}>
                    {PRETTY(f.feature)}
                  </span>
                  <div className="bar-mini" style={{ width: `${(f.importance / max) * 100}%` }} />
                </div>
              ));
            })()}
            <p className="gauge-label" style={{ marginTop: 6 }}>
              Mean |SHAP| across training data — the strongest population-level drivers.
            </p>
          </div>
        ) : (
          <p className="empty">Global importance unavailable.</p>
        ))}

      {tab === "fairness" &&
        (fairness && fairness.disparities.length ? (
          <div className="fade-in">
            <table>
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th>Recall gap</th>
                  <th>Selection gap</th>
                  <th>Weakest group</th>
                </tr>
              </thead>
              <tbody>
                {fairness.disparities.map((d) => (
                  <tr key={d.attribute}>
                    <td>{PRETTY(d.attribute)}</td>
                    <td className="num">{(d.recall_gap * 100).toFixed(0)}%</td>
                    <td className="num">{(d.selection_rate_gap * 100).toFixed(0)}%</td>
                    <td>{d.worst_recall_group}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="gauge-label" style={{ marginTop: 10 }}>
              Large gaps in small subgroups (e.g. pediatric) reflect very few positive cases and are
              flagged, not acted on automatically.
            </p>
          </div>
        ) : (
          <p className="empty">Fairness report unavailable.</p>
        ))}
    </div>
  );
}
