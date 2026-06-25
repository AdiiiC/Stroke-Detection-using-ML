import type { Contribution } from "../types";

interface Props {
  items: Contribution[];
}

const PRETTY: Record<string, string> = {
  age: "Age",
  age_squared: "Age (non-linear)",
  age_glucose_interaction: "Age × glucose",
  avg_glucose_level: "Avg glucose",
  bmi: "BMI",
  comorbidity_count: "Comorbidities",
  is_elderly: "Elderly flag",
  is_pediatric: "Pediatric flag",
};

function prettify(name: string): string {
  const base = name.replace(/^.*__/, "");
  if (PRETTY[base]) return PRETTY[base];
  return base.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Contributions({ items }: Props) {
  if (!items.length) {
    return <p className="empty">Run an assessment to see what drives the prediction.</p>;
  }
  const max = Math.max(...items.map((i) => Math.abs(i.contribution))) || 1;
  return (
    <div className="contrib">
      {items.map((it) => {
        const up = it.contribution > 0;
        const w = (Math.abs(it.contribution) / max) * 50;
        return (
          <div className="contrib-row" key={it.feature}>
            <span className="contrib-name" title={prettify(it.feature)}>
              {prettify(it.feature)}
            </span>
            <div className="contrib-track">
              <div
                className={`contrib-fill ${up ? "up" : "down"}`}
                style={{ width: `${w}%` }}
              />
            </div>
            <span className="contrib-val" style={{ color: up ? "#fb923c" : "#34d399" }}>
              {up ? "+" : "−"}
              {Math.abs(it.contribution).toFixed(2)}
            </span>
          </div>
        );
      })}
      <p className="gauge-label" style={{ marginTop: 6 }}>
        SHAP log-odds contributions · orange raises risk, green lowers it
      </p>
    </div>
  );
}
