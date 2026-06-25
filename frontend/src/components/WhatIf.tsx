import type { WhatIfResponse } from "../types";

interface Props {
  data: WhatIfResponse | null;
}

export default function WhatIf({ data }: Props) {
  if (!data) {
    return <p className="empty">Assess a patient to see actionable risk-reduction levers.</p>;
  }
  if (!data.levers.length) {
    return (
      <p className="empty">
        No modifiable risk factors detected for this profile — the score is driven by
        non-modifiable factors (e.g. age).
      </p>
    );
  }
  const maxRel = Math.max(...data.levers.map((l) => Math.abs(l.relative_reduction))) || 1;
  return (
    <div>
      {data.levers.map((l) => {
        const good = l.delta < 0;
        const w = (Math.abs(l.relative_reduction) / maxRel) * 100;
        return (
          <div className="lever" key={l.label}>
            <div>
              <div className="lever-name">{l.label}</div>
              <div className="lever-bar">
                <span style={{ width: `${w}%` }} />
              </div>
            </div>
            <div className={`lever-delta ${good ? "good" : "bad"}`}>
              {good ? "−" : "+"}
              {Math.abs(l.relative_reduction * 100).toFixed(0)}%
            </div>
          </div>
        );
      })}
      <p className="gauge-label" style={{ marginTop: 4 }}>
        Estimated relative change in modelled risk if each factor were addressed. Illustrative,
        not medical advice.
      </p>
    </div>
  );
}
