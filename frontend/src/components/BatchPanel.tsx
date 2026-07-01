import { useRef, useState } from "react";
import { api } from "../api";
import type { Patient, PredictionResponse } from "../types";

function parseCsv(text: string): Patient[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const header = lines[0].split(",").map((h) => h.trim());
  const idx = (name: string) => header.findIndex((h) => h.toLowerCase() === name.toLowerCase());
  const out: Patient[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    if (cols.length < header.length) continue;
    const g = (name: string) => cols[idx(name)]?.trim();
    const num = (name: string) => Number(g(name));
    out.push({
      gender: (g("gender") as Patient["gender"]) ?? "Male",
      age: num("age"),
      hypertension: (Number(g("hypertension")) ? 1 : 0) as 0 | 1,
      heart_disease: (Number(g("heart_disease")) ? 1 : 0) as 0 | 1,
      ever_married: (g("ever_married") as Patient["ever_married"]) ?? "Yes",
      work_type: (g("work_type") as Patient["work_type"]) ?? "Private",
      Residence_type: (g("Residence_type") as Patient["Residence_type"]) ?? "Urban",
      avg_glucose_level: num("avg_glucose_level"),
      bmi: g("bmi") && g("bmi") !== "N/A" ? num("bmi") : null,
      smoking_status: (g("smoking_status") as Patient["smoking_status"]) ?? "Unknown",
    });
  }
  return out;
}

const BAND_BG: Record<string, string> = {
  low: "var(--low-wash)",
  moderate: "var(--moderate-wash)",
  elevated: "var(--elevated-wash)",
  high: "var(--high-wash)",
};
const BAND_FG: Record<string, string> = {
  low: "var(--low)",
  moderate: "var(--moderate)",
  elevated: "var(--elevated)",
  high: "var(--high)",
};

export default function BatchPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [rows, setRows] = useState<PredictionResponse[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [count, setCount] = useState(0);

  const handleFile = async (file: File) => {
    setErr(null);
    setBusy(true);
    try {
      const text = await file.text();
      const patients = parseCsv(text).slice(0, 500);
      setCount(patients.length);
      if (!patients.length) throw new Error("No valid rows found in CSV.");
      const res = await api.predictBatch(patients);
      setRows(res.predictions);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Batch scoring failed.");
    } finally {
      setBusy(false);
    }
  };

  const flagged = rows.filter((r) => r.predicted_stroke).length;

  return (
    <div className="card">
      <h2>Cohort scoring</h2>
      <p className="sub">
        Upload a CSV of patients (Kaggle schema) to score an entire cohort and triage by risk.
      </p>
      <div className="drop" onClick={() => inputRef.current?.click()}>
        {busy ? (
          <>
            <span className="spin" /> Scoring {count} patients…
          </>
        ) : (
          <>
            <strong>Click to upload</strong> a patient CSV
            <div className="gauge-label" style={{ marginTop: 6 }}>
              columns: gender, age, hypertension, heart_disease, ever_married, work_type,
              Residence_type, avg_glucose_level, bmi, smoking_status
            </div>
          </>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        hidden
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      {err && <div className="err">{err}</div>}

      {rows.length > 0 && (
        <div className="fade-in" style={{ marginTop: 16 }}>
          <p className="gauge-label">
            {rows.length} scored · <strong style={{ color: "var(--high)" }}>{flagged} flagged</strong> for
            review
          </p>
          <div style={{ maxHeight: 320, overflow: "auto", marginTop: 8 }}>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th className="num">Probability</th>
                  <th>Band</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td className="num">{(r.stroke_probability * 100).toFixed(1)}%</td>
                    <td>
                      <span
                        className="risk-tag"
                        style={{
                          background: BAND_BG[r.risk_band],
                          color: BAND_FG[r.risk_band],
                        }}
                      >
                        {r.risk_band}
                      </span>
                    </td>
                    <td>{r.predicted_stroke ? "Flag" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
