import type { Patient, SampleSet } from "../types";

interface Props {
  patient: Patient;
  setPatient: (p: Patient) => void;
  samples: SampleSet | null;
  onSubmit: () => void;
  loading: boolean;
}

const WORK_TYPES: Patient["work_type"][] = [
  "Private",
  "Self-employed",
  "Govt_job",
  "children",
  "Never_worked",
];
const SMOKING: Patient["smoking_status"][] = [
  "never smoked",
  "formerly smoked",
  "smokes",
  "Unknown",
];

export default function PatientForm({ patient, setPatient, samples, onSubmit, loading }: Props) {
  const set = <K extends keyof Patient>(key: K, value: Patient[K]) =>
    setPatient({ ...patient, [key]: value });

  return (
    <div className="card">
      <h2>Patient profile</h2>
      <p className="sub">Enter risk factors or load a preset to assess stroke risk.</p>

      {samples && (
        <div className="preset-row">
          {Object.entries(samples).map(([key, s]) => (
            <button key={key} className="chip" onClick={() => setPatient(s.patient)}>
              {s.label}
            </button>
          ))}
        </div>
      )}

      <div className="section-label">Demographics</div>
      <div className="field-grid">
        <div className="field">
          <label>Age — {patient.age} yrs</label>
          <div className="range-row">
            <input
              type="range"
              min={1}
              max={100}
              value={patient.age}
              onChange={(e) => set("age", Number(e.target.value))}
            />
            <span className="range-val">{patient.age}</span>
          </div>
        </div>
        <div className="field">
          <label>Gender</label>
          <select value={patient.gender} onChange={(e) => set("gender", e.target.value as Patient["gender"])}>
            <option>Male</option>
            <option>Female</option>
            <option>Other</option>
          </select>
        </div>
        <div className="field">
          <label>Ever married</label>
          <div className="toggle-row">
            {(["Yes", "No"] as const).map((v) => (
              <div
                key={v}
                className={`toggle ${patient.ever_married === v ? "active" : ""}`}
                onClick={() => set("ever_married", v)}
              >
                {v}
              </div>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Residence</label>
          <div className="toggle-row">
            {(["Urban", "Rural"] as const).map((v) => (
              <div
                key={v}
                className={`toggle ${patient.Residence_type === v ? "active" : ""}`}
                onClick={() => set("Residence_type", v)}
              >
                {v}
              </div>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Work type</label>
          <select
            value={patient.work_type}
            onChange={(e) => set("work_type", e.target.value as Patient["work_type"])}
          >
            {WORK_TYPES.map((w) => (
              <option key={w}>{w}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Smoking status</label>
          <select
            value={patient.smoking_status}
            onChange={(e) => set("smoking_status", e.target.value as Patient["smoking_status"])}
          >
            {SMOKING.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="section-label">Clinical markers</div>
      <div className="field-grid">
        <div className="field">
          <label>Avg glucose — {patient.avg_glucose_level.toFixed(0)} mg/dL</label>
          <div className="range-row">
            <input
              type="range"
              min={55}
              max={300}
              step={1}
              value={patient.avg_glucose_level}
              onChange={(e) => set("avg_glucose_level", Number(e.target.value))}
            />
            <span className="range-val">{patient.avg_glucose_level.toFixed(0)}</span>
          </div>
        </div>
        <div className="field">
          <label>BMI — {patient.bmi?.toFixed(1) ?? "—"}</label>
          <div className="range-row">
            <input
              type="range"
              min={12}
              max={60}
              step={0.5}
              value={patient.bmi ?? 26}
              onChange={(e) => set("bmi", Number(e.target.value))}
            />
            <span className="range-val">{patient.bmi?.toFixed(1) ?? "—"}</span>
          </div>
        </div>
        <div className="field">
          <label>Hypertension</label>
          <div className="toggle-row">
            {([0, 1] as const).map((v) => (
              <div
                key={v}
                className={`toggle ${patient.hypertension === v ? "active" : ""}`}
                onClick={() => set("hypertension", v)}
              >
                {v === 1 ? "Yes" : "No"}
              </div>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Heart disease</label>
          <div className="toggle-row">
            {([0, 1] as const).map((v) => (
              <div
                key={v}
                className={`toggle ${patient.heart_disease === v ? "active" : ""}`}
                onClick={() => set("heart_disease", v)}
              >
                {v === 1 ? "Yes" : "No"}
              </div>
            ))}
          </div>
        </div>
      </div>

      <button className="btn" onClick={onSubmit} disabled={loading}>
        {loading ? <span className="spin" /> : null}
        {loading ? "Assessing…" : "Assess stroke risk"}
      </button>
    </div>
  );
}
