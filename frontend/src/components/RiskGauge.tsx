interface Props {
  probability: number;
  threshold: number;
  band: string;
}

const BAND_COLOR: Record<string, string> = {
  low: "#34d399",
  moderate: "#fbbf24",
  elevated: "#fb923c",
  high: "#f87171",
};

export default function RiskGauge({ probability, threshold, band }: Props) {
  const size = 230;
  const stroke = 18;
  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  // 270-degree arc starting at 135deg.
  const startAngle = 135;
  const sweep = 270;
  const circumference = 2 * Math.PI * r;
  const arcLen = (sweep / 360) * circumference;
  const pct = Math.min(1, Math.max(0, probability));
  const color = BAND_COLOR[band] ?? "#6d8bff";

  const polar = (angle: number) => {
    const a = (angle * Math.PI) / 180;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  };
  const thAngle = startAngle + threshold * sweep;
  const thOuter = (() => {
    const a = (thAngle * Math.PI) / 180;
    const ro = r + stroke / 2 + 3;
    const ri = r - stroke / 2 - 3;
    return {
      x1: cx + ri * Math.cos(a),
      y1: cy + ri * Math.sin(a),
      x2: cx + ro * Math.cos(a),
      y2: cy + ro * Math.sin(a),
    };
  })();

  const start = polar(startAngle);

  return (
    <div className="gauge-wrap">
      <div className="gauge">
        <svg width={size} height={size} style={{ transform: "rotate(0deg)" }}>
          <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={color} stopOpacity="0.9" />
              <stop offset="100%" stopColor={color} stopOpacity="0.55" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="4" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* track */}
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${arcLen} ${circumference}`}
            transform={`rotate(${startAngle} ${cx} ${cy})`}
          />
          {/* value */}
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth={stroke}
            strokeLinecap="round"
            filter="url(#glow)"
            strokeDasharray={`${arcLen * pct} ${circumference}`}
            transform={`rotate(${startAngle} ${cx} ${cy})`}
            style={{ transition: "stroke-dasharray 0.9s cubic-bezier(0.2,0.8,0.2,1)" }}
          />
          {/* threshold tick */}
          <line
            x1={thOuter.x1}
            y1={thOuter.y1}
            x2={thOuter.x2}
            y2={thOuter.y2}
            stroke="#eaf0ff"
            strokeWidth={2}
            strokeLinecap="round"
            opacity={0.8}
          />
          <circle cx={start.x} cy={start.y} r={2.5} fill="rgba(255,255,255,0.3)" />
        </svg>
        <div className="gauge-center">
          <div className="gauge-pct" style={{ color }}>
            {(pct * 100).toFixed(1)}%
          </div>
          <div className="gauge-label">stroke probability</div>
        </div>
      </div>
      <span className={`band ${band}`}>● {band.toUpperCase()} RISK</span>
      <div className="gauge-label" style={{ marginTop: 2 }}>
        decision threshold {(threshold * 100).toFixed(1)}%
      </div>
    </div>
  );
}
