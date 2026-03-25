import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "../services/api";
import {
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import {
  Activity,
  BarChart3,
  RefreshCw,
  TrendingUp,
  AlertCircle,
  Brain,
  Target,
  Zap,
} from "lucide-react";

// ─── animation variants ───────────────────────────────────────────────────────
const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, delay: i * 0.1, ease: [0.22, 1, 0.36, 1] },
  }),
};

const cardVariant = {
  hidden: { opacity: 0, y: 32, scale: 0.97 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.5, delay: i * 0.12, ease: [0.22, 1, 0.36, 1] },
  }),
};

// ─── class config ─────────────────────────────────────────────────────────────
const CLASS_CONFIG = {
  CN:  { color: "#10b981", bg: "rgba(16,185,129,0.12)", label: "Cognitively Normal", badge: "badge-success" },
  MCI: { color: "#f59e0b", bg: "rgba(245,158,11,0.12)",  label: "Mild Impairment",    badge: "badge-warning" },
  AD:  { color: "#ef4444", bg: "rgba(239,68,68,0.12)",   label: "Alzheimer's",        badge: "badge-error"   },
};

// ─── animated stat number ─────────────────────────────────────────────────────
function AnimatedNumber({ value, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const target = parseFloat(value) || 0;
    let start = 0;
    const steps = 40;
    const increment = target / steps;
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) { setDisplay(target); clearInterval(timer); }
      else setDisplay(start);
    }, 20);
    return () => clearInterval(timer);
  }, [value]);
  return <span>{typeof value === "string" && value.includes(".") ? display.toFixed(1) : Math.round(display)}{suffix}</span>;
}

// ─── custom scatter dot with pulse on high confidence ────────────────────────
const HIGH_COLOR = "#6366f1"; // indigo - primary-ish
const NORMAL_COLOR = "#a78bfa"; // violet - secondary-ish

function ConfidenceDot(props) {
  const { cx, cy, payload } = props;
  if (cx === undefined || cy === undefined) return null;
  const isHigh = payload.y >= 0.85;
  const color = isHigh ? HIGH_COLOR : NORMAL_COLOR;
  return (
    <g>
      {isHigh && (
        <circle cx={cx} cy={cy} r={8} fill={color} opacity={0.2}>
          <animate attributeName="r" values="8;16;8" dur="2s" repeatCount="indefinite" />
          <animate attributeName="opacity" values="0.2;0.03;0.2" dur="2s" repeatCount="indefinite" />
        </circle>
      )}
      <circle
        cx={cx}
        cy={cy}
        r={7}
        fill={color}
        fillOpacity={0.25}
        stroke={color}
        strokeWidth={2}
      />
      <circle
        cx={cx}
        cy={cy}
        r={3.5}
        fill={color}
        fillOpacity={1}
      />
    </g>
  );
}

// ─── tooltips ─────────────────────────────────────────────────────────────────
function BinaryTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { x, y } = payload[0].payload;
  const pct = (y * 100).toFixed(1);
  const isHigh = y >= 0.85;
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-base-100 border border-base-300 rounded-xl shadow-2xl p-3 text-sm"
    >
      <p className="font-bold text-base-content mb-1">Scan #{x}</p>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full" style={{ background: isHigh ? HIGH_COLOR : NORMAL_COLOR }} />
        <span className="text-base-content/70">Confidence</span>
        <span className="font-bold ml-auto pl-4" style={{ color: isHigh ? HIGH_COLOR : NORMAL_COLOR }}>
          {pct}%
        </span>
      </div>
      {isHigh && (
        <p className="text-xs mt-1 flex items-center gap-1" style={{ color: HIGH_COLOR, opacity: 0.8 }}>
          <Zap className="w-3 h-3" /> High confidence
        </p>
      )}
    </motion.div>
  );
}

function MultiTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const cfg = CLASS_CONFIG[d.class];
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-base-100 border border-base-300 rounded-xl shadow-2xl p-3 text-sm"
    >
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full" style={{ background: cfg.color }} />
        <p className="font-bold">{d.class}</p>
        <span className="text-base-content/50 text-xs">— {cfg.label}</span>
      </div>
      <div className="flex justify-between gap-6">
        <span className="text-base-content/70">Cumulative</span>
        <span className="font-bold" style={{ color: cfg.color }}>{d.count}</span>
      </div>
      <div className="flex justify-between gap-6">
        <span className="text-base-content/70">Average</span>
        <span className="font-bold" style={{ color: cfg.color }}>{d.percentage}%</span>
      </div>
    </motion.div>
  );
}

// ─── animated bar shape ───────────────────────────────────────────────────────
function AnimatedBar(props) {
  const { x, y, width, height, fill } = props;
  if (!width || !height) return null;
  return (
    <g>
      {/* glow shadow */}
      <rect
        x={x + width * 0.1}
        y={y + 4}
        width={width * 0.8}
        height={height}
        fill={fill}
        rx={8}
        opacity={0.2}
        style={{ filter: "blur(6px)" }}
      />
      {/* main bar */}
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill={fill}
        rx={8}
        style={{
          transformOrigin: `${x + width / 2}px ${y + height}px`,
          animation: "barGrow 0.6s cubic-bezier(0.22,1,0.36,1) both",
        }}
      />
    </g>
  );
}

// ─── main component ───────────────────────────────────────────────────────────
export default function Graph() {
  const [binaryPoints, setBinaryPoints]   = useState([]);
  const [multiclassData, setMulticlassData] = useState([]);
  const [loading, setLoading]             = useState(true);
  const [refreshing, setRefreshing]       = useState(false);
  const [error, setError]                 = useState(null);
  const [stats, setStats]                 = useState({ totalBinary: 0, totalMulticlass: 0, avgConfidence: 0 });

  const loadData = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const res = await api.get("/stats/history");
      const history = res.data.history || [];

      const binaryPredictions = history
        .filter(h => h.task === "binary")
        .map((h, i) => ({ x: i + 1, y: h.confidence !== undefined ? Number(h.confidence) : null }))
        .filter(p => p.y !== null);

      const counts = { CN: 0, MCI: 0, AD: 0 };
      const multiclassPredictions = history.filter(h => h.task === "multiclass" && h.probs);
      multiclassPredictions.forEach(h => {
        counts.CN  += h.probs.CN  || 0;
        counts.MCI += h.probs.MCI || 0;
        counts.AD  += h.probs.AD  || 0;
      });

      const multiData = Object.keys(counts).map(k => ({
        class: k,
        count: counts[k].toFixed(2),
        percentage: multiclassPredictions.length > 0
          ? ((counts[k] / multiclassPredictions.length) * 100).toFixed(1)
          : 0,
      }));

      const avgConf = binaryPredictions.length > 0
        ? (binaryPredictions.reduce((s, p) => s + p.y, 0) / binaryPredictions.length).toFixed(3)
        : 0;

      setBinaryPoints(binaryPredictions);
      setMulticlassData(multiData);
      setStats({ totalBinary: binaryPredictions.length, totalMulticlass: multiclassPredictions.length, avgConfidence: avgConf });
    } catch (err) {
      console.error("Failed to load graph data:", err);
      setError("Failed to load prediction data");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(() => loadData(true), 10000);
    return () => clearInterval(interval);
  }, []);

  // ── loading skeleton ────────────────────────────────────────────────────────
  if (loading && binaryPoints.length === 0 && multiclassData.length === 0) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="stat bg-base-100 shadow-xl rounded-box animate-pulse">
              <div className="h-4 w-24 bg-base-300 rounded mb-2" />
              <div className="h-8 w-16 bg-base-300 rounded mb-2" />
              <div className="h-3 w-32 bg-base-300 rounded" />
            </div>
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[1, 2].map(i => (
            <div key={i} className="card bg-base-100 shadow-xl animate-pulse">
              <div className="card-body h-96">
                <div className="h-5 w-48 bg-base-300 rounded mb-4" />
                <div className="flex-1 bg-base-300/50 rounded-xl" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── error state ─────────────────────────────────────────────────────────────
  if (error) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
        className="alert alert-error shadow-lg"
      >
        <AlertCircle className="w-5 h-5" />
        <span>{error}</span>
        <button onClick={() => loadData()} className="btn btn-sm btn-ghost">
          <RefreshCw className="w-4 h-4" /> Retry
        </button>
      </motion.div>
    );
  }

  const avgPct = stats.avgConfidence > 0 ? (stats.avgConfidence * 100).toFixed(1) : null;

  return (
    <div className="space-y-6">
      <style>{`
        @keyframes barGrow {
          from { transform: scaleY(0); }
          to   { transform: scaleY(1); }
        }
      `}</style>

      {/* ── Stat Cards ────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            icon: <Target className="w-8 h-8" />,
            color: "text-primary",
            title: "Binary Predictions",
            value: stats.totalBinary,
            desc: "Tumor detection scans",
            i: 0,
          },
          {
            icon: <Brain className="w-8 h-8" />,
            color: "text-secondary",
            title: "Multiclass Predictions",
            value: stats.totalMulticlass,
            desc: "Alzheimer's classifications",
            i: 1,
          },
          {
            icon: <TrendingUp className="w-8 h-8" />,
            color: "text-accent",
            title: "Avg Confidence",
            value: avgPct ?? "N/A",
            suffix: avgPct ? "%" : "",
            desc: "Binary predictions",
            i: 2,
          },
        ].map(({ icon, color, title, value, suffix = "", desc, i }) => (
          <motion.div
            key={title}
            custom={i}
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            className="stat bg-base-100 shadow-xl rounded-box relative overflow-hidden group"
          >
            {/* subtle glow bg on hover */}
            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none rounded-box`}
              style={{ background: `radial-gradient(ellipse at 80% 50%, ${color.includes("primary") ? "hsl(var(--p)/0.06)" : color.includes("secondary") ? "hsl(var(--s)/0.06)" : "hsl(var(--a)/0.06)"} 0%, transparent 70%)` }}
            />
            <div className={`stat-figure ${color}`}>{icon}</div>
            <div className="stat-title text-base-content/60">{title}</div>
            <div className={`stat-value ${color} tabular-nums`}>
              {value === "N/A" ? "N/A" : <AnimatedNumber value={value} suffix={suffix} />}
            </div>
            <div className="stat-desc">{desc}</div>
          </motion.div>
        ))}
      </div>

      {/* ── Charts ────────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Binary Scatter */}
        <motion.div
          custom={0}
          variants={cardVariant}
          initial="hidden"
          animate="visible"
          className="card bg-base-100 shadow-xl overflow-hidden"
        >
          {/* top accent bar */}
          <div className="h-1 w-full bg-gradient-to-r from-primary via-primary/50 to-transparent" />

          <div className="card-body gap-0">
            <div className="flex items-center justify-between mb-5">
              <h2 className="card-title flex items-center gap-2 text-base">
                <div className="p-1.5 rounded-lg bg-primary/10">
                  <Activity className="w-4 h-4 text-primary" />
                </div>
                Binary Prediction Confidence
              </h2>
              <div className="flex items-center gap-2">
                <AnimatePresence>
                  {refreshing && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="loading loading-ring loading-xs text-primary"
                    />
                  )}
                </AnimatePresence>
                <div className="badge badge-primary badge-outline text-xs">
                  {stats.totalBinary} scans
                </div>
              </div>
            </div>

            {binaryPoints.length === 0 ? (
              <EmptyState icon={<Activity className="w-14 h-14" />} title="No binary predictions yet" sub="Upload a scan to see confidence scores" />
            ) : (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                {/* confidence band legend */}
                <div className="flex items-center gap-4 mb-3 px-1">
                  <div className="flex items-center gap-1.5 text-xs text-base-content/50">
                    <div className="w-2 h-2 rounded-full" style={{ background: HIGH_COLOR }} />
                    High ≥ 85%
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-base-content/50">
                    <div className="w-2 h-2 rounded-full" style={{ background: NORMAL_COLOR }} />
                    Normal
                  </div>
                </div>

                <ResponsiveContainer width="100%" height={280}>
                  <ScatterChart margin={{ top: 10, right: 24, bottom: 28, left: 10 }}>
                    <defs>
                      <linearGradient id="gridGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="hsl(var(--p))" stopOpacity={0.05} />
                        <stop offset="100%" stopColor="hsl(var(--p))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="4 4" stroke="hsl(var(--b3))" strokeOpacity={0.5} />
                    {/* high-confidence zone */}
                    <ReferenceLine y={0.85} stroke={HIGH_COLOR} strokeDasharray="6 3" strokeOpacity={0.5}
                      label={{ value: "85%", position: "right", fill: HIGH_COLOR, fontSize: 10, opacity: 0.7 }}
                    />
                    <XAxis
                      dataKey="x"
                      name="x"
                      type="number"
                      domain={[0, "dataMax + 1"]}
                      allowDecimals={false}
                      label={{ value: "Prediction Number", position: "bottom", offset: 10, fill: "hsl(var(--bc))", opacity: 0.5, fontSize: 12 }}
                      tick={{ fill: "hsl(var(--bc))", opacity: 0.6, fontSize: 11 }}
                      axisLine={{ stroke: "hsl(var(--b3))" }}
                      tickLine={{ stroke: "hsl(var(--b3))" }}
                    />
                    <YAxis
                      dataKey="y"
                      name="Confidence"
                      domain={[0, 1]}
                      label={{ value: "Confidence Score", angle: -90, position: "insideLeft", fill: "hsl(var(--bc))", opacity: 0.5, fontSize: 12 }}
                      tick={{ fill: "hsl(var(--bc))", opacity: 0.6, fontSize: 11 }}
                      tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                      axisLine={{ stroke: "hsl(var(--b3))" }}
                      tickLine={{ stroke: "hsl(var(--b3))" }}
                    />
                    <Tooltip content={<BinaryTooltip />} cursor={{ strokeDasharray: "3 3", stroke: "hsl(var(--p))", strokeOpacity: 0.3 }} />
                    <Scatter
                      data={binaryPoints}
                      dataKey="y"
                      shape={<ConfidenceDot />}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </motion.div>
            )}

            <div className="card-actions justify-end mt-2">
              <button
                onClick={() => loadData(true)}
                className="btn btn-sm btn-ghost gap-1.5 text-xs"
                disabled={refreshing}
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>
          </div>
        </motion.div>

        {/* Multiclass Bar */}
        <motion.div
          custom={1}
          variants={cardVariant}
          initial="hidden"
          animate="visible"
          className="card bg-base-100 shadow-xl overflow-hidden"
        >
          <div className="h-1 w-full bg-gradient-to-r from-secondary via-secondary/50 to-transparent" />

          <div className="card-body gap-0">
            <div className="flex items-center justify-between mb-5">
              <h2 className="card-title flex items-center gap-2 text-base">
                <div className="p-1.5 rounded-lg bg-secondary/10">
                  <BarChart3 className="w-4 h-4 text-secondary" />
                </div>
                Multiclass Distribution
              </h2>
              <div className="flex items-center gap-2">
                <AnimatePresence>
                  {refreshing && (
                    <motion.span
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      className="loading loading-ring loading-xs text-secondary"
                    />
                  )}
                </AnimatePresence>
                <div className="badge badge-secondary badge-outline text-xs">
                  {stats.totalMulticlass} scans
                </div>
              </div>
            </div>

            {multiclassData.length === 0 || multiclassData.every(d => parseFloat(d.count) === 0) ? (
              <EmptyState icon={<BarChart3 className="w-14 h-14" />} title="No multiclass predictions yet" sub="Upload a scan to see class distribution" />
            ) : (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={multiclassData} margin={{ top: 10, right: 24, bottom: 28, left: 10 }} barSize={52}>
                    <CartesianGrid strokeDasharray="4 4" stroke="hsl(var(--b3))" strokeOpacity={0.5} vertical={false} />
                    <XAxis
                      dataKey="class"
                      label={{ value: "Class", position: "bottom", offset: 10, fill: "hsl(var(--bc))", opacity: 0.5, fontSize: 12 }}
                      tick={{ fill: "hsl(var(--bc))", opacity: 0.6, fontSize: 12, fontWeight: 600 }}
                      axisLine={{ stroke: "hsl(var(--b3))" }}
                      tickLine={false}
                    />
                    <YAxis
                      label={{ value: "Cumulative Probability", angle: -90, position: "insideLeft", fill: "hsl(var(--bc))", opacity: 0.5, fontSize: 12 }}
                      tick={{ fill: "hsl(var(--bc))", opacity: 0.6, fontSize: 11 }}
                      axisLine={{ stroke: "hsl(var(--b3))" }}
                      tickLine={{ stroke: "hsl(var(--b3))" }}
                    />
                    <Tooltip content={<MultiTooltip />} cursor={{ fill: "hsl(var(--b2))", radius: 8 }} />
                    <Bar dataKey="count" shape={<AnimatedBar />} radius={[8, 8, 0, 0]}>
                      {multiclassData.map((entry) => (
                        <Cell key={entry.class} fill={CLASS_CONFIG[entry.class]?.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {/* Class legend cards */}
                <div className="grid grid-cols-3 gap-2 mt-4">
                  {Object.entries(CLASS_CONFIG).map(([key, cfg], i) => {
                    const d = multiclassData.find(x => x.class === key);
                    return (
                      <motion.div
                        key={key}
                        custom={i}
                        variants={fadeUp}
                        initial="hidden"
                        animate="visible"
                        className="flex flex-col gap-1 p-2.5 rounded-xl border border-base-300/50 relative overflow-hidden group cursor-default"
                        style={{ background: cfg.bg }}
                        whileHover={{ scale: 1.03 }}
                        transition={{ type: "spring", stiffness: 400, damping: 20 }}
                      >
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: cfg.color }} />
                          <span className="text-xs font-bold" style={{ color: cfg.color }}>{key}</span>
                        </div>
                        <p className="text-xs text-base-content/50 leading-tight">{cfg.label}</p>
                        {d && (
                          <p className="text-xs font-semibold mt-0.5" style={{ color: cfg.color }}>
                            {d.percentage}% avg
                          </p>
                        )}
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            )}

            <div className="card-actions justify-end mt-2">
              <button
                onClick={() => loadData(true)}
                className="btn btn-sm btn-ghost gap-1.5 text-xs"
                disabled={refreshing}
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

// ─── shared empty state ───────────────────────────────────────────────────────
function EmptyState({ icon, title, sub }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="flex flex-col items-center justify-center h-72 text-base-content/40 gap-3"
    >
      <motion.div
        animate={{ y: [0, -6, 0] }}
        transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
        className="opacity-25"
      >
        {icon}
      </motion.div>
      <p className="text-sm font-medium">{title}</p>
      <p className="text-xs opacity-70">{sub}</p>
    </motion.div>
  );
}