import { useEffect, useState } from "react";
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
  Legend,
  Cell,
} from "recharts";
import { 
  Activity, 
  BarChart3, 
  RefreshCw, 
  TrendingUp,
  AlertCircle,
  Brain,
  Target
} from "lucide-react";

export default function Graph() {
  const [binaryPoints, setBinaryPoints] = useState([]);
  const [multiclassData, setMulticlassData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    totalBinary: 0,
    totalMulticlass: 0,
    avgConfidence: 0,
  });

  const classColors = {
    CN: "#10b981",  // emerald
    MCI: "#f59e0b", // amber
    AD: "#ef4444",  // red
  };

  const loadData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const res = await api.get("/stats/history");
      const history = res.data.history || [];

      // Process binary predictions
      const binaryPredictions = history
        .filter(h => h.task === "binary")
        .map(h => ({
          x: h.index,
          y: h.confidence || 0,
          label: `#${h.index}`,
        }));

      // Process multiclass predictions
      const counts = { CN: 0, MCI: 0, AD: 0 };
      const multiclassPredictions = history.filter(
        h => h.task === "multiclass" && h.probs
      );

      multiclassPredictions.forEach(h => {
        counts.CN += h.probs.CN || 0;
        counts.MCI += h.probs.MCI || 0;
        counts.AD += h.probs.AD || 0;
      });

      const multiData = Object.keys(counts).map(k => ({
        class: k,
        count: counts[k].toFixed(2),
        percentage: multiclassPredictions.length > 0 
          ? ((counts[k] / multiclassPredictions.length) * 100).toFixed(1)
          : 0,
      }));

      // Calculate stats
      const avgConf = binaryPredictions.length > 0
        ? (binaryPredictions.reduce((sum, p) => sum + p.y, 0) / binaryPredictions.length).toFixed(3)
        : 0;

      setBinaryPoints(binaryPredictions);
      setMulticlassData(multiData);
      setStats({
        totalBinary: binaryPredictions.length,
        totalMulticlass: multiclassPredictions.length,
        avgConfidence: avgConf,
      });
    } catch (err) {
      console.error("Failed to load graph data:", err);
      setError("Failed to load prediction data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    
    // Auto-refresh every 10 seconds
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  const CustomTooltip = ({ active, payload, type }) => {
    if (!active || !payload || !payload.length) return null;

    if (type === "binary") {
      return (
        <div className="bg-base-100 border border-base-300 rounded-lg shadow-lg p-3">
          <p className="text-sm font-semibold">Prediction #{payload[0].payload.x}</p>
          <p className="text-sm text-primary">
            Confidence: <span className="font-bold">{(payload[0].value * 100).toFixed(1)}%</span>
          </p>
        </div>
      );
    }

    if (type === "multiclass") {
      const data = payload[0].payload;
      return (
        <div className="bg-base-100 border border-base-300 rounded-lg shadow-lg p-3">
          <p className="text-sm font-semibold">{data.class}</p>
          <p className="text-sm">
            Total: <span className="font-bold">{data.count}</span>
          </p>
          <p className="text-sm text-accent">
            Avg: <span className="font-bold">{data.percentage}%</span>
          </p>
        </div>
      );
    }

    return null;
  };

  if (loading && binaryPoints.length === 0 && multiclassData.length === 0) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {[1, 2].map(i => (
          <div key={i} className="card bg-base-100 shadow-xl">
            <div className="card-body items-center justify-center h-96">
              <span className="loading loading-spinner loading-lg text-primary"></span>
              <p className="text-sm text-base-content/60 mt-4">Loading predictions...</p>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-error shadow-lg">
        <AlertCircle className="w-5 h-5" />
        <span>{error}</span>
        <button onClick={loadData} className="btn btn-sm btn-ghost">
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="stat bg-base-100 shadow-xl rounded-box">
          <div className="stat-figure text-primary">
            <Target className="w-8 h-8" />
          </div>
          <div className="stat-title">Binary Predictions</div>
          <div className="stat-value text-primary">{stats.totalBinary}</div>
          <div className="stat-desc">Tumor detection scans</div>
        </div>

        <div className="stat bg-base-100 shadow-xl rounded-box">
          <div className="stat-figure text-secondary">
            <Brain className="w-8 h-8" />
          </div>
          <div className="stat-title">Multiclass Predictions</div>
          <div className="stat-value text-secondary">{stats.totalMulticlass}</div>
          <div className="stat-desc">Alzheimer's classifications</div>
        </div>

        <div className="stat bg-base-100 shadow-xl rounded-box">
          <div className="stat-figure text-accent">
            <TrendingUp className="w-8 h-8" />
          </div>
          <div className="stat-title">Avg Confidence</div>
          <div className="stat-value text-accent">
            {stats.totalBinary > 0 ? `${(stats.avgConfidence * 100).toFixed(1)}%` : "N/A"}
          </div>
          <div className="stat-desc">Binary predictions</div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Binary Scatter Chart */}
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body">
            <div className="flex items-center justify-between mb-4">
              <h2 className="card-title flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                Binary Prediction Confidence
              </h2>
              <div className="badge badge-primary badge-outline">
                {stats.totalBinary} scans
              </div>
            </div>

            {binaryPoints.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-80 text-base-content/60">
                <Activity className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-sm">No binary predictions yet</p>
                <p className="text-xs mt-2">Upload a scan to see confidence scores</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-base-300" />
                  <XAxis 
                    dataKey="x" 
                    name="Prediction #" 
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    allowDecimals={false}
                    label={{ value: 'Prediction Number', position: 'bottom', offset: 0 }}
                    tick={{ fill: 'currentColor' }}
                  />
                  <YAxis 
                    domain={[0, 1]} 
                    name="Confidence"
                    label={{ value: 'Confidence Score', angle: -90, position: 'insideLeft' }}
                    tick={{ fill: 'currentColor' }}
                    tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                  />
                  <Tooltip content={<CustomTooltip type="binary" />} />
                  <Scatter 
                    data={binaryPoints} 
                    fill="hsl(var(--p))" 
                    fillOpacity={0.7}
                    strokeWidth={2}
                    stroke="hsl(var(--p))"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            )}

            <div className="card-actions justify-end mt-4">
              <button 
                onClick={loadData} 
                className="btn btn-sm btn-ghost gap-2"
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Multiclass Histogram */}
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body">
            <div className="flex items-center justify-between mb-4">
              <h2 className="card-title flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-secondary" />
                Multiclass Distribution
              </h2>
              <div className="badge badge-secondary badge-outline">
                {stats.totalMulticlass} scans
              </div>
            </div>

            {multiclassData.length === 0 || multiclassData.every(d => parseFloat(d.count) === 0) ? (
              <div className="flex flex-col items-center justify-center h-80 text-base-content/60">
                <BarChart3 className="w-16 h-16 mb-4 opacity-30" />
                <p className="text-sm">No multiclass predictions yet</p>
                <p className="text-xs mt-2">Upload a scan to see class distribution</p>
              </div>
            ) : (
              <>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart 
                    data={multiclassData}
                    margin={{ top: 10, right: 20, bottom: 40, left: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" className="stroke-base-300" />
                    <XAxis 
                      dataKey="class" 
                      label={{ value: 'Class', position: 'bottom', offset: 0 }}
                      tick={{ fill: 'currentColor' }}
                    />
                    <YAxis 
                      label={{ value: 'Cumulative Probability', angle: -90, position: 'insideLeft' }}
                      tick={{ fill: 'currentColor' }}
                    />
                    <Tooltip content={<CustomTooltip type="multiclass" />} />
                    <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                      {multiclassData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={classColors[entry.class]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                {/* Legend */}
                <div className="grid grid-cols-3 gap-2 mt-4">
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-success/10">
                    <div className="w-3 h-3 rounded-full bg-success"></div>
                    <div>
                      <p className="text-xs font-semibold">CN</p>
                      <p className="text-xs text-base-content/60">Cognitively Normal</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-warning/10">
                    <div className="w-3 h-3 rounded-full bg-warning"></div>
                    <div>
                      <p className="text-xs font-semibold">MCI</p>
                      <p className="text-xs text-base-content/60">Mild Impairment</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 p-2 rounded-lg bg-error/10">
                    <div className="w-3 h-3 rounded-full bg-error"></div>
                    <div>
                      <p className="text-xs font-semibold">AD</p>
                      <p className="text-xs text-base-content/60">Alzheimer's</p>
                    </div>
                  </div>
                </div>
              </>
            )}

            <div className="card-actions justify-end mt-4">
              <button 
                onClick={loadData} 
                className="btn btn-sm btn-ghost gap-2"
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}