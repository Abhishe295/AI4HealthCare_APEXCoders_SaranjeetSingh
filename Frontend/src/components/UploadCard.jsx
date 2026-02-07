import { useState } from "react";
import { api } from "../services/api.js";
import { 
  Upload, 
  FileUp, 
  Loader2, 
  CheckCircle2, 
  XCircle,
  Brain,
  Activity,
  AlertCircle,
  BarChart3,
  TrendingUp,
  FileWarning
} from "lucide-react";

export default function UploadCard({ title, endpoint }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const isBinary = endpoint.includes("binary");

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;
    
    // Validate file type
    if (!selectedFile.name.endsWith('.npy')) {
      setError("Please upload a .npy file");
      setFile(null);
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await api.post(endpoint, formData);
      console.log("API Response:", res.data); // Debug logging
      setResult(res.data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.response?.data?.detail || "Prediction failed. Please try again.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  // Render result based on prediction type
  const renderResult = () => {
    if (!result) return null;

    if (isBinary) {
      // Binary prediction result
      const confidence = result.confidence || 0;
      const prediction = result.prediction || result.label || "Unknown";
      const hasTumor = prediction.toLowerCase().includes("tumor");
      
      console.log("Binary - Confidence:", confidence, "Prediction:", prediction); // Debug

      return (
        <div className="space-y-4 animate-in fade-in duration-500">
          <div className="divider my-2"></div>
          
          <div className={`alert ${hasTumor ? 'alert-error' : 'alert-success'} shadow-lg`}>
            {hasTumor ? (
              <AlertCircle className="w-6 h-6" />
            ) : (
              <CheckCircle2 className="w-6 h-6" />
            )}
            <div>
              <h3 className="font-bold text-lg">
                {hasTumor ? "Tumor Detected" : "No Tumor Detected"}
              </h3>
              <div className="text-sm opacity-90">
                Prediction: {prediction}
              </div>
            </div>
          </div>

          <div className="stats stats-vertical lg:stats-horizontal shadow w-full">
            <div className="stat">
              <div className="stat-figure text-primary">
                <Activity className="w-8 h-8" />
              </div>
              <div className="stat-title">Confidence Score</div>
              <div className="stat-value text-primary">
                {(confidence * 100).toFixed(1)}%
              </div>
              <div className="stat-desc">
                {confidence > 0.9 ? "Very high confidence" : 
                 confidence > 0.7 ? "High confidence" : 
                 confidence > 0.5 ? "Moderate confidence" : 
                 "Low confidence"}
              </div>
            </div>

            <div className="stat">
              <div className="stat-figure text-secondary">
                <TrendingUp className="w-8 h-8" />
              </div>
              <div className="stat-title">Prediction</div>
              <div className={`stat-value ${hasTumor ? 'text-error' : 'text-success'}`}>
                {prediction}
              </div>
              <div className="stat-desc">Classification result</div>
            </div>
          </div>

          {/* Confidence Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="font-medium">Confidence Level</span>
              <span className="text-base-content/60">{(confidence * 100).toFixed(2)}%</span>
            </div>
            <progress 
              className={`progress ${hasTumor ? 'progress-error' : 'progress-success'} w-full`} 
              value={confidence * 100} 
              max="100"
            ></progress>
          </div>
        </div>
      );
    } else {
      // Multiclass prediction result
      const probs = result.probabilities || result.probs || {};
      const prediction = result.predicted_class || result.label || result.prediction || "Unknown";
      
      console.log("Multiclass - Prediction:", prediction, "Probs:", probs); // Debug
      
      const classes = [
        { name: "CN", label: "Cognitively Normal", color: "success", icon: CheckCircle2 },
        { name: "MCI", label: "Mild Cognitive Impairment", color: "warning", icon: AlertCircle },
        { name: "AD", label: "Alzheimer's Disease", color: "error", icon: XCircle }
      ];

      const sortedClasses = classes
        .map(c => ({ ...c, prob: probs[c.name] || 0 }))
        .sort((a, b) => b.prob - a.prob);

      const topPrediction = sortedClasses[0];

      return (
        <div className="space-y-4 animate-in fade-in duration-500">
          <div className="divider my-2"></div>
          
          <div className={`alert alert-${topPrediction.color} shadow-lg`}>
            <Brain className="w-6 h-6" />
            <div>
              <h3 className="font-bold text-lg">Prediction: {topPrediction.label}</h3>
              <div className="text-sm opacity-90">
                Probability: {(topPrediction.prob * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium mb-2">
              <BarChart3 className="w-4 h-4" />
              <span>Class Probabilities</span>
            </div>
            
            {sortedClasses.map((cls) => {
              const Icon = cls.icon;
              return (
                <div key={cls.name} className="space-y-1">
                  <div className="flex justify-between items-center text-sm">
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4" />
                      <span className="font-medium">{cls.name}</span>
                      <span className="text-base-content/60">- {cls.label}</span>
                    </div>
                    <span className="font-semibold">{(cls.prob * 100).toFixed(2)}%</span>
                  </div>
                  <progress 
                    className={`progress progress-${cls.color} w-full`} 
                    value={cls.prob * 100} 
                    max="100"
                  ></progress>
                </div>
              );
            })}
          </div>

          {/* Summary Stats */}
          <div className="stats shadow w-full">
            <div className="stat">
              <div className="stat-title text-xs">Predicted Class</div>
              <div className="stat-value text-2xl">{prediction}</div>
              <div className="stat-desc">{topPrediction.label}</div>
            </div>
            <div className="stat">
              <div className="stat-title text-xs">Top Probability</div>
              <div className="stat-value text-2xl">
                {(topPrediction.prob * 100).toFixed(1)}%
              </div>
              <div className="stat-desc">
                {topPrediction.prob > 0.8 ? "High" : topPrediction.prob > 0.6 ? "Moderate" : "Low"}
              </div>
            </div>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="card bg-base-100 shadow-xl h-full min-h-[600px] flex flex-col">
      <div className="card-body flex flex-col">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          {isBinary ? (
            <Activity className="w-6 h-6 text-primary" />
          ) : (
            <Brain className="w-6 h-6 text-secondary" />
          )}
          <h2 className="card-title">{title}</h2>
        </div>
        
        <p className="text-sm text-base-content/60 mb-4">
          {isBinary 
            ? "Upload a brain MRI scan (.npy format) to detect tumors"
            : "Upload a brain MRI scan (.npy format) to classify cognitive state"
          }
        </p>

        {/* File Upload Area */}
        <div
          className={`
            border-2 border-dashed rounded-lg p-8 text-center transition-all flex-shrink-0
            ${dragActive ? 'border-primary bg-primary/5' : 'border-base-300 hover:border-primary/50'}
            ${file ? 'bg-base-200' : ''}
          `}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!file ? (
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".npy"
                className="hidden"
                onChange={(e) => handleFileChange(e.target.files[0])}
              />
              <div className="space-y-3">
                <div className="flex justify-center">
                  <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                    <Upload className="w-8 h-8 text-primary" />
                  </div>
                </div>
                <div>
                  <p className="font-semibold">Drop your .npy file here</p>
                  <p className="text-sm text-base-content/60 mt-1">or click to browse</p>
                </div>
                <div className="badge badge-outline badge-sm">Only .npy files</div>
              </div>
            </label>
          ) : (
            <div className="space-y-3">
              <div className="flex justify-center">
                <div className="w-16 h-16 rounded-full bg-success/10 flex items-center justify-center">
                  <FileUp className="w-8 h-8 text-success" />
                </div>
              </div>
              <div>
                <p className="font-semibold text-success">File Selected</p>
                <p className="text-sm text-base-content/60 mt-1 break-all">{file.name}</p>
                <p className="text-xs text-base-content/40 mt-1">
                  {(file.size / 1024).toFixed(2)} KB
                </p>
              </div>
              <button 
                className="btn btn-ghost btn-sm gap-2" 
                onClick={resetUpload}
                disabled={loading}
              >
                <XCircle className="w-4 h-4" />
                Change File
              </button>
            </div>
          )}
        </div>

        {/* Error Alert */}
        {error && (
          <div className="alert alert-error shadow-lg animate-in fade-in flex-shrink-0">
            <FileWarning className="w-5 h-5" />
            <span>{error}</span>
          </div>
        )}

        {/* Action Buttons */}
        <div className="card-actions justify-end mt-4 flex-shrink-0">
          {result && (
            <button 
              className="btn btn-ghost gap-2" 
              onClick={resetUpload}
              disabled={loading}
            >
              <Upload className="w-4 h-4" />
              New Upload
            </button>
          )}
          <button
            className={`btn btn-primary gap-2 ${loading ? 'loading' : ''}`}
            onClick={handlePredict}
            disabled={!file || loading}
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                Predict
              </>
            )}
          </button>
        </div>

        {/* Results Display */}
        <div className="flex-grow">
          {renderResult()}
        </div>

        {/* Debug JSON (collapsed by default) */}
        {result && (
          <div className="collapse collapse-arrow bg-base-200 mt-4 flex-shrink-0">
            <input type="checkbox" /> 
            <div className="collapse-title text-sm font-medium">
              View Raw JSON Response
            </div>
            <div className="collapse-content"> 
              <pre className="text-xs overflow-x-auto bg-base-300 p-3 rounded">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}