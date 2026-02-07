import Navbar from "../components/Navbar";
import UploadCard from "../components/UploadCard";
import Graph from "../components/Graph.jsx";
import { Brain, Activity, Zap, Shield } from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-base-100">
      <Navbar />

      <div className="max-w-7xl mx-auto px-6 py-12 space-y-12">
        {/* Hero Header */}
        <div className="text-center space-y-6 animate-in fade-in duration-700">
          <div className="flex justify-center mb-4">
            <div className="relative">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary via-secondary to-accent shadow-xl flex items-center justify-center animate-pulse">
                <Brain className="w-12 h-12 text-white" />
              </div>
              <div className="absolute -top-1 -right-1 w-6 h-6 bg-success rounded-full flex items-center justify-center animate-bounce">
                <Zap className="w-4 h-4 text-white" />
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h1 className="text-5xl font-extrabold bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent leading-tight">
              AI-Powered Neurological Screening
            </h1>
            <p className="text-lg text-base-content/70 max-w-2xl mx-auto">
              Upload T1-weighted MRI scans for instant AI-based analysis with state-of-the-art deep learning models
            </p>
          </div>

          {/* Feature Pills */}
          <div className="flex flex-wrap justify-center gap-3 pt-4">
            <div className="badge badge-lg gap-2 bg-primary/10 text-primary border-primary/20 px-4 py-3">
              <Activity className="w-4 h-4" />
              Real-time Analysis
            </div>
            <div className="badge badge-lg gap-2 bg-secondary/10 text-secondary border-secondary/20 px-4 py-3">
              <Shield className="w-4 h-4" />
              Medical-Grade AI
            </div>
            <div className="badge badge-lg gap-2 bg-accent/10 text-accent border-accent/20 px-4 py-3">
              <Zap className="w-4 h-4" />
              Instant Results
            </div>
          </div>
        </div>

        {/* Prediction Cards Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-3 px-2">
            <div className="h-1 w-12 bg-gradient-to-r from-primary to-transparent rounded-full"></div>
            <h2 className="text-2xl font-bold">Diagnostic Models</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6 items-start">
            <UploadCard
              title="Binary Classification (CN vs AD)"
              endpoint="/predict/binary"
            />

            <UploadCard
              title="Multi-Class Classification (CN / MCI / AD)"
              endpoint="/predict/multiclass"
            />
          </div>
        </div>

        {/* Analytics Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-3 px-2">
            <div className="h-1 w-12 bg-gradient-to-r from-secondary to-transparent rounded-full"></div>
            <h2 className="text-2xl font-bold">Prediction Analytics</h2>
          </div>
          
          <div className="card bg-gradient-to-br from-base-100 to-base-200 shadow-xl border border-base-300">
            <div className="card-body p-8">
              <Graph/>
            </div>
          </div>
        </div>

        {/* Footer Info */}
        <div className="text-center pt-8 pb-4">
          <div className="divider"></div>
          <p className="text-sm text-base-content/50">
            Powered by advanced deep learning • Trained on medical imaging datasets
          </p>
        </div>
      </div>
    </div>
  );
}