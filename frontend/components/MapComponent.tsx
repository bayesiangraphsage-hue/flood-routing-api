"use client";

import { motion } from "framer-motion";
import { AlertTriangle, RotateCcw, Route } from "lucide-react";

interface Props {
  origin: [number, number] | null;
  destination: [number, number] | null;
  loading: boolean;
  risk: number;
  setRisk: (value: number) => void;
  model: string;
  setModel: (value: string) => void;
  onCalculate: () => void;
  onReset: () => void;
}

export default function Sidebar({
  origin,
  destination,
  loading,
  risk,
  setRisk,
  model,
  setModel,
  onCalculate,
  onReset,
}: Props) {
  return (
    <motion.div
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      className="glass absolute z-[1000] left-4 top-4 w-[340px] rounded-2xl p-5 shadow-2xl"
    >
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangle className="text-cyan-400" />
        <h1 className="text-xl font-bold">
          Flood Routing AI
        </h1>
      </div>

      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-300">
            Origin
          </p>

          <div className="text-xs mt-1 bg-slate-900 p-2 rounded-lg">
            {origin
              ? `${origin[0].toFixed(5)}, ${origin[1].toFixed(5)}`
              : "Click map"}
          </div>
        </div>

        <div>
          <p className="text-sm text-gray-300">
            Destination
          </p>

          <div className="text-xs mt-1 bg-slate-900 p-2 rounded-lg">
            {destination
              ? `${destination[0].toFixed(5)}, ${destination[1].toFixed(5)}`
              : "Click map"}
          </div>
        </div>

        <div>
          <label className="text-sm">
            Routing Model
          </label>

          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full mt-2 p-2 rounded-lg bg-slate-900"
          >
            <option>Baseline Dijkstra</option>
            <option>Deterministic GraphSAGE</option>
            <option>Bayesian MC Dropout</option>
            <option>Variational Inference</option>
          </select>
        </div>

        <div>
          <label className="text-sm">
            Risk Aversion (CVaR β): {risk.toFixed(2)}
          </label>

          <input
            type="range"
            min="0.50"
            max="0.99"
            step="0.01"
            value={risk}
            onChange={(e) => setRisk(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <button
          onClick={onCalculate}
          disabled={loading || !origin || !destination}
          className="w-full bg-cyan-500 hover:bg-cyan-400 transition rounded-xl py-3 flex items-center justify-center gap-2"
        >
          <Route size={18} />

          {loading ? "Calculating..." : "Calculate Safe Route"}
        </button>

        <button
          onClick={onReset}
          className="w-full bg-red-500 hover:bg-red-400 transition rounded-xl py-3 flex items-center justify-center gap-2"
        >
          <RotateCcw size={18} />
          Reset
        </button>
      </div>
    </motion.div>
  );
}