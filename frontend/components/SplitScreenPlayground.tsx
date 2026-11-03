import React, { useState } from 'react';

interface BenchmarkTelemetry {
  tier: string;
  confidence: number;
  latencyMs: number;
  costSavedUsd: number;
  savingsPct: number;
  citations: string[];
}

export const SplitScreenPlayground: React.FC = () => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'economy' | 'balanced' | 'strict'>('balanced');
  const [loading, setLoading] = useState(false);
  const [calibResult, setCalibResult] = useState<any | null>(null);

  const handleRunComparison = async () => {
    if (!query) return;
    setLoading(true);
    // Simulates side-by-side execution against /v1/chat/completions
    setTimeout(() => {
      setCalibResult({
        tier: query.includes('defect') ? 'Tier 1: Fast Vector RAG' : query.includes('yesterday') ? 'Tier 3: LangGraph + MCP (Escalated)' : 'Tier 2: Hybrid RAG',
        confidence: query.includes('2027') ? 0.14 : 0.94,
        latencyMs: query.includes('defect') ? 680 : query.includes('yesterday') ? 3900 : 1800,
        costSavedUsd: query.includes('defect') ? 0.00832 : 0.0021,
        savingsPct: query.includes('defect') ? 97.8 : 24.7,
        citations: query.includes('defect') ? ['QA_SOP, p.12'] : ['MCP-ERP #402', 'VoltCore_SLA, p.6'],
        calibAnswer: query.includes('defect')
          ? 'The maximum defect tolerance percentage allowed for lithium cell batches is 0.5% [QA_SOP, p.12].'
          : query.includes('2027')
          ? 'CalibRAG Refusal to Guess: Tender submissions and pricing for Q3 2027 are not present in workspace documents (0% hallucination).'
          : 'Line halted yesterday due to 9-day shipment delay on Batch #402, breaching SLA Clause 8.2 [MCP-ERP #402, VoltCore_SLA, p.6].',
        fixedRagAnswer: query.includes('2027')
          ? 'I believe VoltCore India will likely win the Q3 2027 cobalt tender at approximately $32,000 per ton based on industry trends (Hallucination).'
          : 'Based on search results, late delivery penalty is 5% per week.',
      });
      setLoading(false);
    }, 600);
  };

  return (
    <div className="flex flex-col gap-6 p-8 max-w-7xl mx-auto font-sans">
      <div className="flex justify-between items-center border-b pb-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            CalibRAG Split-Screen Comparison Playground
          </h1>
          <p className="text-sm text-slate-500">
            Watch Standard Fixed RAG vs. CalibRAG Adaptive System 1/2 execute side-by-side.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-semibold uppercase text-slate-500">Operating Mode:</span>
          {(['economy', 'balanced', 'strict'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md capitalize transition ${
                mode === m ? 'bg-indigo-600 text-white shadow' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question (e.g., 'What is maximum defect tolerance?', 'Why was line halted yesterday?')"
          className="flex-1 px-4 py-2.5 rounded-lg border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button
          onClick={handleRunComparison}
          disabled={loading}
          className="px-6 py-2.5 bg-indigo-600 text-white font-medium text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Executing...' : 'Run Side-by-Side'}
        </button>
      </div>

      {calibResult && (
        <div className="grid grid-cols-2 gap-6 mt-4">
          {/* Left Panel: Standard Fixed RAG */}
          <div className="border border-slate-200 rounded-xl p-6 bg-slate-50/50 shadow-sm flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center pb-3 border-b border-slate-200">
                <span className="font-semibold text-slate-700">Standard Fixed RAG (Always Basic or Agentic)</span>
                <span className="text-xs bg-slate-200 text-slate-700 px-2.5 py-1 rounded-full font-mono">
                  Fixed Pipeline
                </span>
              </div>
              <p className="mt-4 text-sm text-slate-800 leading-relaxed">{calibResult.fixedRagAnswer}</p>
            </div>
            <div className="mt-8 pt-4 border-t border-slate-200 grid grid-cols-3 gap-2 text-xs text-slate-600 font-mono">
              <div>Latency: 12,400 ms</div>
              <div>Cost: $0.008500</div>
              <div>Hallucination: Vulnerable</div>
            </div>
          </div>

          {/* Right Panel: CalibRAG */}
          <div className="border border-indigo-200 rounded-xl p-6 bg-white shadow-md flex flex-col justify-between ring-1 ring-indigo-500/10">
            <div>
              <div className="flex justify-between items-center pb-3 border-b border-indigo-100">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-indigo-950">CalibRAG (Adaptive Control Plane)</span>
                  <span className="text-xs bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded font-medium">
                    {calibResult.savingsPct}% Cost Saved
                  </span>
                </div>
                <span className="text-xs bg-indigo-50 text-indigo-700 px-2.5 py-1 rounded-full font-mono">
                  {calibResult.tier}
                </span>
              </div>
              <p className="mt-4 text-sm text-slate-900 leading-relaxed font-normal">{calibResult.calibAnswer}</p>
              
              <div className="mt-4 flex flex-wrap gap-2">
                {calibResult.citations.map((cite: string) => (
                  <span key={cite} className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded border border-slate-200 font-mono">
                    [{cite}]
                  </span>
                ))}
              </div>
            </div>

            <div className="mt-8 pt-4 border-t border-slate-100 grid grid-cols-4 gap-2 text-xs font-mono text-slate-700">
              <div>Latency: {calibResult.latencyMs} ms</div>
              <div>Confidence: {calibResult.confidence}</div>
              <div className="text-emerald-600 font-semibold">Saved: ${calibResult.costSavedUsd}</div>
              <div>Mode: {mode}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
