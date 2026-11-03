import React, { useState } from 'react';

export const BYOKWorkspaceStudio: React.FC = () => {
  const [workspaceName, setWorkspaceName] = useState('Pune-Manufacturing-Ops');
  const [mcpPostgresActive, setMcpPostgresActive] = useState(true);
  const [mcpGithubActive, setMcpGithubActive] = useState(true);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([
    'Supplier_SLA_Contracts_2026.pdf',
    'QA_SOP_Lithium_Batches.pdf',
    'Operations_Manual_v3.docx'
  ]);

  return (
    <div className="max-w-5xl mx-auto p-8 font-sans space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">3-Click BYOK Knowledge Studio</h2>
        <p className="text-sm text-slate-500">
          Upload unstructured enterprise documents and connect live FastMCP tools in under 3 minutes.
        </p>
      </div>

      {/* Workspace Config */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <label className="block text-xs font-semibold text-slate-700 uppercase">Active Workspace Name</label>
        <input
          type="text"
          value={workspaceName}
          onChange={(e) => setWorkspaceName(e.target.value)}
          className="w-full max-w-md px-3.5 py-2 border border-slate-300 rounded-lg text-sm"
        />
      </div>

      {/* Drag and Drop Box */}
      <div className="bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-8 text-center hover:border-indigo-500 transition cursor-pointer">
        <p className="text-sm font-medium text-slate-700">Drag & drop PDF, DOCX, or Markdown files here</p>
        <p className="text-xs text-slate-500 mt-1">Automatic parsing via IBM Docling into pgvector, BM25, and LightRAG</p>
        
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {uploadedFiles.map((file) => (
            <span key={file} className="bg-white border border-slate-200 text-xs px-3 py-1.5 rounded-lg text-slate-600 font-mono">
              📄 {file}
            </span>
          ))}
        </div>
      </div>

      {/* FastMCP Connectors */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-slate-900 uppercase">Live FastMCP Connectors</h3>
        
        <div className="flex items-center justify-between p-4 border border-slate-200 rounded-lg">
          <div>
            <div className="font-semibold text-sm text-slate-800">PostgreSQL ERP Connector</div>
            <div className="text-xs text-slate-500">Read-only connection for live shipment tables and batch defects</div>
          </div>
          <button
            onClick={() => setMcpPostgresActive(!mcpPostgresActive)}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold ${
              mcpPostgresActive ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-700'
            }`}
          >
            {mcpPostgresActive ? 'Connected' : 'Disabled'}
          </button>
        </div>

        <div className="flex items-center justify-between p-4 border border-slate-200 rounded-lg">
          <div>
            <div className="font-semibold text-sm text-slate-800">GitHub Issues MCP Connector</div>
            <div className="text-xs text-slate-500">Queries open blockers, engineering PRs, and team assignees</div>
          </div>
          <button
            onClick={() => setMcpGithubActive(!mcpGithubActive)}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold ${
              mcpGithubActive ? 'bg-emerald-600 text-white' : 'bg-slate-200 text-slate-700'
            }`}
          >
            {mcpGithubActive ? 'Connected' : 'Disabled'}
          </button>
        </div>
      </div>
    </div>
  );
};
