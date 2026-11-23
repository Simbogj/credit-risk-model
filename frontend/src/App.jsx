import React, { useState, useEffect } from 'react';
import { checkHealth, scoreCustomer } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [apiStatus, setApiStatus] = useState('checking');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const [formData, setFormData] = useState({
    totalAmount: 25000.0,
    avgAmount: 1250.0,
    transactionCount: 20,
    recency: 5,
    frequency: 20,
    monetary: 25000.0,
    fraudRate: 0.02,
    stdAmount: 500.0,
    creditRatio: 0.15,
    debitRatio: 0.85,
  });

  useEffect(() => {
    checkHealth().then(res => {
      setApiStatus(res.status === 'healthy' ? 'online' : 'offline');
    });
  }, []);

  const applyPreset = (type) => {
    if (type === 'prime') {
      setFormData({
        totalAmount: 48000.0, avgAmount: 2400.0, transactionCount: 25, recency: 2,
        frequency: 25, monetary: 48000.0, fraudRate: 0.00, stdAmount: 350.0,
        creditRatio: 0.10, debitRatio: 0.90
      });
    } else if (type === 'borderline') {
      setFormData({
        totalAmount: 16000.0, avgAmount: 1100.0, transactionCount: 14, recency: 16,
        frequency: 11, monetary: 14000.0, fraudRate: 0.04, stdAmount: 850.0,
        creditRatio: 0.40, debitRatio: 0.60
      });
    } else if (type === 'high_risk') {
      setFormData({
        totalAmount: 3200.0, avgAmount: 380.0, transactionCount: 4, recency: 48,
        frequency: 3, monetary: 2100.0, fraudRate: 0.18, stdAmount: 1200.0,
        creditRatio: 0.75, debitRatio: 0.25
      });
    }
  };

  const handleScore = async () => {
    setLoading(true);
    const res = await scoreCustomer('REACT-CLI-001', {
      TotalAmount: parseFloat(formData.totalAmount),
      AvgAmount: parseFloat(formData.avgAmount),
      TransactionCount: parseInt(formData.transactionCount),
      Recency: parseInt(formData.recency),
      Frequency: parseInt(formData.frequency),
      Monetary: parseFloat(formData.monetary),
      FraudRate: parseFloat(formData.fraudRate),
      StdAmount: parseFloat(formData.stdAmount),
      CreditRatio: parseFloat(formData.creditRatio),
      DebitRatio: parseFloat(formData.debitRatio),
    });
    setResult(res);
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-[#F8FAFC]">
      {/* Sidebar */}
      <aside className="w-full md:w-64 bg-white border-r border-slate-200 p-5 flex flex-col justify-between shrink-0">
        <div>
          <div className="flex items-center gap-3 pb-5 mb-5 border-b border-slate-200">
            <div className="w-10 h-10 rounded-lg bg-[#0B2545] text-white flex items-center justify-center font-black text-xl shadow-md">
              B
            </div>
            <div>
              <div className="text-base font-extrabold text-[#0B2545]">BATI BANK</div>
              <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Credit Intelligence</div>
            </div>
          </div>

          <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 px-1">Workspaces</div>
          <nav className="space-y-1">
            {[
              { id: 'overview', label: '🏢 Portfolio Overview' },
              { id: 'underwriting', label: '🎯 Loan Underwriting' },
              { id: 'governance', label: 'ℹ️ Regulatory & Governance' },
            ].map(item => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-bold transition flex items-center justify-between ${
                  activeTab === item.id ? 'bg-[#0B2545] text-white' : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                <span>{item.label}</span>
                {activeTab === item.id && <span className="w-1.5 h-1.5 rounded-full bg-blue-400"></span>}
              </button>
            ))}
          </nav>
        </div>

        <div className="mt-8 pt-4 border-t border-slate-200 text-[11px] text-slate-500 space-y-1">
          <div><b>Model:</b> DecisionTree v1.0</div>
          <div><b>Status:</b> Basel II Aligned</div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 md:p-8 overflow-y-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-[#0B2545] to-[#133E87] rounded-xl p-6 mb-6 text-white shadow-md">
          <div className="flex flex-col md:flex-row justify-between gap-4">
            <div>
              <div className="text-xs font-bold text-blue-200 uppercase tracking-wider mb-1">Bati Bank Financial Services</div>
              <h1 className="text-2xl font-black">Credit Risk Intelligence Platform (React)</h1>
              <p className="text-sm text-slate-300 mt-1">Enterprise Credit Risk Scoring & Underwriting Engine</p>
            </div>
            <div className="flex items-center gap-2">
              <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                apiStatus === 'online' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
              }`}>
                ● API: {apiStatus}
              </span>
            </div>
          </div>
        </div>

        {/* Content Tabs */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white p-5 rounded-xl border border-slate-200">
                <div className="text-xs font-bold text-slate-400 uppercase">Scored Customers</div>
                <div className="text-2xl font-black text-slate-900 mt-1">3,742</div>
                <div className="text-xs text-slate-500 mt-1">100% Coverage</div>
              </div>
              <div className="bg-white p-5 rounded-xl border border-slate-200">
                <div className="text-xs font-bold text-slate-400 uppercase">Prime Borrowers</div>
                <div className="text-2xl font-black text-emerald-600 mt-1">2,488 (66.5%)</div>
                <div className="text-xs text-emerald-600 mt-1">Low Risk</div>
              </div>
              <div className="bg-white p-5 rounded-xl border border-slate-200">
                <div className="text-xs font-bold text-slate-400 uppercase">Watchlist Borrowers</div>
                <div className="text-2xl font-black text-rose-600 mt-1">1,254 (33.5%)</div>
                <div className="text-xs text-rose-600 mt-1">Elevated Risk</div>
              </div>
              <div className="bg-white p-5 rounded-xl border border-slate-200">
                <div className="text-xs font-bold text-slate-400 uppercase">Champion ROC-AUC</div>
                <div className="text-2xl font-black text-blue-600 mt-1">1.0000</div>
                <div className="text-xs text-blue-600 mt-1">DecisionTree</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'underwriting' && (
          <div className="space-y-6">
            {/* Presets */}
            <div className="bg-white p-5 rounded-xl border border-slate-200">
              <h3 className="font-bold text-slate-900 text-sm mb-2">⚡ Quick Presets</h3>
              <div className="grid grid-cols-3 gap-3">
                <button onClick={() => applyPreset('prime')} className="p-3 bg-emerald-50 text-emerald-800 text-xs font-bold rounded-lg border border-emerald-200">
                  🟢 Prime Borrower
                </button>
                <button onClick={() => applyPreset('borderline')} className="p-3 bg-amber-50 text-amber-800 text-xs font-bold rounded-lg border border-amber-200">
                  🟡 Borderline Review
                </button>
                <button onClick={() => applyPreset('high_risk')} className="p-3 bg-rose-50 text-rose-800 text-xs font-bold rounded-lg border border-rose-200">
                  🔴 High Risk Borrower
                </button>
              </div>
            </div>

            {/* Inputs */}
            <div className="bg-white p-6 rounded-xl border border-slate-200">
              <h3 className="font-bold text-slate-900 text-base mb-4">Underwriting Dossier</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-600 block mb-1">Total Transaction Amount ($)</label>
                  <input type="number" value={formData.totalAmount} onChange={(e) => setFormData({...formData, totalAmount: e.target.value})}
                    className="w-full text-sm border border-slate-300 rounded p-2" />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600 block mb-1">Recency (Days)</label>
                  <input type="number" value={formData.recency} onChange={(e) => setFormData({...formData, recency: e.target.value})}
                    className="w-full text-sm border border-slate-300 rounded p-2" />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-600 block mb-1">Fraud Rate</label>
                  <input type="number" step="0.01" value={formData.fraudRate} onChange={(e) => setFormData({...formData, fraudRate: e.target.value})}
                    className="w-full text-sm border border-slate-300 rounded p-2" />
                </div>
              </div>

              <button onClick={handleScore} disabled={loading}
                className="mt-6 w-full py-3 bg-[#0B2545] text-white text-sm font-bold rounded-lg hover:bg-[#133E87] transition">
                {loading ? 'Evaluating...' : '⚡ EXECUTE UNDERWRITING ASSESSMENT'}
              </button>
            </div>

            {/* Result */}
            {result && (
              <div className="bg-white p-6 rounded-xl border-2 border-slate-200">
                <div className={`p-4 rounded-lg mb-4 ${
                  result.decision.decision === 'Approve' ? 'bg-emerald-50 text-emerald-900 border border-emerald-300' :
                  result.decision.decision === 'Review' ? 'bg-amber-50 text-amber-900 border border-amber-300' :
                  'bg-rose-50 text-rose-900 border border-rose-300'
                }`}>
                  <div className="text-xs font-bold uppercase tracking-wider">Determination</div>
                  <div className="text-2xl font-black mt-1">
                    {result.decision.decision.toUpperCase()} — Score: {result.prediction.credit_score}
                  </div>
                  <div className="text-xs mt-1">Default Probability: {(result.prediction.risk_probability * 100).toFixed(1)}%</div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'governance' && (
          <div className="bg-white p-6 rounded-xl border border-slate-200">
            <h3 className="font-bold text-slate-900 text-base mb-2">⚖️ Basel II Compliance</h3>
            <p className="text-xs text-slate-600">
              This React view interfaces with the Bati Bank credit risk API to deliver real-time, explainable loan decisions.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
