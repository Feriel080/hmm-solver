'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlgorithmFormulas } from '@/components/AlgorithmFormulas';
import { ParameterInput } from '@/components/ParameterInput';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { DemoSection } from '@/components/DemoSection';

const DEFAULT_STATES   = ['S1', 'S2'];
const DEFAULT_OBS_SYM  = ['o1', 'o2', 'o3'];

const ALGORITHM_OPTIONS = [
  { value: 'forward',    label: 'Forward Algorithm' },
  { value: 'backward',   label: 'Backward Algorithm' },
  { value: 'viterbi',    label: 'Viterbi Algorithm' },
  { value: 'baum_welch', label: 'Baum-Welch Algorithm' },
];

export default function Home() {
  const [mode, setMode] = useState<'demo' | 'numerical-discrete' | 'numerical-continuous'>('demo');
  const [algorithms, setAlgorithms] = useState<string[]>(['forward']);
  const [dimension, setDimension]   = useState<'1d' | 'nd'>('1d');

  // shared states / vocab
  const [states, setStates] = useState(DEFAULT_STATES);
  const [vocab,  setVocab]  = useState(DEFAULT_OBS_SYM);

  // continuous observation labels + values
  const [obsValues, setObsValues]       = useState<string[]>(DEFAULT_OBS_SYM.map(() => ''));

  // discrete vocab label editing
  const [editingDiscIdx, setEditingDiscIdx] = useState<number | null>(null);
  const [editingDiscVal, setEditingDiscVal] = useState('');

  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const [showConvertModal, setShowConvertModal] = useState(false);
  const [convertSymbols, setConvertSymbols] = useState([
    { name: 'Low', lo: '', hi: '' },
    { name: 'Mid', lo: '', hi: '' },
    { name: 'High', lo: '', hi: '' },
  ]);
  const [pendingConvertParams, setPendingConvertParams] = useState<any>(null);
  const [convertedB, setConvertedB] = useState<any>(null);
  const [convertedObs, setConvertedObs] = useState<string>('');

  const [convertedPi,  setConvertedPi]  = useState<number[] | null>(null);
  const [convertedA,   setConvertedA]   = useState<number[][] | null>(null);

  const [convertKey, setConvertKey] = useState(0);

  const [showRetrieveModal, setShowRetrieveModal] = useState(false);

  // ── algorithm toggle ──────────────────────────────────────────────────────
  const toggleAlgorithm = (value: string) =>
    setAlgorithms(prev =>
      prev.includes(value)
        ? prev.length === 1 ? prev : prev.filter(a => a !== value)
        : [...prev, value]
    );

  // ── API call helper ───────────────────────────────────────────────────────
  const callHMM = async (body: object) => {
    const r = await fetch('/api/hmm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return r.json();
  };

  // ── discrete handler ──────────────────────────────────────────────────────
  const handleDiscreteChange = async (params: any) => {
    setLoading(true);
    try {
      const responses = await Promise.all(
        algorithms.map(algo =>
          callHMM({
            type: 'numerical-discrete',
            algorithm: algo,
            observation: params.observation,
            pi: params.pi,
            a: params.a,
            b: params.b,
            states,
            vocab,
          }).then(data => ({ algo, data }))
        )
      );
      setResults(Object.fromEntries(responses.map(({ algo, data }) => [algo, data])));
    } catch (e) {
      setResults({ error: 'Failed to compute', details: String(e) });
    } finally {
      setLoading(false);
    }
  };

  // ── continuous handler ────────────────────────────────────────────────────
  const handleContinuousChange = async (params: any) => {
    setLoading(true);
    try {
      const responses = await Promise.all(
        algorithms.map(algo =>
          callHMM({
            type: 'numerical-continuous',
            algorithm: algo,
            observation: params.observation,
            pi: params.pi,
            a: params.a,
            means: params.means,
            sigmas_or_covs: params.sigmas,
            states,
            dimension,
            iterations: params.bwIterations,
          }).then(data => ({ algo, data }))
        )
      );
      setResults(Object.fromEntries(responses.map(({ algo, data }) => [algo, data])));
    } catch (e) {
      setResults({ error: 'Failed to compute', details: String(e) });
    } finally {
      setLoading(false);
    }
  };

  // ── convert continuous → discrete ────────────────────────────────────────
  const handleConvertToDiscrete = (params: any) => {
    setPendingConvertParams(params);
    setShowConvertModal(true);
  };

  const handleConvertConfirm = async () => {
    if (!pendingConvertParams) return;
    setLoading(true);
    setShowConvertModal(false);
    try {
      const intervals = convertSymbols.map(s => [
        s.lo === '' ? null : parseFloat(s.lo),
        s.hi === '' ? null : parseFloat(s.hi),
      ]);
      const data = await callHMM({
        type: 'convert-to-discrete',
        observation: pendingConvertParams.observation,
        means: pendingConvertParams.means,
        sigmas: pendingConvertParams.sigmas,
        pi: pendingConvertParams.pi,
        a: pendingConvertParams.a,
        states,
        symbols: convertSymbols.map(s => s.name),
        intervals,
      });
      if (data.vocab){
        setVocab(data.vocab);
        setConvertedB(data.b);
        setConvertedObs(data.discrete_observation.join(' '));
        setConvertedPi(pendingConvertParams.pi);
        setConvertedA(pendingConvertParams.a);
        setConvertKey(k => k + 1);
        setResults(null);
        setMode('numerical-discrete');
        setShowRetrieveModal(true); 
      }

    } catch (e){
      setResults({ error: 'Conversion failed', details: String(e) });
    } finally {
      setLoading(false);
    }
  }

  // ── states manager ────────────────────────────────────────────────────────
  const addState    = () => setStates(p => [...p, `S${p.length + 1}`]);
  const removeState = (idx: number) => { if (states.length > 1) setStates(p => p.filter((_, i) => i !== idx)); };

  // ── discrete vocab ────────────────────────────────────────────────────────
  const addObsSym    = () => setVocab(p => [...p, `o${p.length + 1}`]);
  const removeVocab  = (idx: number) => { if (vocab.length > 1) setVocab(p => p.filter((_, i) => i !== idx)); };

  // ── continuous vocab ──────────────────────────────────────────────────────
  const addContObs    = () => { setVocab(p => [...p, `obs${p.length + 1}`]); setObsValues(p => [...p, '']); };
  const removeContObs = (idx: number) => {
    if (vocab.length > 1) {
      setVocab(p => p.filter((_, i) => i !== idx));
      setObsValues(p => p.filter((_, i) => i !== idx));
    }
  };

  // ── shared checkbox block ─────────────────────────────────────────────────
  const AlgorithmCheckboxes = () => (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">Algorithm</label>
      <div className="space-y-2">
        {ALGORITHM_OPTIONS.map(({ value, label }) => (
          <label key={value} className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={algorithms.includes(value)}
              onChange={() => toggleAlgorithm(value)}
              className="accent-primary w-4 h-4 cursor-pointer"
            />
            <span className="text-sm text-foreground group-hover:text-primary transition-colors">{label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  // ── shared states manager ─────────────────────────────────────────────────
  const [editingStateIdx, setEditingStateIdx] = useState<number | null>(null);
  const [editingStateVal, setEditingStateVal] = useState('');

  const StatesManager = () => (
    <div>
      <label className="text-sm font-medium text-foreground block mb-2">
        Hidden States ({states.length})
      </label>
      <div className="space-y-2 max-h-40 overflow-y-auto">
        {states.map((state, idx) => (
          <div key={idx} className="flex items-center justify-between gap-2">
            {editingStateIdx === idx ? (
              <input
                autoFocus
                className="text-sm border-b border-primary bg-transparent outline-none flex-1 text-foreground"
                value={editingStateVal}
                onChange={e => setEditingStateVal(e.target.value)}
                onBlur={() => {
                  const t = editingStateVal.trim();
                  if (t) { const u = [...states]; u[idx] = t; setStates(u); }
                  setEditingStateIdx(null);
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                  if (e.key === 'Escape') setEditingStateIdx(null);
                }}
              />
            ) : (
              <span
                className="text-sm text-foreground cursor-pointer hover:text-primary hover:underline underline-offset-2 select-none"
                title="Double-click to rename"
                onDoubleClick={() => { setEditingStateIdx(idx); setEditingStateVal(state); }}
              >
                {state}
              </span>
            )}
            <Button onClick={() => removeState(idx)} variant="ghost" size="sm" disabled={states.length === 1}>×</Button>
          </div>
        ))}
      </div>
      <Button onClick={addState} variant="outline" size="sm" className="w-full mt-2">+ Add State</Button>
    </div>
  );

  // ── sidebar for discrete ──────────────────────────────────────────────────
  const DiscreteSidebar = () => (
    <Card className="p-6">
      <h2 className="text-xl font-bold text-foreground mb-4">Configuration</h2>
      <div className="space-y-6">
        <AlgorithmCheckboxes />
        <StatesManager />
        <div>
          <label className="text-sm font-medium text-foreground block mb-2">
            Observation Symbols ({vocab.length})
          </label>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {vocab.map((word, idx) => (
              <div key={idx} className="flex items-center justify-between gap-2">
                {editingDiscIdx === idx ? (
                  <input
                    autoFocus
                    className="text-sm border-b border-primary bg-transparent outline-none flex-1 text-foreground"
                    value={editingDiscVal}
                    onChange={e => setEditingDiscVal(e.target.value)}
                    onBlur={() => {
                      const t = editingDiscVal.trim();
                      if (t) { const u = [...vocab]; u[idx] = t; setVocab(u); }
                      setEditingDiscIdx(null);
                    }}
                    onKeyDown={e => {
                      if (e.key === 'Enter') e.currentTarget.blur();
                      if (e.key === 'Escape') setEditingDiscIdx(null);
                    }}
                  />
                ) : (
                  <span
                    className="text-sm text-foreground cursor-pointer hover:text-primary hover:underline underline-offset-2 select-none"
                    title="Double-click to rename"
                    onDoubleClick={() => { setEditingDiscIdx(idx); setEditingDiscVal(word); }}
                  >
                    {word}
                  </span>
                )}
                <Button onClick={() => removeVocab(idx)} variant="ghost" size="sm" disabled={vocab.length === 1}>×</Button>
              </div>
            ))}
          </div>
          <Button onClick={addObsSym} variant="outline" size="sm" className="w-full mt-2">+ Add Symbol</Button>
        </div>
      </div>
    </Card>
  );

  // ── sidebar for continuous ────────────────────────────────────────────────
  const ContinuousSidebar = () => (
    <Card className="p-6">
      <h2 className="text-xl font-bold text-foreground mb-4">Configuration</h2>
      <div className="space-y-6">
        {/* dimension */}
        <div>
          <label className="text-sm font-medium text-foreground block mb-2">Problem Dimension</label>
          <Select value={dimension} onValueChange={(v: any) => setDimension(v)}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="1d">1 Dimension</SelectItem>
              <SelectItem value="nd">Multi-Dimension</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <AlgorithmCheckboxes />
        <StatesManager />

        {/* Convert to discrete button */}
        <Button
          variant="outline"
          size="sm"
          className="w-full border-dashed"
          onClick={() => {
            // trigger a special "convert" action from ParameterInput
            // We fire a custom event the ParameterInput listens to
            window.dispatchEvent(new CustomEvent('hmm-request-convert-params'));
          }}
        >
          ⇄ Convert to Discrete HMM
        </Button>
      </div>
    </Card>
  );

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">HMM Solver</h1>
          <p className="text-muted-foreground">
            Explore Hidden Markov Models with Forward, Backward, Viterbi, and Baum-Welch algorithms
          </p>
        </div>

        <Tabs
          value={mode}
          onValueChange={(v: string) => { setMode(v as typeof mode); setResults(null); }}
          className="space-y-6"
        >
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="demo">Ambiguous Phrase</TabsTrigger>
            <TabsTrigger value="numerical-discrete">Discrete HMM</TabsTrigger>
            <TabsTrigger value="numerical-continuous">Continuous HMM</TabsTrigger>
          </TabsList>

          {/* ── Demo ── */}
          <TabsContent value="demo" className="space-y-6">
            <DemoSection onRun={() => {}} />
          </TabsContent>

          {/* ── Discrete ── */}
          <TabsContent value="numerical-discrete" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1"><DiscreteSidebar /></div>
              <div className="lg:col-span-2 space-y-6">
                <Card className="p-6">
                  <h2 className="text-xl font-bold text-foreground mb-4">HMM Parameters</h2>
                  <ParameterInput
                    key={convertKey} 
                    states={states}
                    vocab={vocab}
                    algorithms={algorithms}
                    mode="discrete"
                    onParametersChange={handleDiscreteChange}
                    initialB={convertedB}
                    initialObservation={convertedObs}
                  />
                </Card>
                <div>
                  <h2 className="text-xl font-bold text-foreground mb-4">Results</h2>
                  <ResultsDisplay results={results} loading={loading} algorithms={algorithms} />
                </div>
                <AlgorithmFormulas selectedAlgorithms={algorithms} />
              </div>
            </div>
          </TabsContent>

          {/* ── Continuous ── */}
          <TabsContent value="numerical-continuous" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1"><ContinuousSidebar /></div>
              <div className="lg:col-span-2 space-y-6">
                <Card className="p-6">
                  <h2 className="text-xl font-bold text-foreground mb-4">HMM Parameters</h2>
                  <ParameterInput
                    key={convertKey} 
                    states={states}
                    vocab={vocab}
                    algorithms={algorithms}
                    mode="continuous-1d"
                    onParametersChange={handleContinuousChange}
                    onConvertToDiscrete={handleConvertToDiscrete}
                    initialB={convertedB}
                    initialObservation={convertedObs}
                  />
                </Card>
                <div>
                  <h2 className="text-xl font-bold text-foreground mb-4">Results</h2>
                  <ResultsDisplay results={results} loading={loading} algorithms={algorithms} />
                </div>
                <AlgorithmFormulas selectedAlgorithms={algorithms} />
              </div>
            </div>
          </TabsContent>

          {showConvertModal && (
            <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
              <Card className="p-6 w-full max-w-md space-y-4">
                <h2 className="text-xl font-bold text-foreground">Define Discrete Symbols</h2>
                <p className="text-sm text-muted-foreground">
                  Each symbol maps to an interval [lo, hi). Leave lo empty for −∞, hi empty for +∞.
                  B is computed as Φ((hi−μ)/σ) − Φ((lo−μ)/σ).
                </p>

                <div className="space-y-2">
                  {convertSymbols.map((sym, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <input
                          className="w-20 text-sm px-2 py-1 rounded border border-input bg-background text-foreground"
                          placeholder="Name"
                          value={sym.name}
                          onChange={e => {
                            const u = [...convertSymbols]; u[i] = { ...u[i], name: e.target.value }; setConvertSymbols(u);
                          }}
                        />
                        <span className="text-muted-foreground text-sm">[</span>
                        <input
                          className="w-20 text-sm px-2 py-1 rounded border border-input bg-background text-foreground"
                          placeholder="-∞"
                          value={sym.lo}
                          onChange={e => {
                            const u = [...convertSymbols]; u[i] = { ...u[i], lo: e.target.value }; setConvertSymbols(u);
                          }}
                        />
                        <span className="text-muted-foreground text-sm">,</span>
                        <input
                          className="w-20 text-sm px-2 py-1 rounded border border-input bg-background text-foreground"
                          placeholder="+∞"
                          value={sym.hi}
                          onChange={e => {
                            const u = [...convertSymbols]; u[i] = { ...u[i], hi: e.target.value }; setConvertSymbols(u);
                          }}
                        />
                        <span className="text-muted-foreground text-sm">)</span>
                        <Button variant="ghost" size="sm" onClick={() => setConvertSymbols(p => p.filter((_, j) => j !== i))}>×</Button>
                      </div>
                  ))}
                </div>

                <Button variant="outline" size="sm" className="w-full"
                  onClick={() => setConvertSymbols(p => [...p, { name: `S${p.length+1}`, lo: '', hi: '' }])}>
                  + Add Symbol
                </Button>

                <div className="flex gap-2 pt-2">
                  <Button className="flex-1" onClick={handleConvertConfirm}>Convert</Button>
                  <Button variant="outline" className="flex-1" onClick={() => setShowConvertModal(false)}>Cancel</Button>
                </div>
              </Card>
            </div>
          )}

          {showRetrieveModal && (
            <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
              <Card className="p-6 w-full max-w-sm space-y-4">
                <h2 className="text-xl font-bold text-foreground">Load Converted Parameters?</h2>
                <p className="text-sm text-muted-foreground">
                  Do you want to pre-fill the discrete HMM with the parameters from the conversion
                  (π, A, B, and discretized observation sequence)?
                </p>
                <div className="flex gap-2 pt-2">
                  <Button
                    className="flex-1"
                    onClick={() => {
                      setConvertKey(k => k + 1); // remount ParameterInput with initial values
                      setShowRetrieveModal(false);
                    }}
                  >
                    Yes, load them
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      // clear converted values so ParameterInput starts fresh
                      setConvertedB(null);
                      setConvertedPi(null);
                      setConvertedA(null);
                      setConvertedObs('');
                      setConvertKey(k => k + 1);
                      setShowRetrieveModal(false);
                    }}
                  >
                    No, start fresh
                  </Button>
                </div>
              </Card>
            </div>
          )}
        </Tabs>
      </div>
    </div>
  );
}