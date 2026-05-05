'use client';

import { useState, useEffect, useRef } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';

interface ParameterInputProps {
  states: string[];
  vocab: string[];
  algorithms: string[];
  mode: 'discrete' | 'continuous-1d' | 'continuous-nd';
  onParametersChange: (params: any) => void;
  onConvertToDiscrete?: (params: any) => void;
  initialB?: number[][] | null;           
  initialObservation?: string;  
}

// ─── sensible defaults ────────────────────────────────────────────────────────
function makeDefaultPi(n: number): string[] {
  const val = (1 / n).toFixed(4);
  return Array(n).fill(val);
}

function makeDefaultA(n: number): string[][] {
  const val = (1 / n).toFixed(4);
  return Array.from({ length: n }, () => Array(n).fill(val));
}

function makeDefaultB(n: number, m: number): string[][] {
  const val = (1 / m).toFixed(4);
  return Array.from({ length: n }, () => Array(m).fill(val));
}

// ─── validation helpers ───────────────────────────────────────────────────────
const TOLERANCE = 0.01;

function rowSumError(row: string[]): boolean {
  const s = row.reduce((acc, v) => acc + (parseFloat(v) || 0), 0);
  return Math.abs(s - 1) > TOLERANCE;
}

function piSumError(pi: string[]): boolean {
  return rowSumError(pi);
}

// ─── component ────────────────────────────────────────────────────────────────
export function ParameterInput({
  states,
  vocab,
  algorithms,
  mode,
  onParametersChange,
  onConvertToDiscrete,
  initialB,           
  initialObservation, 
}: ParameterInputProps) {
  const N = states.length;
  const M = vocab.length;

  const [pi, setPi]           = useState<string[]>(() => makeDefaultPi(N));
  const [a,  setA]            = useState<string[][]>(() => makeDefaultA(N));
  const [b,  setB]            = useState<string[][]>(() => makeDefaultB(N, M));
  // continuous-1d per-state params
  const [means,  setMeans]    = useState<string[]>(() => Array.from({ length: N }, (_, i) => String(i)));
  const [sigmas, setSigmas]   = useState<string[]>(() => Array(N).fill('1.0'));
  // observation sequence
  const [observation, setObservation] = useState('');
  // Baum-Welch iterations
  const [bwIterations, setBwIterations] = useState(5);
  // submission state
  const [submitting, setSubmitting]     = useState(false);
  // whether user tried to submit (shows validation errors)
  const [attempted, setAttempted]       = useState(false);

  // ── listen for "convert to discrete" event from sidebar ───────────────────
  useEffect(() => {
    if (!onConvertToDiscrete) return;
    const handler = () => {
      onConvertToDiscrete(currentParamsRef.current);
    };
    window.addEventListener('hmm-request-convert-params', handler);
    return () => window.removeEventListener('hmm-request-convert-params', handler);
  }, [onConvertToDiscrete]);

  // ── resize matrices when dimensions change ──────────────────────────────────
  useEffect(() => {
    setPi(prev => {
      const next = makeDefaultPi(N);
      prev.slice(0, N).forEach((v, i) => { next[i] = v; });
      return next;
    });
    setA(prev => Array.from({ length: N }, (_, i) => {
      const row = makeDefaultA(N)[0];
      if (prev[i]) prev[i].slice(0, N).forEach((v, j) => { row[j] = v; });
      return row;
    }));
    setB(prev => Array.from({ length: N }, (_, i) => {
      const row = makeDefaultB(N, M)[0];
      if (prev[i]) prev[i].slice(0, M).forEach((v, j) => { row[j] = v; });
      return row;
    }));
    setMeans(prev => {
      const next = Array.from({ length: N }, (_, i) => String(i));
      prev.slice(0, N).forEach((v, i) => { next[i] = v; });
      return next;
    });
    setSigmas(prev => {
      const next = Array(N).fill('1.0');
      prev.slice(0, N).forEach((v, i) => { next[i] = v; });
      return next;
    });
  }, [N, M]);

  const currentParamsRef = useRef({
    pi: [] as number[],
    a: [] as number[][],
    means: [] as number[],
    sigmas: [] as number[],
    observation: [] as number[]
  });

  useEffect(() => {
    currentParamsRef.current.pi = pi.map(v => parseFloat(v) || 0);
  }, [pi]);
  useEffect(() => {
    currentParamsRef.current.a = a.map(row => row.map(v => parseFloat(v) || 0));
  }, [a]);
  useEffect(() => {
    currentParamsRef.current.means = means.map(v => parseFloat(v) || 0);
  }, [means]);
  useEffect(() => {
    currentParamsRef.current.sigmas = sigmas.map(v => parseFloat(v) || 1);
  }, [sigmas]);
  useEffect(() => {
    currentParamsRef.current.observation = observation.trim().split(/\s+/).filter(Boolean).map(Number);
  }, [observation]);

  useEffect(() => {
    if (initialB) { 
      setB(initialB.map(row => row.map((v: number) => String(parseFloat(v.toFixed(5))))));
    }
  }, [initialB, states.length, vocab.length]);  

  useEffect(() => {
    if (initialObservation !== undefined) {
      setObservation(initialObservation);
    }
  }, [initialObservation]);

  // ── cell class helper ───────────────────────────────────────────────────────
  const cellCls = (invalid: boolean) =>
    `h-8 text-xs text-center ${attempted && invalid ? 'border-red-500 bg-red-500/10 focus-visible:ring-red-500' : ''}`;

  // ── submit ──────────────────────────────────────────────────────────────────
  const handleSubmit = () => {
    setAttempted(true);

    // Validate Pi
    if (piSumError(pi)) return;
    // Validate A rows
    if (a.some(row => rowSumError(row))) return;
    // Validate B rows (only discrete mode)
    if (mode === 'discrete' && b.some(row => rowSumError(row))) return;

    const piNum  = pi.map(v => parseFloat(v) || 0);
    const aNum   = a.map(row => row.map(v => parseFloat(v) || 0));

    if (mode === 'discrete') {
      const obsArray = observation.trim().split(/\s+/).filter(Boolean);
      if (!obsArray.length) { alert('Enter at least one observation'); return; }
      setSubmitting(true);
      onParametersChange({
        type: 'discrete',
        pi: piNum,
        a: aNum,
        b: b.map(row => row.map(v => parseFloat(v) || 0)),
        observation: obsArray,
        bwIterations,
      });
      setSubmitting(false);

    } else if (mode === 'continuous-1d') {
      const obsArray = observation.trim().split(/\s+/).filter(Boolean).map(Number);
      if (!obsArray.length) { alert('Enter at least one observation value'); return; }
      setSubmitting(true);
      onParametersChange({
        type: 'continuous-1d',
        pi: piNum,
        a: aNum,
        means: means.map(v => parseFloat(v) || 0),
        sigmas: sigmas.map(v => parseFloat(v) || 1),
        observation: obsArray,
        bwIterations,
      });
      setSubmitting(false);
    }
  };

  const hasBW = algorithms.includes('baum_welch');

  // ── Pi sum indicator ────────────────────────────────────────────────────────
  const piSum = pi.reduce((acc, v) => acc + (parseFloat(v) || 0), 0);
  const piError = attempted && Math.abs(piSum - 1) > TOLERANCE;

  return (
    <div className="space-y-6">

      {/* ── π ─────────────────────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-foreground">Initial Probabilities (π)</h3>
          <span className={`text-xs font-mono ${piError ? 'text-red-500' : 'text-muted-foreground'}`}>
            Σ = {piSum.toFixed(4)} {piError && '— must equal 1'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {pi.map((value, i) => (
            <div key={i}>
              <label className="text-xs text-muted-foreground">{states[i]}</label>
              <Input
                type="number" step="0.01" min="0" max="1"
                value={value}
                onChange={e => { const n = [...pi]; n[i] = e.target.value; setPi(n); }}
                className={cellCls(piError)}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── A ─────────────────────────────────────────────────────────────── */}
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-2">Transition Matrix (A)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr>
                <th className="border border-border p-2 text-left bg-muted">From \ To</th>
                {states.map(s => (
                  <th key={s} className="border border-border p-2 text-center bg-muted">{s}</th>
                ))}
                <th className="border border-border p-2 bg-muted text-xs text-muted-foreground">Σ</th>
              </tr>
            </thead>
            <tbody>
              {a.map((row, i) => {
                const sum = row.reduce((acc, v) => acc + (parseFloat(v) || 0), 0);
                const err = attempted && Math.abs(sum - 1) > TOLERANCE;
                return (
                  <tr key={i}>
                    <td className="border border-border p-2 font-semibold bg-muted">{states[i]}</td>
                    {row.map((value, j) => (
                      <td key={j} className="border border-border p-1">
                        <Input
                          type="number" step="0.01" min="0" max="1"
                          value={value}
                          onChange={e => { const n = a.map(r => [...r]); n[i][j] = e.target.value; setA(n); }}
                          className={cellCls(err)}
                        />
                      </td>
                    ))}
                    <td className={`border border-border p-2 text-center text-xs font-mono ${err ? 'text-red-500 font-bold' : 'text-muted-foreground'}`}>
                      {sum.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── B (discrete only) ─────────────────────────────────────────────── */}
      {mode === 'discrete' && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-2">Emission Matrix (B)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr>
                  <th className="border border-border p-2 text-left bg-muted sticky left-0 z-10">State \ Obs</th>
                  {vocab.map(w => (
                    <th key={w} className="border border-border p-2 text-center bg-muted">{w}</th>
                  ))}
                  <th className="border border-border p-2 bg-muted text-xs text-muted-foreground">Σ</th>
                </tr>
              </thead>
              <tbody>
                {b.map((row, i) => {
                  const sum = row.reduce((acc, v) => acc + (parseFloat(v) || 0), 0);
                  const err = attempted && Math.abs(sum - 1) > TOLERANCE;
                  return (
                    <tr key={i}>
                      <td className="border border-border p-2 font-semibold bg-muted sticky left-0 z-10">{states[i]}</td>
                      {row.map((value, j) => (
                        <td key={j} className="border border-border p-1">
                          <Input
                            type="number" step="0.01" min="0" max="1"
                            value={value}
                            onChange={e => { const n = b.map(r => [...r]); n[i][j] = e.target.value; setB(n); }}
                            className={cellCls(err)}
                          />
                        </td>
                      ))}
                      <td className={`border border-border p-2 text-center text-xs font-mono ${err ? 'text-red-500 font-bold' : 'text-muted-foreground'}`}>
                        {sum.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Gaussian params (continuous-1d) ───────────────────────────────── */}
      {mode === 'continuous-1d' && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-2">
            Gaussian Emission Parameters
          </h3>
          <p className="text-xs text-muted-foreground mb-3">
            B is computed automatically from N(μ, σ²). Enter μ and σ per state.
          </p>
          <div className="space-y-2">
            {states.map((s, i) => (
              <div key={i} className="grid grid-cols-3 gap-2 items-center">
                <span className="text-sm font-medium text-foreground">{s}</span>
                <div>
                  <label className="text-xs text-muted-foreground">μ (mean)</label>
                  <Input
                    type="number" step="any"
                    value={means[i]}
                    onChange={e => { const n = [...means]; n[i] = e.target.value; setMeans(n); }}
                    className="h-8 text-xs text-center"
                  />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">σ (std dev)</label>
                  <Input
                    type="number" step="any" min="0.001"
                    value={sigmas[i]}
                    onChange={e => { const n = [...sigmas]; n[i] = e.target.value; setSigmas(n); }}
                    className="h-8 text-xs text-center"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Observation sequence ──────────────────────────────────────────── */}
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-2">Observation Sequence</h3>
        <Input
          placeholder={
            mode === 'discrete'
              ? `Symbols separated by spaces (e.g. ${vocab.slice(0, 3).join(' ')})`
              : 'Numeric values separated by spaces (e.g. 1.2 3.4 0.8)'
          }
          value={observation}
          onChange={e => setObservation(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') handleSubmit(); }}
        />
        {mode === 'discrete' && (
          <p className="text-xs text-muted-foreground mt-1">
            Available symbols: {vocab.join(', ')}
          </p>
        )}
      </div>

      {/* ── Baum-Welch iterations ─────────────────────────────────────────── */}
      {hasBW && (
        <div className="flex items-center gap-4 p-3 rounded-lg bg-muted/50 border border-border">
          <label className="text-sm font-medium text-foreground whitespace-nowrap">
            Baum-Welch iterations
          </label>
          <input
            type="range" min={1} max={50} step={1}
            value={bwIterations}
            onChange={e => setBwIterations(Number(e.target.value))}
            className="flex-1 accent-primary"
          />
          <span className="text-sm font-mono font-bold text-primary w-6 text-right">
            {bwIterations}
          </span>
        </div>
      )}

      {/* ── Submit ────────────────────────────────────────────────────────── */}
      <Button onClick={handleSubmit} className="w-full" size="lg" disabled={submitting}>
        {submitting ? (
          <>
            <Spinner className="mr-2 h-4 w-4" />
            Computing...
          </>
        ) : (
          `Compute${algorithms.length > 1 ? ` (${algorithms.length} algorithms)` : ''}`
        )}
      </Button>
    </div>
  );
}