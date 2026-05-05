'use client';

import { Card } from '@/components/ui/card';
import { Spinner } from '@/components/ui/spinner';

interface ResultsDisplayProps {
  results: any;
  loading: boolean;
  algorithms: string[];
}

const ALGO_LABELS: Record<string, string> = {
  forward: 'Forward Algorithm',
  backward: 'Backward Algorithm',
  viterbi: 'Viterbi Algorithm',
  baum_welch: 'Baum-Welch Algorithm',
};

function GaussianPlot({
  means, sigmas, states, observations,
}: {
  means: number[]; sigmas: number[]; states: string[]; observations?: number[];
}) {
  const W = 480, H = 200, PAD = 30;
  const plotW = W - PAD * 2, plotH = H - PAD * 2;
  const allMeans = means ?? [], allSigmas = sigmas ?? [];
  const xMin = Math.min(...allMeans.map((m, i) => m - 3 * (allSigmas[i] ?? 1)));
  const xMax = Math.max(...allMeans.map((m, i) => m + 3 * (allSigmas[i] ?? 1)));
  const xRange = xMax - xMin || 1;
  const gaussian = (x: number, mu: number, sigma: number) =>
    (1 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
  const STEPS = 200;
  const xs = Array.from({ length: STEPS + 1 }, (_, k) => xMin + (k / STEPS) * xRange);
  const yMax = Math.max(...allMeans.map((m, i) => gaussian(m, m, allSigmas[i] ?? 1))) * 1.1;
  const toSvgX = (x: number) => PAD + ((x - xMin) / xRange) * plotW;
  const toSvgY = (y: number) => PAD + plotH - (y / yMax) * plotH;
  const COLORS = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#06b6d4'];
  const paths = allMeans.map((mu, i) => {
    const sigma = allSigmas[i] ?? 1;
    const pts = xs.map(x => `${toSvgX(x).toFixed(1)},${toSvgY(gaussian(x, mu, sigma)).toFixed(1)}`);
    return `M ${pts.join(' L ')}`;
  });

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-lg text-xs font-mono">
        <line x1={PAD} y1={PAD} x2={PAD} y2={PAD + plotH} stroke="currentColor" strokeOpacity="0.3" />
        <line x1={PAD} y1={PAD + plotH} x2={PAD + plotW} y2={PAD + plotH} stroke="currentColor" strokeOpacity="0.3" />
        {paths.map((d, i) => <path key={i} d={d} fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth="2" />)}
        {allMeans.map((mu, i) => (
          <g key={i}>
            <line x1={toSvgX(mu)} y1={PAD} x2={toSvgX(mu)} y2={PAD + plotH}
              stroke={COLORS[i % COLORS.length]} strokeWidth="1" strokeDasharray="4 2" strokeOpacity="0.6" />
            <text x={toSvgX(mu)} y={PAD - 4} textAnchor="middle" fill={COLORS[i % COLORS.length]} fontSize="9">
              {states[i]}
            </text>
          </g>
        ))}
        {observations?.map((o, idx) => (
          <circle key={idx} cx={toSvgX(o)} cy={PAD + plotH} r={4} fill="#f43f5e" opacity="0.8" />
        ))}
        {[xMin, (xMin + xMax) / 2, xMax].map((v, i) => (
          <text key={i} x={toSvgX(v)} y={PAD + plotH + 14} textAnchor="middle" fill="currentColor" fontSize="8" opacity="0.6">
            {v.toFixed(1)}
          </text>
        ))}
        {allMeans.map((mu, i) => (
          <g key={i} transform={`translate(${PAD + plotW - 90}, ${PAD + i * 14})`}>
            <rect width="20" height="3" y="5" fill={COLORS[i % COLORS.length]} rx="1" />
            <text x="24" y="11" fill="currentColor" fontSize="8" opacity="0.8">
              {states[i]}: μ={mu.toFixed(2)}, σ={allSigmas[i]?.toFixed(2)}
            </text>
          </g>
        ))}
      </svg>
      {observations && observations.length > 0 && (
        <p className="text-xs text-muted-foreground mt-1">● red dots = observations on x-axis</p>
      )}
    </div>
  );
}

function computeGaussianB(obsValues: number[], means: number[], sigmas: number[]): number[][] {
  const gaussian = (o: number, mu: number, sigma: number) =>
    (1 / (Math.sqrt(2 * Math.PI * sigma))) * Math.exp(-0.5 * ((o - mu)**2 / sigma));

  means.map((mu, s) =>{
    obsValues.map(o => {
      gaussian(o, mu, sigmas[s] ?? 1)})})
  return means.map((mu, s) => obsValues.map(o => gaussian(o, mu, sigmas[s] ?? 1)));
}

function MatrixTable({ table, states, title }: { table: number[][]; states: string[]; title: string }) {
  return (
    <div>
      <h3 className="text-lg font-semibold text-foreground mb-4">{title}</h3>
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="border border-border p-2 bg-muted">t \ State</th>
              {states.map(s => <th key={s} className="border border-border p-2 bg-muted text-center">{s}</th>)}
            </tr>
          </thead>
          <tbody>
            {table.map((row, t) => (
              <tr key={t}>
                <td className="border border-border p-2 bg-muted font-semibold">{t}</td>
                {row.map((val, s) => (
                  <td key={s} className="border border-border p-2 text-center hover:bg-muted/50">{val.toFixed(6)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GaussianEmissionCard({ r }: { r: any }) {
  const obs = Array.isArray(r.observation_sequence) ? r.observation_sequence as number[] : [];
  return (
    <Card className="p-6 space-y-6">
      <h2 className="text-xl font-bold text-foreground border-b border-border pb-2">
        Emission Parameters
      </h2>

      {obs.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-1">Observation Sequence</h3>
          <p className="text-muted-foreground text-sm">{obs.join(' → ')}</p>
        </div>
      )}

      <div>
        <h3 className="text-lg font-semibold text-foreground mb-3">Emission Distributions</h3>
        <GaussianPlot
          means={r.means}
          sigmas={r.sigmas}
          states={r.states ?? []}
          observations={obs.map(Number)}
        />
      </div>

      {obs.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-3">
            Emission Matrix B
          </h3>
          <div className="overflow-x-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="border border-border p-2 bg-muted">State \ obs</th>
                  {obs.map((o, t) => (
                    <th key={t} className="border border-border p-2 bg-muted text-center">
                      {Number(o).toFixed(2)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {computeGaussianB(obs, r.means, r.sigmas).map((row, s) => (
                  <tr key={s}>
                    <td className="border border-border p-2 bg-muted font-semibold">
                      {r.states?.[s] ?? `s${s}`}
                    </td>
                    {row.map((val, t) => (
                      <td
                        key={t}
                        className="border border-border p-2 text-center hover:bg-muted/50 font-mono"
                        style={{ backgroundColor: `hsl(var(--primary) / ${Math.min(val * 40, 0.35)})` }}
                      >
                        {val.toFixed(4)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}

function AlgoResult({ algo, r }: { algo: string; r: any }) {
  if (r?.error) {
    return (
      <div className="rounded border border-red-500/50 bg-red-500/5 p-4">
        <p className="text-red-500 text-sm">{r.error}</p>
        {r.traceback && <pre className="text-xs mt-2 overflow-auto max-h-48">{r.traceback}</pre>}
      </div>
    );
  }
  if (!r) return <p className="text-muted-foreground text-sm">No result returned.</p>;

  const isGaussian = r.emission_type === 'gaussian_1d';

  return (
    <div className="space-y-6">
      {/* ── forward ── */}
      {algo === 'forward' && (
        <>
          {r.probability !== undefined && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-1">Probability of Sequence</h3>
              <p className="text-2xl font-bold text-primary">{r.probability.toFixed(8)}</p>
            </div>
          )}
          {r.alpha_table && <MatrixTable table={r.alpha_table} states={r.states ?? []} title="Alpha Table (Forward)" />}
        </>
      )}

      {/* ── backward ── */}
      {algo === 'backward' && (
        <>
          {r.probability !== undefined && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-1">Probability of Sequence</h3>
              <p className="text-2xl font-bold text-primary">{r.probability.toFixed(8)}</p>
            </div>
          )}
          {r.beta_table && <MatrixTable table={r.beta_table} states={r.states ?? []} title="Beta Table (Backward)" />}
        </>
      )}

      {/* ── viterbi ── */}
      {algo === 'viterbi' && (
        <>
          {r.tagged_pairs && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3">Most Likely Path</h3>
              <div className="space-y-2">
                {r.tagged_pairs.map(([word, tag]: [string | number, string], idx: number) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded">
                    <span className="font-medium text-foreground">{String(word)}</span>
                    <span className="text-primary font-semibold">{tag}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {r.probability !== undefined && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-1">Path Probability</h3>
              <p className="text-2xl font-bold text-primary">{r.probability.toFixed(8)}</p>
            </div>
          )}
          {r.viterbi_table && <MatrixTable table={r.viterbi_table} states={r.states ?? []} title="Viterbi Table" />}
        </>
      )}

      {/* ── baum-welch ── */}
      {algo === 'baum_welch' && (
        <>
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-1">Iterations</h3>
            <p className="text-muted-foreground">Completed {r.iterations} EM iterations</p>
          </div>

          {isGaussian && r.updated_means && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3">Updated Emission Distributions</h3>
              <GaussianPlot means={r.updated_means} sigmas={r.updated_sigmas ?? []} states={r.states ?? []} />
              <div className="mt-3 space-y-1">
                {r.updated_means.map((mu: number, i: number) => (
                  <div key={i} className="flex justify-between p-2 bg-muted rounded text-sm">
                    <span className="text-muted-foreground">{r.states?.[i]}</span>
                    <span className="font-mono">μ = {mu.toFixed(4)}, σ = {r.updated_sigmas?.[i]?.toFixed(4)}</span>
                  </div>
                ))}
              </div>
              {/* Updated B matrix */}
              {Array.isArray(r.observation_sequence) && (
                <div className="mt-4">
                  <h4 className="font-medium text-foreground mb-2 text-sm">Re-estimated Emission Matrix B̂</h4>
                  <div className="overflow-x-auto">
                    <table className="text-xs border-collapse">
                      <thead>
                        <tr>
                          <th className="border border-border p-2 bg-muted">State \ obs</th>
                          {(r.observation_sequence as number[]).map((o, t) => (
                            <th key={t} className="border border-border p-2 bg-muted text-center">
                              {Number(o).toFixed(2)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {computeGaussianB(
                          (r.observation_sequence as number[]).map(Number),
                          r.updated_means,
                          r.updated_sigmas ?? []
                        ).map((row, s) => (
                          <tr key={s}>
                            <td className="border border-border p-2 bg-muted font-semibold">
                              {r.states?.[s] ?? `s${s}`}
                              <span className="block text-xs font-normal text-muted-foreground">
                                μ̂={r.updated_means[s].toFixed(2)}, σ̂={r.updated_sigmas?.[s]?.toFixed(2)}
                              </span>
                            </td>
                            {row.map((val, t) => (
                              <td
                                key={t}
                                className="border border-border p-2 text-center hover:bg-muted/50 font-mono"
                                style={{ backgroundColor: `hsl(var(--primary) / ${Math.min(val * 40, 0.35)})` }}
                              >
                                {val.toExponential(3)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {!isGaussian && r.updated_pi && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3">Updated Parameters</h3>
              <div className="space-y-3">
                <div>
                  <h4 className="font-medium text-foreground mb-1 text-sm">Initial Probabilities (π̂)</h4>
                  {r.updated_pi.map((val: number, idx: number) => (
                    <div key={idx} className="flex justify-between p-2 bg-muted rounded text-sm">
                      <span className="text-muted-foreground">{r.states?.[idx]}</span>
                      <span className="font-mono font-semibold">{val.toFixed(6)}</span>
                    </div>
                  ))}
                </div>
                {r.updated_a && (
                  <div>
                    <h4 className="font-medium text-foreground mb-1 text-sm">Transition Matrix (Â)</h4>
                    <div className="overflow-x-auto text-xs">
                      <table className="border-collapse">
                        <thead>
                          <tr>
                            <th className="border border-border p-2 bg-muted">From \ To</th>
                            {r.states?.map((s: string) => (
                              <th key={s} className="border border-border p-2 bg-muted text-center">{s}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {r.updated_a.map((row: number[], i: number) => (
                            <tr key={i}>
                              <td className="border border-border p-2 bg-muted font-semibold">{r.states?.[i]}</td>
                              {row.map((val: number, j: number) => (
                                <td key={j} className="border border-border p-2 text-center">{val.toFixed(6)}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {r.likelihood_history && (
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3">Likelihood Progress</h3>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {r.likelihood_history.map((ll: number, idx: number) => (
                  <div key={idx} className="flex justify-between p-2 bg-muted rounded text-sm">
                    <span className="text-muted-foreground">Iteration {idx + 1}</span>
                    <span className="font-mono font-semibold">{ll.toFixed(8)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── main export ──────────────────────────────────────────────────────────────
export function ResultsDisplay({ results, loading, algorithms }: ResultsDisplayProps) {
  if (loading) {
    return (
      <Card className="p-8 flex items-center justify-center min-h-48">
        <div className="flex flex-col items-center gap-4">
          <Spinner />
          <p className="text-muted-foreground">
            Computing {algorithms.map(a => ALGO_LABELS[a] ?? a).join(', ')}...
          </p>
        </div>
      </Card>
    );
  }

  if (!results) {
    return (
      <Card className="p-8">
        <p className="text-muted-foreground text-center">
          No results yet. Enter parameters and click Compute.
        </p>
      </Card>
    );
  }

  if (results.error) {
    return (
      <Card className="p-8 border-red-500/50 bg-red-500/5">
        <p className="text-red-600 font-semibold">Error:</p>
        <p className="text-red-500 text-sm mt-2">{results.error}</p>
        {results.details && (
          <pre className="text-xs bg-background p-4 rounded mt-4 overflow-auto max-h-96">
            {results.details}
          </pre>
        )}
      </Card>
    );
  }

  // Check if this is a continuous Gaussian result using the first algo's result
  const firstResult = results[algorithms[0]];
  const isGaussian  = firstResult?.emission_type === 'gaussian_1d';

  return (
    <div className="space-y-6">
      {/* Shared Gaussian card — once, above all algo cards */}
      {isGaussian && firstResult?.means && firstResult?.sigmas && (
        <GaussianEmissionCard r={firstResult} />
      )}

      {/* Per-algorithm result cards */}
      {algorithms.map(algo => (
        <Card key={algo} className="p-6 space-y-6">
          <h2 className="text-xl font-bold text-foreground border-b border-border pb-2">
            {ALGO_LABELS[algo] ?? algo}
          </h2>
          <AlgoResult algo={algo} r={results[algo]} />
        </Card>
      ))}
    </div>
  );
}