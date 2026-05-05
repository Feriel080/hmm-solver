'use client';

import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const FORMULAS: Record<string, { name: string; description: string; formula: string }> = {
  forward: {
    name: 'Forward Algorithm',
    description:
      'Computes the probability of an observation sequence by calculating forward probabilities.',
    formula: `
\\alpha_1 = \\pi \\odot b(o_1)

\\alpha_t = A^T (\\alpha_{t-1} \\odot b(o_t)),   t = 2 ... T

P_\\lambda(O) = \\sum_{s} α_T(s)
    `,
  },
  backward: {
    name: 'Backward Algorithm',
    description:
      'Computes backward probabilities to find the probability of an observation sequence.',
    formula: `
β_T = 1

\\beta_t = A(b_{t + 1} \\odot β_{t + 1}), t = T - 1 ... 1

P_\\lambda(O) = \\pi^t \\cdot (b_1 \\odot β_1)
    `,
  },
  viterbi: {
    name: 'Viterbi Algorithm',
    description:
      'Finds the most likely state sequence given an observation sequence.',
    formula: `
\\delta_1 = \\pi \\odot b(o_1), \\psi_1 = 0

δ_t(i) = \\max_{1\\leq i\\leq N} [δ_{t-1}(i) \\cdot a_{ij}] \\odot b_j(o_t), t = 2 ... T

ψ_t(i) = \\arg\\max_{1\\leq i\\leq N} [δ_t(i) \\cdot a_{ij}]

P^* = \\max_{1\\leq i\\leq N} δ_T(i), q_T^* = \\arg\\max_{1\\leq i\\leq N} δ_T(i)

q_t^* = ψ_{t+1}(q_{t+1}^*), t = T - 1 ... 1

Q^* = {q_1^*, q_2^*, ..., q_T^*}
    `,
  },
  baum_welch: {
    name: 'Baum-Welch Algorithm (EM)',
    description:
      'Re-estimates HMM parameters using the Expectation-Maximization algorithm.',
    formula: `
γ_t = \\frac{α_t \\odot β_t}{α_t^t \\cdot β_t} = \\frac{α_t \\odot β_t}{P_\\lambda(O)}

[\\Xi_t]_{ij} = \\frac{α_t(i) \\cdot a_{ij} \\cdot b_j(o_{t+1}) \\cdot β_{t+1}(j)}{P_\\lambda(O)}

\\hat{π}(s) = γ_1(s)

\\hat{a}_{ij} = \\frac{\\sum_t [\\xi_t]_{ij}}{\\sum_t γ_t(i)}
\\hat{b}_j(o) = \\frac{\\sum_{t: o_t=o} γ_t(j)}{\\sum_t γ_t(j)}
    `,
  },
};

const renderFormulaWithKatex = (formula: string) => {
  // Split by double newlines to separate different equations
  const equations = formula.split('\n\n');

  return equations.map((equation, idx) => {
    // Skip empty lines
    if (!equation.trim()) return null;

    try {
      // Check if it's a multi-line case statement
      if (equation.includes('\\begin{cases}')) {
        // Split into lines for better display
        const lines = equation.split('\n');
        const renderedLines = lines.map(line => {
          if (line.trim() === '') return '';
          try {
            return katex.renderToString(line.trim(), {
              throwOnError: false,
              displayMode: false
            });
          } catch {
            return line;
          }
        });

        return (
          <div
            key={idx}
            className="my-4 overflow-x-auto"
            dangerouslySetInnerHTML={{
              __html: renderedLines.join('<br />')
            }}
          />
        );
      } else {
        // Render single line equations
        const html = katex.renderToString(equation.trim(), {
          throwOnError: false,
          displayMode: true  // Use display mode for better visibility
        });

        return (
          <div
            key={idx}
            className="my-4 overflow-x-auto"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
    } catch (error) {
      console.error('KaTeX error:', error);
      // Fallback to plain text
      return (
        <pre key={idx} className="text-xs my-2 whitespace-pre-wrap">
          {equation}
        </pre>
      );
    }
  });
};

export function AlgorithmFormulas({ selectedAlgorithms }: { selectedAlgorithms: string[] }) {
  const [expandedAlgorithms, setExpandedAlgorithms] = useState<Set<string>>(
    new Set(selectedAlgorithms)
  );

  const toggleExpanded = (algo: string) => {
    const newSet = new Set(expandedAlgorithms);
    if (newSet.has(algo)) {
      newSet.delete(algo);
    } else {
      newSet.add(algo);
    }
    setExpandedAlgorithms(newSet);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-foreground">Algorithm Formulas</h2>
      {Object.entries(FORMULAS).map(([key, { name, description, formula }]) => (
        <div
          key={key}
          className={`border rounded-lg overflow-hidden transition-colors ${selectedAlgorithms.includes(key)
            ? 'border-primary bg-primary/5'
            : 'border-border bg-background'
            }`}
        >
          <button
            onClick={() => toggleExpanded(key)}
            className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors"
          >
            <div className="text-left">
              <h3 className="font-semibold text-foreground">{name}</h3>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
            <ChevronDown
              size={20}
              className={`shrink-0 text-muted-foreground transition-transform ${expandedAlgorithms.has(key) ? 'rotate-180' : ''
                }`}
            />
          </button>
          {expandedAlgorithms.has(key) && (
            <div className="px-4 pb-4 pt-0 border-t border-border bg-muted/30">
              <div className="overflow-x-auto">
                <div className="bg-background p-4 rounded border border-border font-mono text-xs">
                  {renderFormulaWithKatex(formula)}
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
