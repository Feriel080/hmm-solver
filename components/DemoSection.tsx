'use client';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Spinner } from '@/components/ui/spinner';
import { useState } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface DemoSectionProps {
  onRun: () => void;
}

const DEFAULT_PHRASE = 'la petite brise la glace';

export function DemoSection({ onRun }: DemoSectionProps) {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [showPhraseInput, setShowPhraseInput] = useState(false);
  const [customPhrase, setCustomPhrase] = useState('');
  const [activePhrase, setActivePhrase] = useState(DEFAULT_PHRASE);

  const handleRun = async (phraseOverride?: string) => {
    const phraseToUse = phraseOverride ?? activePhrase;
    setLoading(true);
    try {
      const response = await fetch('/api/hmm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'demo', sentence: phraseToUse }),
      });
      const data = await response.json();
      setResults(data);
      onRun();
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyPhrase = () => {
    const trimmed = customPhrase.trim();
    if (!trimmed) return;
    setActivePhrase(trimmed);
    setShowPhraseInput(false);
    setCustomPhrase('');
    handleRun(trimmed);
  };

  const handleCancelPhrase = () => {
    setShowPhraseInput(false);
    setCustomPhrase('');
  };

  const renderMatrix = (matrix: any, title: string) => {
        if (!matrix || !Array.isArray(matrix) || matrix.length === 0) {
            return null;
        }

        let matrix2D = matrix;
        if (!Array.isArray(matrix[0])) {
            matrix2D = [matrix]; 
        }

        const matrixRows = matrix2D.map(row => {
            if (!Array.isArray(row)) return '';
            return row.map(value => Number(value).toFixed(2)).join(' & ');
        }).join(' \\\\ ');

        const latexString = `
            ${title ? title + ' = ' : ''} 
            \\begin{pmatrix}
                ${matrixRows}
            \\end{pmatrix}
        `;

        return (
            <div className='matrix-container'>
                <div
                    className='renderMatrix'
                    dangerouslySetInnerHTML={{ __html: katex.renderToString(latexString, { throwOnError: false, displayMode: false }) }}
                />
            </div>
        );
    };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-primary/5 border-primary/30">
        <div className="space-y-4">
          <div>
            <h3 className="text-xl font-semibold text-foreground mb-2">
              Ambiguous French Phrase Analysis
            </h3>
            {/* Active phrase display */}
            <div className="flex items-center gap-2 mb-3">
              <p className="text-muted-foreground text-sm">Analyzing:</p>
              <span className="font-semibold text-foreground italic">
                &quot;{activePhrase}&quot;
              </span>
            </div>

            <p className="text-muted-foreground mb-4">
              The phrase &quot;la petite brise la glace&quot; is ambiguous in French:
            </p>
            <ul className="space-y-2 text-sm text-muted-foreground list-disc list-inside">
              <li>
                &quot;petite&quot; could be an <strong>adjective</strong>, as it could be a <strong>noun</strong>. 
              </li>
              <li>
                &quot;brise&quot; could be a <strong>verb</strong>, as it could be a <strong>noun</strong>. 
              </li>
              <li>
                &quot;la&quot; could be a <strong>determiner</strong>, as it could be a <strong>pronoun</strong>.
              </li>
            </ul>
          </div>

          {/* Custom phrase input — revealed on demand */}
          {showPhraseInput && (
            <div className="space-y-2 pt-1">
              <label className="text-sm font-medium text-foreground">
                Enter a custom phrase to analyze:
              </label>
              <p className="text-xs text-muted-foreground">
                Note: words outside the demo vocabulary (
                <em>la, petite, brise, glace</em>) will map to index 0 and may
                produce unexpected results.
              </p>
              <input
                type="text"
                value={customPhrase}
                onChange={(e) => setCustomPhrase(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleApplyPhrase();
                  if (e.key === 'Escape') handleCancelPhrase();
                }}
                placeholder="e.g. la glace brise la brise"
                className="w-full px-3 py-2 rounded-md border border-input bg-background text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                autoFocus
              />
              <div className="flex gap-2">
                <Button
                  onClick={handleApplyPhrase}
                  disabled={!customPhrase.trim() || loading}
                  size="sm"
                  className="flex-1"
                >
                  Analyze this phrase
                </Button>
                <Button
                  onClick={handleCancelPhrase}
                  variant="outline"
                  size="sm"
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}

          {/* OOV warning */}
          {results?.oov_words?.length > 0 && (
            <div className="rounded-md border border-yellow-400/50 bg-yellow-400/10 px-3 py-2 text-xs text-yellow-700 dark:text-yellow-300">
              ⚠️ Out-of-vocabulary words detected:{' '}
              <strong>{results.oov_words.join(', ')}</strong>. These were
              mapped to a default index and results may be unreliable.
            </div>
          )}
              
          <Button onClick={() => handleRun()} disabled={loading} size="lg" className="flex-1">
            {loading ? (
              <>
                <Spinner className="mr-2" />
                Running Viterbi Algorithm...
              </>
            ) : (
              'Find best hidden state'
            )}
          </Button>
          <Button
              onClick={() => setShowPhraseInput((v) => !v)}
              disabled={loading}
              variant="outline"
              size="lg"
            >
              Change phrase
            </Button>
        </div>
      </Card>

      {results && (
        <Card className="p-6">
          {results.corpus && (
              <div>
                <h3 className="text-lg font-semibold text-foreground mb-4">HMM Parameters</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <p className="font-semibold text-muted-foreground mb-2">States: {results.corpus.states.join(', ')}</p>
                    <p className="font-semibold text-muted-foreground mb-2">Vocabulary: {results.corpus.vocab.join(', ')}</p>
                  </div>
                  <div>
                    <p className="font-semibold text-muted-foreground mb-2">Initial vector:</p>
                    {renderMatrix(results.corpus.pi, '\\pi')}
                  </div>
                  <div>
                    <p className="font-semibold text-muted-foreground mb-2">Transition matrix A:</p>
                    {renderMatrix(results.corpus.a, 'A')}
                  </div>
                  <div>
                    <p className="font-semibold text-muted-foreground mb-2">Emission matrix B:</p>
                    {renderMatrix(results.corpus.b, 'B')}
                  </div>
                </div>
              </div>
            )}

          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Viterbi Results</h3>
            <div>
              <h4 className="font-medium text-foreground mb-3">Most Likely Tagging</h4>
              <div className="space-y-2">
                {results.tagged_pairs?.map(([word, tag]: [string, string], idx: number) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-muted rounded">
                    <span className="font-medium text-foreground">{word}</span>
                    <span className="px-3 py-1 bg-primary/10 text-primary font-semibold rounded text-sm">
                      {tag}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-medium text-foreground mb-2">Path Probability</h4>
              <p className="text-3xl font-bold text-primary">{results.probability?.toFixed(8)}</p>
            </div>
          </div>

          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Baum-Welch Results in {results.baum_welch.iterations} iterations.</h3>
            <div>
                <p className="font-semibold text-muted-foreground mb-2">Re-estimated initial vector:</p>
                {renderMatrix(results.baum_welch.updated_pi, '\\hat{\\pi}')}
              </div>
              <div>
                <p className="font-semibold text-muted-foreground mb-2">Re-estimated transition matrix:</p>
                {renderMatrix(results.baum_welch.updated_a, '\\hat{A}')}
              </div>
              <div>
                <p className="font-semibold text-muted-foreground mb-2">Re-estimated emission matrix :</p>
                {renderMatrix(results.baum_welch.updated_b, '\\hat{B}')}
              </div>
              <div>
                <p className="font-semibold text-muted-foreground mb-2">Probability history:</p>
                {results.baum_welch.likelihood_history?.map((likelihood: number, idx: number) => (
                  <p key={idx} className="text-sm text-muted-foreground">
                    Iteration {idx + 1}: {likelihood.toFixed(8)}
                  </p>
                ))}
              </div>
          </div>
        </Card>
      )}
    </div>
  );
}
