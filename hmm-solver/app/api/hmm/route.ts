import { solveNumericalHMM } from '../../../lib/discrete_hmm.js';
import { solveContinuousHMM, discretizeContinuous } from '../../../lib/helping_functions.js';

export async function POST(request: Request) {
  try {
    const body = await request.json();

    if (body.type === 'numerical-discrete') {
      const result = solveNumericalHMM(
        body.algorithm,
        body.observation,
        body.pi,
        body.a,
        body.b,
        body.states,
        body.vocab,
        body.iterations ?? 5
      );

      return Response.json(result);
    }

    if (body.type === 'numerical-continuous') {
      const result = solveContinuousHMM(
        body.algorithm,
        body.observation,
        body.pi,
        body.a,
        body.means,
        body.sigmas_or_covs,
        body.states,
        body.dimension ?? '1d',
        body.iterations ?? 5
      );

      return Response.json(result);
    }

    if (body.type === 'convert-to-discrete') {
      const result = discretizeContinuous(
        body.observation,
        body.means,
        body.sigmas,
        body.states,
        body.symbols,
        (body.intervals ?? []).map(([lo, hi]) => [
          lo == null ? -Infinity : lo,
          hi == null ? Infinity : hi,
        ])
      );

      return Response.json(result);
    }

    return Response.json({ error: 'Invalid type' }, { status: 400 });
  } catch (error: any) {
    console.error('API error:', error);
    return Response.json(
      { error: 'Failed to process request', details: error.message },
      { status: 500 }
    );
  }
}
