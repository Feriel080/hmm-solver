import sys
import json
import traceback
import math
from hmm_algorithms import solve_numerical_hmm, solve_continuous_hmm, discretize_continuous

try:
    data = json.loads(sys.argv[1])
 
    if data['type'] == 'numerical-discrete':
        result = solve_numerical_hmm(
            data['algorithm'],
            data['observation'],
            data['pi'],
            data['a'],
            data['b'],
            data['states'],
            data['vocab'],
        )
        
    elif data['type'] == 'numerical-continuous':
        result = solve_continuous_hmm(
            algorithm = data['algorithm'],
            obs_values = data['observation'],
            Pi = data['pi'],
            A = data['a'],
            means = data['means'],
            sigmas_or_covs = data['sigmas_or_covs'],
            states = data['states'],
            dimension = data.get('dimension', '1d'),
            iterations = data.get('iterations', 5),
        )
    
    elif data['type'] == 'convert-to-discrete':
        results = discretize_continuous(
            obs_values = data['observation'],
            means      = data['means'],
            sigmas     = data['sigmas'],
            states     = data['states'],
            symbols    = data['symbols'],
            intervals  = [
                (lo if lo is not None else -math.inf,
                hi if hi is not None else  math.inf)
                for lo, hi in data['intervals']
        ],
        )
        result = {
            'discrete_observation': results['discrete_observation'],
            'vocab': results['vocab'],
            'b': results['b_discrete'],
            'pi': data['pi'],
            'a': data['a'],
            'states': results['states'],
        }
         
    else:
        result = {'error': 'Invalid type'}
    
    print(json.dumps(result, default=str))
    
except Exception as e:
    error_details = {
        'error': str(e),
        'traceback': traceback.format_exc()
    }
    print(json.dumps(error_details, default=str))
