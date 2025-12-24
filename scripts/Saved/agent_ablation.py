"""
PRISM Agent Ablation Study
Compare: Agent-routed engines vs Direct engine execution
"""

from prism_geometry_signature_agent import GeometrySignatureAgent
from prism_engine_gates import EngineAuditor, ENGINE_GATES

def run_with_agents(data, domain):
    """Full agent pipeline - geometry detection → routing → weighted engines"""
    agent = GeometrySignatureAgent()
    profile = agent.analyze(data)
    recommendations = agent.recommend_engines(profile)
    
    results = {}
    for rec in recommendations:
        # Only run if weight > threshold
        if rec.weight > 0.3:
            output = run_engine(rec.engine, data)  # your engine runner
            results[rec.engine] = {
                'output': output,
                'weight': rec.weight,
                'reason': rec.reason
            }
    return results, profile

def run_without_agents(data, domain):
    """Direct execution - all engines, no routing, equal weights"""
    all_engines = ['pca', 'correlation', 'entropy', 'wavelets', 
                   'hmm', 'dmd', 'mutual_information']
    
    results = {}
    for engine in all_engines:
        output = run_engine(engine, data)  # same engine runner
        results[engine] = {
            'output': output,
            'weight': 1.0,  # equal weight
            'reason': 'direct_execution'
        }
    return results

def compare_outputs(with_agents, without_agents):
    """Compare the two approaches"""
    comparison = {
        'engines_skipped_by_agents': [],
        'weight_differences': {},
        'output_correlation': {}
    }
    # ... comparison logic
    return comparison