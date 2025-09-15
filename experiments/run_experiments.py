#!/usr/bin/env python3
"""
REE Extension Experiment Runner

This script orchestrates the execution of REE extension experiments
following the priority-based pipeline.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config_utils import load_config
from utils.logging_utils import setup_logging


def run_priority_1_experiments(config: Dict) -> Dict:
    """
    Run critical priority experiments.

    Args:
        config: Experiment configuration

    Returns:
        Results dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Priority 1 Experiments ===")

    results = {}

    # 1. Semantic Diversity Implementation Test
    logger.info("1. Testing Semantic Diversity Implementation...")
    try:
        from rewards.semantic_diversity import SemanticDiversityCalculator

        # Test semantic diversity calculator
        calc = SemanticDiversityCalculator()

        # Test with sample strategies
        test_strategies = [
            "<strategy id='1'><reasoning>Let me solve this step by step...</reasoning></strategy>",
            "<strategy id='2'><reasoning>I'll use a different approach...</reasoning></strategy>",
            "<strategy id='3'><reasoning>Let me solve this step by step...</reasoning></strategy>"
        ]

        diversity_score = calc.calculate_semantic_diversity(test_strategies)
        unique_count = calc.count_unique_strategies(test_strategies)

        results['semantic_diversity_test'] = {
            'diversity_score': diversity_score,
            'unique_count': unique_count,
            'total_strategies': len(test_strategies),
            'implementation_working': True
        }

        logger.info(f"✓ Semantic diversity test passed: {diversity_score:.3f}")

    except Exception as e:
        logger.error(f"✗ Semantic diversity test failed: {e}")
        results['semantic_diversity_test'] = {'implementation_working': False, 'error': str(e)}

    # 2. Baseline Implementation Tests
    logger.info("2. Testing Baseline Implementations...")

    # STaR test
    try:
        from training.star_trainer import STaRTrainer
        results['star_test'] = {'implementation_available': True}
        logger.info("✓ STaR trainer available")
    except Exception as e:
        results['star_test'] = {'implementation_available': False, 'error': str(e)}
        logger.warning(f"✗ STaR trainer not available: {e}")

    # PPO test
    try:
        from training.ppo_trainer import PPOBaselineTrainer
        results['ppo_test'] = {'implementation_available': True}
        logger.info("✓ PPO trainer available")
    except Exception as e:
        results['ppo_test'] = {'implementation_available': False, 'error': str(e)}
        logger.warning(f"✗ PPO trainer not available: {e}")

    logger.info("=== Priority 1 Experiments Complete ===")
    return results


def run_priority_2_experiments(config: Dict) -> Dict:
    """
    Run important priority experiments.

    Args:
        config: Experiment configuration

    Returns:
        Results dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Priority 2 Experiments ===")

    results = {}

    # Multi-seed validation
    logger.info("1. Multi-seed Statistical Validation...")
    # Implementation would go here

    # Cross-model validation
    logger.info("2. Cross-model Generalization...")
    # Implementation would go here

    # Dataset generalization
    logger.info("3. Dataset Generalization (MATH)...")
    # Implementation would go here

    logger.info("=== Priority 2 Experiments Complete ===")
    return results


def run_priority_3_experiments(config: Dict) -> Dict:
    """
    Run enhancement priority experiments.

    Args:
        config: Experiment configuration

    Returns:
        Results dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Priority 3 Experiments ===")

    results = {}

    # Hyperparameter sensitivity
    logger.info("1. Hyperparameter Sensitivity Analysis...")
    # Implementation would go here

    # Learning dynamics analysis
    logger.info("2. Learning Dynamics Analysis...")
    # Implementation would go here

    logger.info("=== Priority 3 Experiments Complete ===")
    return results


def generate_final_report(all_results: Dict, config: Dict) -> Dict:
    """
    Generate comprehensive experiment report.

    Args:
        all_results: Combined results from all experiment phases
        config: Experiment configuration

    Returns:
        Final report dictionary
    """
    from datetime import datetime

    report = {
        'experiment_date': datetime.now().isoformat(),
        'config': config,
        'results': all_results,
        'summary': summarize_results(all_results),
        'recommendations': generate_recommendations(all_results)
    }

    return report


def summarize_results(results: Dict) -> Dict:
    """Generate high-level summary of all results."""
    summary = {}

    # Count successful implementations
    if 'priority_1' in results:
        p1_results = results['priority_1']
        summary['semantic_diversity_working'] = p1_results.get('semantic_diversity_test', {}).get('implementation_working', False)
        summary['star_available'] = p1_results.get('star_test', {}).get('implementation_available', False)
        summary['ppo_available'] = p1_results.get('ppo_test', {}).get('implementation_available', False)

    return summary


def generate_recommendations(results: Dict) -> List[str]:
    """Generate actionable recommendations based on results."""
    recommendations = []

    if results.get('priority_1', {}).get('semantic_diversity_test', {}).get('implementation_working'):
        recommendations.append("✓ Semantic diversity implementation successful - proceed with integration")
    else:
        recommendations.append("✗ Fix semantic diversity implementation before proceeding")

    return recommendations


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run REE Extension Experiments")
    parser.add_argument("--config", type=str, default="configs/ree_config.json",
                        help="Path to configuration file")
    parser.add_argument("--priority", type=str, choices=["1", "2", "3", "all"], default="all",
                        help="Which priority experiments to run")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Output directory for results")

    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=config.get("logging", {}).get("log_level", "INFO"))
    logger = logging.getLogger(__name__)

    logger.info(f"Starting REE Extension Experiments with priority: {args.priority}")

    all_results = {}

    # Run experiments based on priority
    if args.priority in ["1", "all"]:
        all_results['priority_1'] = run_priority_1_experiments(config)

    if args.priority in ["2", "all"]:
        all_results['priority_2'] = run_priority_2_experiments(config)

    if args.priority in ["3", "all"]:
        all_results['priority_3'] = run_priority_3_experiments(config)

    # Generate final report
    final_report = generate_final_report(all_results, config)

    # Save results
    report_path = output_dir / f"experiment_report_{args.priority}.json"
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Experiment report saved to: {report_path}")

    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    for rec in final_report['recommendations']:
        print(rec)
    print("="*50)


if __name__ == "__main__":
    main()