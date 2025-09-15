# REE Extension: Enhanced Reasoning for Small Language Models

## Overview

This repository contains an extension to the REE (Reasoning Explore and Exploit) framework, implementing methodological improvements, rigorous evaluation, and theoretical grounding for enhanced reasoning capabilities in Small Language Models.

## Key Innovations

- **Semantic Diversity Metrics**: Replace XML tag counting with sentence embedding similarity measures
- **Comprehensive Baselines**: STaR, PPO, and enhanced GRPO comparisons
- **Statistical Rigor**: Multi-seed validation with confidence intervals and significance testing
- **Cross-Model Generalization**: Validation across Qwen, Llama, and Phi model families
- **Theoretical Grounding**: Formal multi-objective RL framework

## Project Structure

```
REE_Ext/
├── README.md                           # This file
├── REE_Extension_Pipeline.md           # Research pipeline and timeline
├── Implementation_Specifications.md    # Technical implementation details
├── src/                               # Source code
│   ├── rewards/                       # Reward system components
│   ├── training/                      # Training frameworks (GRPO, STaR, PPO)
│   ├── evaluation/                    # Evaluation and statistical testing
│   ├── models/                        # Model adapters and utilities
│   └── utils/                         # Utility functions
├── configs/                           # Configuration files
├── experiments/                       # Experiment scripts and results
├── data/                             # Dataset processing and storage
├── notebooks/                        # Analysis and visualization notebooks
├── tests/                            # Unit and integration tests
└── docs/                             # Additional documentation
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (16GB+ VRAM recommended)
- HuggingFace Transformers
- Sentence Transformers

### Installation

```bash
git clone git@github.com:Mishrakshitij/REE_Ext.git
cd REE_Ext
pip install -r requirements.txt
```

### Basic Usage

```python
from src.training.ree_trainer import REETrainer
from src.rewards.semantic_diversity import SemanticDiversityCalculator

# Initialize semantic diversity calculator
semantic_calc = SemanticDiversityCalculator()

# Setup REE trainer
trainer = REETrainer(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    semantic_calculator=semantic_calc,
    config_path="configs/ree_config.json"
)

# Train model
trainer.train(dataset_name="GSM8K", num_seeds=5)
```

## Research Pipeline

The project follows a structured 10-week research pipeline:

### Phase 1: Methodological Improvements (P1)
- **Week 1**: Semantic diversity metric implementation
- **Week 2-3**: STaR and PPO baseline implementations

### Phase 2: Statistical Rigor and Generalization (P2)
- **Week 4**: Multi-seed statistical validation
- **Week 5-6**: Cross-model validation (Qwen, Llama, Phi)
- **Week 7**: MATH dataset integration

### Phase 3: Analysis and Documentation (P1-P3)
- **Week 8-10**: Theoretical analysis, visualization, and paper writing

## Key Components

### Semantic Diversity Reward
```python
# Enhanced exploration reward using sentence embeddings
R_sd(strategies) = 1 - mean(cosine_similarity(embeddings))
```

### Multi-Objective Reward Framework
- **R_oc**: Outcome correctness
- **R_sd**: Semantic diversity (exploration)
- **R_re**: Reasoning exploitation
- **R_fa**: Format adherence

### Statistical Testing
- Multi-seed validation (5 seeds minimum)
- Paired t-tests with effect size calculation
- Bonferroni correction for multiple comparisons
- Power analysis for sample size determination

## Results Summary

*Results will be updated as experiments complete*

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/semantic-diversity`)
3. Commit your changes (`git commit -am 'Add semantic diversity metric'`)
4. Push to the branch (`git push origin feature/semantic-diversity`)
5. Create a Pull Request

## Citation

```bibtex
@article{ree_extension_2025,
  title={REE Extension: Semantic Diversity and Statistical Rigor in Small Language Model Reasoning},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original REE framework authors
- HuggingFace for model and tokenizer implementations
- Sentence Transformers for semantic similarity calculations
- Research community for theoretical foundations