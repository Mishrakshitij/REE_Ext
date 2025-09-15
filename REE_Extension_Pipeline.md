# REE Extension Research Pipeline

## Executive Summary

This document outlines a comprehensive research pipeline to extend and strengthen the REE (Reasoning Explore and Exploit) framework with methodological improvements, rigorous evaluation, and theoretical grounding. The pipeline prioritizes high-impact changes within computational constraints (140-hour training limit).

## Priority Classification

**P1 (Critical - Must Have)**: Methodological soundness and essential baselines
**P2 (Important - Should Have)**: Generalization and statistical rigor
**P3 (Enhancement - Nice to Have)**: Additional analyses and theoretical depth

---

## Phase 1: Methodological Improvements (P1)

### 1.1 Semantic Diversity Metric (P1 - Week 1)

**Current Issue**: XML tag counting is superficial and non-robust
**Solution**: Replace with semantic similarity-based diversity measurement

#### Implementation Plan:
```python
# Semantic Diversity Reward (R_sd)
def compute_semantic_diversity(strategies, embeddings_model):
    """
    Replace simple strategy counting with semantic similarity measurement
    """
    strategy_embeddings = []
    for strategy in strategies:
        reasoning_text = extract_reasoning_text(strategy)
        embedding = embeddings_model.encode(reasoning_text)
        strategy_embeddings.append(embedding)

    # Compute pairwise cosine similarities
    similarities = compute_pairwise_similarities(strategy_embeddings)

    # Diversity = 1 - average similarity
    diversity_score = 1 - np.mean(similarities)
    return diversity_score

# New reward formulation
R_sd(a|q) = γ_diversity * semantic_diversity_score(S(a))
```

#### Technical Details:
- **Embedding Model**: Use sentence-transformers/all-MiniLM-L6-v2 (lightweight)
- **Similarity Threshold**: Strategies with >0.8 similarity considered redundant
- **Integration**: Replace current R_rd diversity component
- **Hyperparameter**: γ_diversity ∈ [0.1, 0.5] (grid search)

#### Expected Outcome:
- More meaningful exploration reward
- Reduced redundant strategy generation
- Stronger theoretical foundation

---

### 1.2 Critical Baseline Implementation (P1 - Week 2-3)

#### 1.2.1 STaR (Self-Taught Reasoner) Baseline
**Why Critical**: Most directly comparable methodology in reasoning improvement

```python
# STaR Implementation Pipeline
class STaRTrainer:
    def __init__(self, base_model, datasets):
        self.model = base_model
        self.datasets = datasets

    def generate_rationales(self, questions):
        # Generate CoT reasoning for correct answers
        rationales = []
        for q in questions:
            rationale = self.model.generate_cot_reasoning(q)
            if self.verify_correctness(rationale, q.answer):
                rationales.append((q, rationale))
        return rationales

    def fine_tune_on_rationales(self, rationales):
        # Standard supervised fine-tuning on generated rationales
        return self.supervised_training(rationales)
```

**Training Setup**:
- Same base model: Qwen2.5-3B-Instruct
- Same datasets: GSM8K, MedMCQA
- Same computational budget: ~140 hours
- Generate rationales → Filter correct ones → Fine-tune

#### 1.2.2 PPO with Standard Rewards Baseline
```python
# PPO Baseline Configuration
ppo_rewards = {
    "outcome_correctness": 1.0,
    "length_penalty": -0.01,
    "format_adherence": 0.3
}
# No process-level exploration/exploitation rewards
```

#### 1.2.3 GRPO-CFL (Keep as Ablation)
- Maintain current implementation for direct ablation studies

---

## Phase 2: Statistical Rigor and Generalization (P2)

### 2.1 Multi-Seed Experimental Design (P2 - Week 4)

#### Statistical Framework:
```python
# Experimental Design
SEEDS = [42, 123, 456, 789, 1337]  # 5 random seeds
CONFIDENCE_LEVEL = 0.95
MIN_EFFECT_SIZE = 2.0  # Minimum meaningful accuracy improvement

def statistical_evaluation(method_results, baseline_results):
    """
    Perform statistical significance testing
    """
    # Paired t-test for accuracy comparisons
    t_stat, p_value = scipy.stats.ttest_rel(method_results, baseline_results)

    # Effect size (Cohen's d)
    effect_size = (np.mean(method_results) - np.mean(baseline_results)) / np.std(baseline_results)

    # Confidence intervals
    ci_lower, ci_upper = scipy.stats.t.interval(
        CONFIDENCE_LEVEL, len(method_results)-1,
        loc=np.mean(method_results),
        scale=scipy.stats.sem(method_results)
    )

    return {
        'p_value': p_value,
        'effect_size': effect_size,
        'confidence_interval': (ci_lower, ci_upper),
        'significant': p_value < 0.05 and effect_size > MIN_EFFECT_SIZE
    }
```

#### Reporting Format:
- **Mean ± Std Dev (95% CI)**
- **Effect size (Cohen's d)**
- **Statistical significance (p < 0.05)**

### 2.2 Cross-Model Generalization (P2 - Week 5-6)

#### Target Models:
1. **Llama-3.2-3B-Instruct** (Primary alternative)
2. **Phi-3.5-mini-instruct** (Secondary if resources allow)

#### Evaluation Protocol:
```python
models_to_test = [
    "Qwen2.5-3B-Instruct",    # Original
    "Llama-3.2-3B-Instruct", # Primary generalization test
    "Phi-3.5-mini-instruct"  # Secondary (if time permits)
]

for model in models_to_test:
    results = train_and_evaluate(
        model=model,
        method="REE",
        seeds=SEEDS,
        datasets=["GSM8K", "MedMCQA"]
    )
    statistical_analysis(results)
```

### 2.3 Dataset Generalization (P2 - Week 7)

#### MATH Dataset Integration (Priority)
```python
# MATH dataset preparation
def prepare_math_dataset():
    """
    Adapt MATH dataset to REE multi-strategy format
    """
    math_data = load_dataset("competition_math")

    # Convert to multi-strategy prompt format
    formatted_prompts = []
    for item in math_data:
        prompt = create_multi_strategy_prompt(
            problem=item['problem'],
            expected_format="mathematical_reasoning"
        )
        formatted_prompts.append(prompt)

    return formatted_prompts
```

#### MMLU (If resources allow)
- Focus on MMLU-STEM subjects for reasoning evaluation
- Sample subset for computational efficiency

---

## Phase 3: Theoretical Grounding (P1-P2 - Week 8)

### 3.1 Multi-Objective RL Formalization

#### Theoretical Framework:
```latex
% Formal problem definition
\mathcal{P} = \langle \mathcal{S}, \mathcal{A}, \mathcal{R}^{(1)}, ..., \mathcal{R}^{(k)}, \mathcal{T}, \gamma \rangle

% Where:
% S: State space (prompt + partial generation)
% A: Action space (next token generation)
% R^(i): Individual reward functions (outcome, exploration, exploitation, format)
% T: Transition function (autoregressive generation)
% γ: Discount factor
```

#### Key Theoretical Contributions:
1. **Exploration-Exploitation Decomposition**:
   - Formalize why separating exploration (R_rd) and exploitation (R_re) helps reasoning
   - Connect to bandit literature and multi-armed bandits in sequence generation

2. **Convergence Properties**:
   - Analyze GRPO convergence with multi-objective rewards
   - Bounded reward functions ensure convergence guarantees

### 3.2 Connection to Existing Literature

#### Process Reward Models:
- Relate R_rd and R_re to process supervision (Lightman et al., 2023)
- Distinguish from outcome-only supervision

#### Tree of Thoughts:
- Compare multi-strategy generation to ToT tree search
- Position REE as learned exploration vs. explicit search

---

## Phase 4: Presentation and Analysis (P2-P3)

### 4.1 Framing Corrections

#### Terminology Updates:
- **"Zero-shot"** → **"Intrinsic reasoning without inference-time prompting"**
- **"Small Language Model"** → **"Parameter-efficient Language Model"** (more precise)
- **"State-of-the-art"** → **"Competitive performance"** (more conservative)

### 4.2 Key Visualizations

#### Figure 1: Comparative Reasoning Analysis
```python
def create_comparison_figure():
    """
    Side-by-side comparison showing:
    - Baseline failure case
    - REE success case
    - Strategy diversity visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Baseline reasoning (single path, incorrect)
    # Middle: REE reasoning (multiple paths, correct path found)
    # Right: Semantic diversity heatmap
```

#### Figure 2: Learning Dynamics
```python
def plot_exploration_exploitation_dynamics():
    """
    Show evolution of:
    - Exploration reward over training
    - Exploitation reward over training
    - Final accuracy
    """
    # Track R_rd, R_re, and accuracy across training steps
```

### 4.3 Hyperparameter Analysis (P3)

#### Sensitivity Analysis:
```python
# 8 Lambda parameters to analyze
lambda_params = {
    'λ₁': 'outcome_correctness_weight',
    'λ₂': 'exploration_correct_bonus',
    'λ₃': 'exploration_diversity_cap',
    'λ₄': 'exploration_diversity_rate',
    'λ₅': 'exploitation_weight',
    'λ₆': 'format_strategy_weight',
    'λ₇': 'format_answer_weight',
    'λ₈': 'format_completeness_bonus'
}

def hyperparameter_sensitivity_study():
    """
    Grid search over key parameters with reduced ranges
    """
    # Focus on most impactful parameters: λ₂, λ₅, λ₆
    key_params = ['λ₂', 'λ₅', 'λ₆']
    # Test ±50% from default values
```

---

## Implementation Timeline (10 Weeks)

| Week | Phase | Task | Priority | Resources |
|------|-------|------|----------|-----------|
| 1 | P1 | Semantic Diversity Implementation | Critical | 1 GPU-week |
| 2-3 | P1 | STaR + PPO Baselines | Critical | 2 GPU-weeks |
| 4 | P2 | Multi-seed REE Training (5 seeds) | Important | 5 GPU-weeks |
| 5-6 | P2 | Cross-model Validation (Llama-3.2) | Important | 2 GPU-weeks |
| 7 | P2 | MATH Dataset Integration | Important | 1 GPU-week |
| 8 | P1-P2 | Theoretical Analysis + Writing | Critical | 0 GPU-weeks |
| 9 | P3 | Hyperparameter Analysis | Enhancement | 1 GPU-week |
| 10 | P3 | Final Analysis + Paper Writing | Enhancement | 0 GPU-weeks |

**Total GPU Requirements**: ~12-13 GPU-weeks (within 140-hour constraint per experiment)

---

## Risk Mitigation

### Computational Constraints:
- **Fallback 1**: If multi-model testing exceeds budget, focus on Qwen + Llama-3.2 only
- **Fallback 2**: If 5 seeds too expensive, use 3 seeds minimum
- **Fallback 3**: If MATH dataset too large, sample 2K problems

### Technical Risks:
- **Semantic Embedding Integration**: Test on small scale first
- **Baseline Reproduction**: Allocate buffer time for STaR implementation
- **Statistical Power**: Pre-compute required effect sizes

---

## Success Metrics

### Primary Success Criteria:
1. **Methodological Soundness**: Semantic diversity metric improves theoretical foundation
2. **Empirical Rigor**: Statistical significance across multiple seeds and models
3. **Baseline Dominance**: Outperform STaR and PPO baselines significantly
4. **Generalization**: Results hold across ≥2 model families and ≥2 datasets

### Secondary Success Criteria:
1. **Theoretical Contribution**: Clear connection to multi-objective RL theory
2. **Practical Impact**: Demonstrate token efficiency vs. accuracy trade-offs
3. **Interpretability**: Clear failure mode analysis and reasoning quality assessment

---

## Next Steps

1. **Immediate (Week 1)**: Implement semantic diversity metric
2. **Short-term (Week 2-4)**: Establish rigorous baselines and multi-seed framework
3. **Medium-term (Week 5-8)**: Cross-model and cross-dataset validation
4. **Long-term (Week 9-10)**: Analysis, writing, and submission preparation

This pipeline ensures methodological rigor while respecting computational constraints, positioning the REE extension as a significant contribution to reasoning in language models.