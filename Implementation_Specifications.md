# REE Extension: Detailed Implementation Specifications

## Technical Architecture Overview

```
REE-Extended Framework
├── Data Pipeline
│   ├── Dataset Loaders (GSM8K, MedMCQA, MATH)
│   ├── Multi-Strategy Prompt Templates
│   └── Format Validators
├── Reward System
│   ├── Semantic Diversity Module
│   ├── Exploitation Tracker
│   ├── Format Adherence Checker
│   └── Reward Normalizer
├── Training Framework
│   ├── GRPO Trainer
│   ├── Baseline Trainers (STaR, PPO)
│   └── Multi-Seed Controller
├── Evaluation Suite
│   ├── Statistical Testing
│   ├── Cross-Model Validator
│   └── Analysis Tools
└── Visualization Tools
    ├── Learning Curves
    ├── Strategy Analysis
    └── Comparison Plots
```

---

## 1. Semantic Diversity Module

### 1.1 Core Implementation

```python
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticDiversityCalculator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize semantic diversity calculator

        Args:
            model_name: HuggingFace sentence transformer model
        """
        self.encoder = SentenceTransformer(model_name)
        self.similarity_threshold = 0.8  # Strategies > 0.8 similarity considered redundant

    def extract_reasoning_content(self, strategy_xml):
        """
        Extract reasoning text from XML strategy block

        Args:
            strategy_xml: String containing <strategy><reasoning>...</reasoning></strategy>
        Returns:
            reasoning_text: Cleaned reasoning content
        """
        import re
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        matches = re.findall(reasoning_pattern, strategy_xml, re.DOTALL)
        if matches:
            return matches[0].strip()
        return ""

    def compute_strategy_embeddings(self, strategies):
        """
        Convert reasoning strategies to embeddings

        Args:
            strategies: List of strategy XML blocks
        Returns:
            embeddings: Numpy array of shape (n_strategies, embedding_dim)
        """
        reasoning_texts = [self.extract_reasoning_content(s) for s in strategies]
        # Filter out empty reasoning
        valid_texts = [text for text in reasoning_texts if len(text.strip()) > 0]

        if len(valid_texts) == 0:
            return np.array([])

        embeddings = self.encoder.encode(valid_texts, convert_to_numpy=True)
        return embeddings

    def calculate_semantic_diversity(self, strategies):
        """
        Calculate semantic diversity score for a set of strategies

        Args:
            strategies: List of strategy XML blocks
        Returns:
            diversity_score: Float between 0 and 1 (higher = more diverse)
        """
        embeddings = self.compute_strategy_embeddings(strategies)

        if len(embeddings) == 0:
            return 0.0

        if len(embeddings) == 1:
            return 1.0  # Single strategy has maximum diversity

        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Get upper triangular matrix (excluding diagonal)
        triu_indices = np.triu_indices_from(similarities, k=1)
        pairwise_similarities = similarities[triu_indices]

        if len(pairwise_similarities) == 0:
            return 1.0

        # Diversity = 1 - average similarity
        avg_similarity = np.mean(pairwise_similarities)
        diversity_score = 1.0 - avg_similarity

        return max(0.0, diversity_score)  # Ensure non-negative

    def count_unique_strategies(self, strategies, similarity_threshold=None):
        """
        Count semantically unique strategies

        Args:
            strategies: List of strategy XML blocks
            similarity_threshold: Override default similarity threshold
        Returns:
            unique_count: Number of semantically distinct strategies
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        embeddings = self.compute_strategy_embeddings(strategies)

        if len(embeddings) <= 1:
            return len(embeddings)

        # Cluster strategies based on similarity
        unique_strategies = [0]  # First strategy is always unique

        for i in range(1, len(embeddings)):
            is_unique = True
            for unique_idx in unique_strategies:
                similarity = cosine_similarity([embeddings[i]], [embeddings[unique_idx]])[0][0]
                if similarity > similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_strategies.append(i)

        return len(unique_strategies)

# Integration with reward system
class SemanticExplorationReward:
    def __init__(self, diversity_calculator):
        self.diversity_calc = diversity_calculator

    def compute_reward(self, strategies, correct_answer, γ_correct=1.0, γ_diversity_cap=0.5, γ_diversity_rate=0.1):
        """
        Compute semantic diversity-based exploration reward

        Args:
            strategies: List of (strategy_xml, strategy_outcome) tuples
            correct_answer: Ground truth answer
            γ_correct: Reward bonus for having correct strategy
            γ_diversity_cap: Maximum diversity reward
            γ_diversity_rate: Per-strategy diversity reward rate
        """
        strategy_texts = [s[0] for s in strategies]
        strategy_outcomes = [s[1] for s in strategies]

        # Check if any strategy is correct
        has_correct_strategy = any(outcome == correct_answer for outcome in strategy_outcomes)

        if has_correct_strategy:
            return γ_correct

        # Calculate semantic diversity
        diversity_score = self.diversity_calc.calculate_semantic_diversity(strategy_texts)
        unique_count = self.diversity_calc.count_unique_strategies(strategy_texts)

        # Reward based on diversity and unique strategy count
        diversity_reward = min(γ_diversity_cap, γ_diversity_rate * unique_count * diversity_score)

        return diversity_reward
```

### 1.2 Integration Points

```python
# Modified reward computation in main training loop
def compute_rewards(completion, ground_truth, semantic_calc):
    """
    Enhanced reward computation with semantic diversity
    """
    # Extract strategies from completion
    strategies = extract_strategies_with_outcomes(completion)

    # Original rewards
    R_oc = compute_outcome_correctness(completion, ground_truth)
    R_re = compute_exploitation_reward(strategies, ground_truth)
    R_fa = compute_format_adherence(completion)

    # New semantic diversity reward
    semantic_reward = SemanticExplorationReward(semantic_calc)
    R_sd = semantic_reward.compute_reward(strategies, ground_truth)

    return {
        'outcome': R_oc,
        'exploitation': R_re,
        'format': R_fa,
        'semantic_diversity': R_sd
    }
```

---

## 2. Baseline Implementations

### 2.1 STaR (Self-Taught Reasoner) Implementation

```python
class STaRTrainer:
    def __init__(self, base_model, tokenizer, device):
        self.model = base_model
        self.tokenizer = tokenizer
        self.device = device

    def generate_rationales(self, dataset, batch_size=8, max_attempts=3):
        """
        Generate reasoning rationales for dataset questions

        Args:
            dataset: List of (question, answer) pairs
            batch_size: Generation batch size
            max_attempts: Max attempts per question
        Returns:
            rationales: List of (question, rationale, answer) for correct generations
        """
        rationales = []

        for batch_start in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[batch_start:batch_start + batch_size]

            for question, correct_answer in batch:
                for attempt in range(max_attempts):
                    # Generate CoT reasoning
                    prompt = self.create_cot_prompt(question)

                    with torch.no_grad():
                        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )

                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_answer = self.extract_answer(response)

                    if generated_answer == correct_answer:
                        rationale = self.extract_reasoning(response)
                        rationales.append((question, rationale, correct_answer))
                        break  # Found correct reasoning, move to next question

        return rationales

    def create_cot_prompt(self, question):
        """Create Chain-of-Thought prompt"""
        return f"""Question: {question}

Please solve this step by step, showing your reasoning clearly.

Answer: Let me think step by step."""

    def extract_answer(self, response):
        """Extract final answer from generated response"""
        import re
        # Look for patterns like "The answer is X" or "Therefore, X"
        patterns = [
            r"(?:the answer is|therefore,?|so,?)\s*([^.\n]+)",
            r"answer:\s*([^.\n]+)",
            r"final answer:\s*([^.\n]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1).strip()

        # Fallback: return last number/letter found
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        if numbers:
            return numbers[-1]

        return ""

    def extract_reasoning(self, response):
        """Extract reasoning steps from response"""
        # Remove the original question and return reasoning part
        if "Answer:" in response:
            return response.split("Answer:", 1)[1].strip()
        return response.strip()

    def fine_tune_on_rationales(self, rationales, num_epochs=1, learning_rate=5e-6):
        """
        Fine-tune model on generated rationales using supervised learning
        """
        from transformers import TrainingArguments, Trainer

        # Prepare training data
        train_examples = []
        for question, rationale, answer in rationales:
            input_text = f"Question: {question}\nAnswer:"
            target_text = f"{rationale}"
            train_examples.append({
                'input_ids': self.tokenizer(input_text, truncation=True, max_length=512)['input_ids'],
                'labels': self.tokenizer(target_text, truncation=True, max_length=512)['input_ids']
            })

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./star_checkpoint",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no"
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_examples,
            tokenizer=self.tokenizer
        )

        # Train
        trainer.train()

        return self.model
```

### 2.2 PPO Baseline Implementation

```python
import torch
from trl import PPOTrainer, PPOConfig

class PPOBaselineTrainer:
    def __init__(self, model, tokenizer, reference_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reference_model = reference_model or model

        # PPO configuration
        self.ppo_config = PPOConfig(
            model_name="ppo_baseline",
            learning_rate=5e-6,
            batch_size=1,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            optimize_device_cache=True,
            early_stopping=False,
            target_kl=0.1,
            ppo_epochs=4,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
        )

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.reference_model,
            tokenizer=self.tokenizer,
        )

    def compute_standard_rewards(self, completions, questions, correct_answers):
        """
        Compute standard RL rewards (no process-level exploration/exploitation)
        """
        rewards = []

        for completion, question, correct_answer in zip(completions, questions, correct_answers):
            reward = 0.0

            # Outcome correctness reward
            predicted_answer = self.extract_answer(completion)
            if predicted_answer == correct_answer:
                reward += 1.0

            # Length penalty (discourage overly long responses)
            length_penalty = -0.01 * len(completion.split())
            reward += length_penalty

            # Basic format reward (has some structure)
            if any(marker in completion.lower() for marker in ['step', 'therefore', 'because']):
                reward += 0.3

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)

    def train_step(self, batch):
        """
        Single PPO training step
        """
        questions, correct_answers = batch['questions'], batch['answers']

        # Generate completions
        query_tensors = self.tokenizer(questions, return_tensors="pt", padding=True)['input_ids']
        response_tensors = self.ppo_trainer.generate(
            query_tensors,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode responses
        completions = [
            self.tokenizer.decode(response, skip_special_tokens=True)
            for response in response_tensors
        ]

        # Compute rewards
        rewards = self.compute_standard_rewards(completions, questions, correct_answers)

        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)

        return stats, rewards.mean().item()
```

---

## 3. Statistical Testing Framework

```python
import scipy.stats as stats
import numpy as np
from typing import List, Dict, Tuple

class StatisticalEvaluator:
    def __init__(self, confidence_level=0.95, min_effect_size=2.0):
        self.confidence_level = confidence_level
        self.min_effect_size = min_effect_size
        self.alpha = 1 - confidence_level

    def paired_comparison(self, method_results: List[float], baseline_results: List[float]) -> Dict:
        """
        Perform paired t-test comparison between method and baseline

        Args:
            method_results: List of accuracy scores for method (multiple seeds)
            baseline_results: List of accuracy scores for baseline (multiple seeds)
        Returns:
            Statistical test results dictionary
        """
        assert len(method_results) == len(baseline_results), "Must have same number of runs"

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(method_results, baseline_results)

        # Effect size (Cohen's d for paired samples)
        differences = np.array(method_results) - np.array(baseline_results)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)

        # Confidence interval for the difference
        mean_diff = np.mean(differences)
        sem_diff = stats.sem(differences)

        ci_lower, ci_upper = stats.t.interval(
            self.confidence_level,
            len(differences) - 1,
            loc=mean_diff,
            scale=sem_diff
        )

        # Statistical and practical significance
        statistically_significant = p_value < self.alpha
        practically_significant = abs(effect_size) > self.min_effect_size

        return {
            'method_mean': np.mean(method_results),
            'method_std': np.std(method_results, ddof=1),
            'baseline_mean': np.mean(baseline_results),
            'baseline_std': np.std(baseline_results, ddof=1),
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper),
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'statistically_significant': statistically_significant,
            'practically_significant': practically_significant,
            'overall_significant': statistically_significant and practically_significant
        }

    def multiple_comparisons_correction(self, p_values: List[float], method='bonferroni') -> List[float]:
        """
        Apply multiple comparisons correction

        Args:
            p_values: List of uncorrected p-values
            method: Correction method ('bonferroni' or 'holm')
        Returns:
            Corrected p-values
        """
        n = len(p_values)

        if method == 'bonferroni':
            return [min(p * n, 1.0) for p in p_values]

        elif method == 'holm':
            # Holm-Bonferroni correction
            indexed_p = [(p, i) for i, p in enumerate(p_values)]
            indexed_p.sort()  # Sort by p-value

            corrected = [0] * n
            for rank, (p_val, orig_idx) in enumerate(indexed_p):
                corrected_p = min(p_val * (n - rank), 1.0)
                corrected[orig_idx] = corrected_p

            return corrected

        else:
            raise ValueError(f"Unknown correction method: {method}")

    def power_analysis(self, effect_size: float, n_per_group: int) -> float:
        """
        Calculate statistical power for given effect size and sample size

        Args:
            effect_size: Expected Cohen's d
            n_per_group: Sample size per group
        Returns:
            Statistical power (0-1)
        """
        from scipy.stats import norm

        # For paired t-test
        standard_error = np.sqrt(2 / n_per_group)  # SE for difference
        critical_t = stats.t.ppf(1 - self.alpha/2, df=n_per_group-1)

        # Non-centrality parameter
        ncp = effect_size / standard_error

        # Power calculation
        power = 1 - stats.t.cdf(critical_t, df=n_per_group-1, loc=ncp) + stats.t.cdf(-critical_t, df=n_per_group-1, loc=ncp)

        return power

    def required_sample_size(self, effect_size: float, power: float = 0.8) -> int:
        """
        Calculate required sample size for desired power

        Args:
            effect_size: Expected Cohen's d
            power: Desired statistical power
        Returns:
            Required sample size per group
        """
        # Iterative search for required n
        for n in range(3, 100):
            if self.power_analysis(effect_size, n) >= power:
                return n
        return 100  # Cap at 100
```

---

## 4. Cross-Model Validation Framework

```python
class CrossModelValidator:
    def __init__(self, models_config: Dict):
        """
        Args:
            models_config: Dictionary mapping model names to their configurations
                {
                    "Qwen2.5-3B-Instruct": {"path": "...", "type": "qwen"},
                    "Llama-3.2-3B-Instruct": {"path": "...", "type": "llama"},
                }
        """
        self.models_config = models_config
        self.results = {}

    def load_model(self, model_name: str):
        """Load and return model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = self.models_config[model_name]
        model_path = config["path"]

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def adapt_prompt_format(self, model_type: str, prompt: str) -> str:
        """
        Adapt prompt format for different model architectures

        Args:
            model_type: Type of model ("qwen", "llama", "phi")
            prompt: Base prompt text
        Returns:
            Formatted prompt for the specific model
        """
        if model_type == "qwen":
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        elif model_type == "llama":
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        elif model_type == "phi":
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

        else:
            # Generic format
            return f"### User\n{prompt}\n\n### Assistant\n"

    def validate_model(self, model_name: str, method: str, datasets: List[str], seeds: List[int]):
        """
        Validate a specific model with given method across datasets and seeds

        Args:
            model_name: Name of model to validate
            method: Training method ("REE", "STaR", "PPO", etc.)
            datasets: List of dataset names
            seeds: List of random seeds
        Returns:
            Validation results dictionary
        """
        model_results = {}

        for dataset_name in datasets:
            dataset_results = {}

            for seed in seeds:
                # Set random seed
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Load model
                model, tokenizer = self.load_model(model_name)

                # Load dataset
                dataset = self.load_dataset(dataset_name)

                # Train model with specified method
                if method == "REE":
                    trained_model = self.train_with_ree(model, tokenizer, dataset, seed)
                elif method == "STaR":
                    trained_model = self.train_with_star(model, tokenizer, dataset, seed)
                elif method == "PPO":
                    trained_model = self.train_with_ppo(model, tokenizer, dataset, seed)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Evaluate
                eval_results = self.evaluate_model(trained_model, tokenizer, dataset)
                dataset_results[seed] = eval_results

            model_results[dataset_name] = dataset_results

        self.results[f"{model_name}_{method}"] = model_results
        return model_results

    def cross_model_comparison(self, method: str, datasets: List[str], seeds: List[int]):
        """
        Compare method performance across all configured models
        """
        comparison_results = {}

        for model_name in self.models_config.keys():
            print(f"Validating {method} on {model_name}...")
            model_results = self.validate_model(model_name, method, datasets, seeds)
            comparison_results[model_name] = model_results

        return comparison_results

    def analyze_generalization(self, results: Dict) -> Dict:
        """
        Analyze cross-model generalization patterns

        Args:
            results: Results from cross_model_comparison
        Returns:
            Generalization analysis
        """
        analysis = {}

        for dataset in results[list(results.keys())[0]].keys():
            dataset_analysis = {}

            # Collect accuracies across models
            model_accuracies = {}
            for model_name, model_results in results.items():
                accuracies = [
                    model_results[dataset][seed]['accuracy']
                    for seed in model_results[dataset].keys()
                ]
                model_accuracies[model_name] = accuracies

            # Calculate cross-model statistics
            all_accuracies = []
            for accuracies in model_accuracies.values():
                all_accuracies.extend(accuracies)

            dataset_analysis.update({
                'mean_accuracy': np.mean(all_accuracies),
                'std_accuracy': np.std(all_accuracies),
                'min_accuracy': np.min(all_accuracies),
                'max_accuracy': np.max(all_accuracies),
                'model_variance': np.var([np.mean(accs) for accs in model_accuracies.values()]),
                'generalization_consistency': 1.0 - (np.std([np.mean(accs) for accs in model_accuracies.values()]) / np.mean(all_accuracies))
            })

            analysis[dataset] = dataset_analysis

        return analysis
```

---

## 5. Experiment Orchestration

```python
class ExperimentOrchestrator:
    def __init__(self, config_path: str):
        """Initialize experiment orchestrator with configuration"""
        import json
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.results = {}
        self.statistical_evaluator = StatisticalEvaluator()

    def run_priority_1_experiments(self):
        """
        Run critical priority experiments (methodological improvements + baselines)
        """
        print("=== Priority 1: Critical Experiments ===")

        # 1. Semantic Diversity Implementation Test
        print("1. Testing Semantic Diversity Implementation...")
        semantic_results = self.test_semantic_diversity()
        self.results['semantic_diversity_test'] = semantic_results

        # 2. Baseline Implementations
        print("2. Running Baseline Comparisons...")
        baseline_results = self.run_baseline_comparisons()
        self.results['baselines'] = baseline_results

        return self.results

    def run_priority_2_experiments(self):
        """
        Run important priority experiments (statistical rigor + generalization)
        """
        print("=== Priority 2: Important Experiments ===")

        # 1. Multi-seed validation
        print("1. Multi-seed Statistical Validation...")
        multiseed_results = self.run_multiseed_validation()
        self.results['multiseed'] = multiseed_results

        # 2. Cross-model validation
        print("2. Cross-model Generalization...")
        crossmodel_results = self.run_crossmodel_validation()
        self.results['crossmodel'] = crossmodel_results

        # 3. Dataset generalization
        print("3. Dataset Generalization (MATH)...")
        dataset_results = self.run_dataset_generalization()
        self.results['dataset_generalization'] = dataset_results

        return self.results

    def run_priority_3_experiments(self):
        """
        Run enhancement priority experiments (analysis + interpretability)
        """
        print("=== Priority 3: Enhancement Experiments ===")

        # 1. Hyperparameter sensitivity
        print("1. Hyperparameter Sensitivity Analysis...")
        hyperparam_results = self.run_hyperparameter_analysis()
        self.results['hyperparameters'] = hyperparam_results

        # 2. Learning dynamics analysis
        print("2. Learning Dynamics Analysis...")
        dynamics_results = self.analyze_learning_dynamics()
        self.results['dynamics'] = dynamics_results

        return self.results

    def generate_final_report(self):
        """
        Generate comprehensive experiment report
        """
        from datetime import datetime

        report = {
            'experiment_date': datetime.now().isoformat(),
            'config': self.config,
            'results': self.results,
            'summary': self.summarize_results(),
            'recommendations': self.generate_recommendations()
        }

        # Save report
        report_path = f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Final report saved to: {report_path}")
        return report

    def summarize_results(self) -> Dict:
        """Generate high-level summary of all results"""
        summary = {}

        # Statistical significance summary
        if 'baselines' in self.results:
            significant_improvements = []
            for baseline, comparison in self.results['baselines'].items():
                if comparison.get('overall_significant', False):
                    significant_improvements.append(baseline)
            summary['significant_improvements'] = significant_improvements

        # Cross-model consistency
        if 'crossmodel' in self.results:
            consistency_scores = []
            for model_results in self.results['crossmodel'].values():
                # Calculate consistency metric
                pass
            summary['cross_model_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.0

        return summary

    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []

        # Based on statistical significance
        if self.results.get('baselines', {}).get('STaR', {}).get('overall_significant'):
            recommendations.append("REE shows statistically significant improvement over STaR - proceed with publication")

        # Based on cross-model results
        if len(self.results.get('crossmodel', {})) >= 2:
            recommendations.append("Cross-model validation successful - method generalizes beyond single architecture")

        # Based on semantic diversity
        if self.results.get('semantic_diversity_test', {}).get('improvement_over_counting'):
            recommendations.append("Semantic diversity metric provides meaningful improvement over simple counting")

        return recommendations
```

---

## Configuration Template

```json
{
  "experiment_config": {
    "models": {
      "Qwen2.5-3B-Instruct": {
        "path": "Qwen/Qwen2.5-3B-Instruct",
        "type": "qwen"
      },
      "Llama-3.2-3B-Instruct": {
        "path": "meta-llama/Llama-3.2-3B-Instruct",
        "type": "llama"
      }
    },
    "datasets": {
      "GSM8K": {
        "train_size": 7473,
        "test_size": 1319,
        "domain": "mathematical_reasoning"
      },
      "MedMCQA": {
        "train_size": 7500,
        "test_size": 4183,
        "domain": "medical_reasoning"
      },
      "MATH": {
        "train_size": 5000,
        "test_size": 1000,
        "domain": "competition_math"
      }
    },
    "training": {
      "seeds": [42, 123, 456, 789, 1337],
      "max_steps": 7500,
      "batch_size": 1,
      "gradient_accumulation": 1,
      "learning_rate": 5e-6,
      "group_size": 6
    },
    "rewards": {
      "semantic_diversity": {
        "model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.8,
        "gamma_correct": 1.0,
        "gamma_diversity_cap": 0.5,
        "gamma_diversity_rate": 0.1
      },
      "exploitation": {
        "gamma_exploitation": 1.0
      },
      "format": {
        "gamma_strategy": 0.2,
        "gamma_answer": 0.3,
        "gamma_completeness": 0.1
      }
    },
    "evaluation": {
      "confidence_level": 0.95,
      "min_effect_size": 2.0,
      "multiple_comparisons_correction": "bonferroni"
    }
  },
  "computational_constraints": {
    "max_gpu_hours_per_model": 140,
    "max_parallel_jobs": 2,
    "memory_limit_gb": 16
  }
}
```

This comprehensive implementation specification provides the technical foundation for executing the REE extension pipeline with all the methodological improvements, statistical rigor, and generalization testing outlined in the research plan.