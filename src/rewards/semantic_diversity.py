"""Semantic Diversity Reward Calculator

This module implements semantic diversity-based exploration rewards using
sentence transformers to measure meaningful diversity in reasoning strategies.
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


class SemanticDiversityCalculator:
    """Calculate semantic diversity of reasoning strategies using sentence embeddings."""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.8,
                 device: Optional[str] = None):
        """
        Initialize semantic diversity calculator.

        Args:
            model_name: HuggingFace sentence transformer model name
            similarity_threshold: Threshold above which strategies are considered redundant
            device: Device to run the model on (auto-detected if None)
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold

        # Initialize sentence transformer
        try:
            self.encoder = SentenceTransformer(model_name, device=device)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise

    def extract_reasoning_content(self, strategy_xml: str) -> str:
        """
        Extract reasoning text from XML strategy block.

        Args:
            strategy_xml: String containing <strategy><reasoning>...</reasoning></strategy>

        Returns:
            reasoning_text: Cleaned reasoning content
        """
        # Extract reasoning content using regex
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        matches = re.findall(reasoning_pattern, strategy_xml, re.DOTALL | re.IGNORECASE)

        if matches:
            # Clean and return the first match
            reasoning_text = matches[0].strip()
            # Remove extra whitespace and normalize
            reasoning_text = re.sub(r'\s+', ' ', reasoning_text)
            return reasoning_text

        # Fallback: return the entire strategy if no reasoning tags found
        return strategy_xml.strip()

    def compute_strategy_embeddings(self, strategies: List[str]) -> np.ndarray:
        """
        Convert reasoning strategies to embeddings.

        Args:
            strategies: List of strategy XML blocks

        Returns:
            embeddings: Numpy array of shape (n_strategies, embedding_dim)
        """
        # Extract reasoning content
        reasoning_texts = [self.extract_reasoning_content(s) for s in strategies]

        # Filter out empty reasoning
        valid_texts = [text for text in reasoning_texts if len(text.strip()) > 10]

        if len(valid_texts) == 0:
            logger.warning("No valid reasoning texts found")
            return np.array([])

        try:
            # Encode texts to embeddings
            embeddings = self.encoder.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            return np.array([])

    def calculate_semantic_diversity(self, strategies: List[str]) -> float:
        """
        Calculate semantic diversity score for a set of strategies.

        Args:
            strategies: List of strategy XML blocks

        Returns:
            diversity_score: Float between 0 and 1 (higher = more diverse)
        """
        if len(strategies) == 0:
            return 0.0

        if len(strategies) == 1:
            return 1.0  # Single strategy has maximum diversity by definition

        # Compute embeddings
        embeddings = self.compute_strategy_embeddings(strategies)

        if len(embeddings) <= 1:
            return 1.0 if len(embeddings) == 1 else 0.0

        try:
            # Compute pairwise cosine similarities
            similarities = cosine_similarity(embeddings)

            # Get upper triangular matrix (excluding diagonal)
            triu_indices = np.triu_indices_from(similarities, k=1)
            pairwise_similarities = similarities[triu_indices]

            if len(pairwise_similarities) == 0:
                return 1.0

            # Diversity = 1 - average similarity
            avg_similarity = np.mean(pairwise_similarities)
            diversity_score = 1.0 - avg_similarity

            return max(0.0, min(1.0, diversity_score))  # Clamp to [0, 1]

        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return 0.0

    def count_unique_strategies(self, strategies: List[str],
                              similarity_threshold: Optional[float] = None) -> int:
        """
        Count semantically unique strategies.

        Args:
            strategies: List of strategy XML blocks
            similarity_threshold: Override default similarity threshold

        Returns:
            unique_count: Number of semantically distinct strategies
        """
        threshold = similarity_threshold or self.similarity_threshold

        embeddings = self.compute_strategy_embeddings(strategies)

        if len(embeddings) <= 1:
            return len(embeddings)

        # Cluster strategies based on similarity
        unique_strategies = [0]  # First strategy is always unique

        for i in range(1, len(embeddings)):
            is_unique = True
            for unique_idx in unique_strategies:
                try:
                    similarity = cosine_similarity(
                        [embeddings[i]],
                        [embeddings[unique_idx]]
                    )[0][0]

                    if similarity > threshold:
                        is_unique = False
                        break
                except Exception as e:
                    logger.warning(f"Error computing similarity: {e}")
                    continue

            if is_unique:
                unique_strategies.append(i)

        return len(unique_strategies)

    def get_diversity_metrics(self, strategies: List[str]) -> Dict[str, float]:
        """
        Get comprehensive diversity metrics for strategies.

        Args:
            strategies: List of strategy XML blocks

        Returns:
            metrics: Dictionary containing various diversity metrics
        """
        return {
            'semantic_diversity': self.calculate_semantic_diversity(strategies),
            'unique_count': self.count_unique_strategies(strategies),
            'total_count': len(strategies),
            'uniqueness_ratio': self.count_unique_strategies(strategies) / max(1, len(strategies))
        }


class SemanticExplorationReward:
    """Compute semantic diversity-based exploration rewards."""

    def __init__(self, diversity_calculator: SemanticDiversityCalculator):
        """
        Initialize semantic exploration reward calculator.

        Args:
            diversity_calculator: SemanticDiversityCalculator instance
        """
        self.diversity_calc = diversity_calculator

    def compute_reward(self,
                      strategies_with_outcomes: List[Tuple[str, str]],
                      correct_answer: str,
                      γ_correct: float = 1.0,
                      γ_diversity_cap: float = 0.5,
                      γ_diversity_rate: float = 0.1) -> float:
        """
        Compute semantic diversity-based exploration reward.

        Args:
            strategies_with_outcomes: List of (strategy_xml, strategy_outcome) tuples
            correct_answer: Ground truth answer
            γ_correct: Reward bonus for having correct strategy
            γ_diversity_cap: Maximum diversity reward
            γ_diversity_rate: Per-strategy diversity reward rate

        Returns:
            reward: Computed exploration reward
        """
        if not strategies_with_outcomes:
            return 0.0

        # Separate strategies and outcomes
        strategies = [s[0] for s in strategies_with_outcomes]
        outcomes = [s[1] for s in strategies_with_outcomes]

        # Check if any strategy is correct
        has_correct_strategy = any(
            outcome.strip().lower() == correct_answer.strip().lower()
            for outcome in outcomes
        )

        if has_correct_strategy:
            # If we have a correct strategy, prioritize that
            return γ_correct

        # Calculate semantic diversity metrics
        diversity_metrics = self.diversity_calc.get_diversity_metrics(strategies)

        # Reward based on diversity and unique strategy count
        diversity_score = diversity_metrics['semantic_diversity']
        unique_count = diversity_metrics['unique_count']

        # Combine diversity score with unique count
        diversity_reward = min(
            γ_diversity_cap,
            γ_diversity_rate * unique_count * diversity_score
        )

        return diversity_reward

    def analyze_strategy_diversity(self,
                                 strategies_with_outcomes: List[Tuple[str, str]],
                                 correct_answer: str) -> Dict[str, any]:
        """
        Analyze strategy diversity in detail for debugging/analysis.

        Args:
            strategies_with_outcomes: List of (strategy_xml, strategy_outcome) tuples
            correct_answer: Ground truth answer

        Returns:
            analysis: Detailed analysis dictionary
        """
        strategies = [s[0] for s in strategies_with_outcomes]
        outcomes = [s[1] for s in strategies_with_outcomes]

        # Get diversity metrics
        diversity_metrics = self.diversity_calc.get_diversity_metrics(strategies)

        # Analyze correctness
        correct_strategies = [
            i for i, outcome in enumerate(outcomes)
            if outcome.strip().lower() == correct_answer.strip().lower()
        ]

        # Compute pairwise similarities
        embeddings = self.diversity_calc.compute_strategy_embeddings(strategies)
        similarity_matrix = None
        if len(embeddings) > 1:
            similarity_matrix = cosine_similarity(embeddings)

        return {
            'diversity_metrics': diversity_metrics,
            'correct_strategies': correct_strategies,
            'num_correct': len(correct_strategies),
            'similarity_matrix': similarity_matrix.tolist() if similarity_matrix is not None else None,
            'strategy_texts': [self.diversity_calc.extract_reasoning_content(s) for s in strategies]
        }