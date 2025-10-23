# ==============================
# FILE: agents/evaluator_agent.py
# ==============================
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.documents import Document
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@dataclass
class BiasScore:
    """Data class to store bias evaluation results."""
    label: str
    score: float
    explanation: str
class EvaluatorAgent:
    """
    Agent responsible for evaluating model outputs for bias, fairness, and explainability.
    Uses various NLP techniques to analyze text for potential issues.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the evaluator agent.
        Args:
            model_name (str): Name of the model to use for sentiment analysis
        """
        self.model_name = model_name
        self.sentiment_analyzer = None
        self._initialize_models()
    def _initialize_models(self):
        """Initialize the required NLP models."""
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name
            )
            logger.info("Successfully initialized sentiment analyzer")
        except Exception as e:
            logger.error(f"Error initializing evaluation models: {str(e)}")
            raise
    def evaluate_bias(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text for potential biases.
        Args:
            text (str): Text to evaluate
        Returns:
            Dict containing bias analysis results
        """
        if not text.strip():
            return {"error": "Empty text provided for bias evaluation"}
        try:
            # Initialize results dictionary
            results = {
                "text": text,
                "bias_scores": {},
                "warnings": [],
                "suggestions": []
            }
            # 1. Sentiment Analysis
            sentiment = self.sentiment_analyzer(text)[0]
            results["sentiment"] = {
                "label": sentiment["label"],
                "score": float(sentiment["score"])
            }
            # 2. Gender Bias Detection (simple heuristic)
            gender_terms = {
                "male_terms": ["he", "him", "his", "man", "men", "boy", "boys"],
                "female_terms": ["she", "her", "woman", "women", "girl", "girls"]
            }
            text_lower = text.lower()
            male_count = sum(text_lower.count(term) for term in gender_terms["male_terms"])
            female_count = sum(text_lower.count(term) for term in gender_terms["female_terms"])
            gender_ratio = male_count / (female_count + 1e-10)  # Avoid division by zero
            results["gender_bias"] = {
                "male_mentions": male_count,
                "female_mentions": female_count,
                "ratio": float(gender_ratio)
            }
            # Add warnings if significant gender imbalance is detected
            if gender_ratio > 3.0:
                results["warnings"].append(
                    "Significant male-gendered language detected. Consider using gender-neutral terms."
                )
            elif gender_ratio < 0.33:
                results["warnings"].append(
                    "Significant female-gendered language detected. Consider using gender-neutral terms."
                )
            # 3. Toxicity Detection (using a simple keyword approach - in practice, use a dedicated model)
            toxic_terms = [
                "hate", "stupid", "idiot", "moron", "worthless", "useless",
                "dumb", "retard", "lame", "suck"
            ]
            toxic_matches = [term for term in toxic_terms if term in text_lower]
            if toxic_matches:
                results["warnings"].append(
                    f"Potentially toxic language detected: {', '.join(toxic_matches)}"
                )
            # 4. Formality Analysis (simple heuristic)
            formal_terms = ["furthermore", "moreover", "however", "nevertheless"]
            informal_terms = ["hey", "guys", "wanna", "gonna", "yeah"]
            formal_count = sum(text_lower.count(term) for term in formal_terms)
            informal_count = sum(text_lower.count(term) for term in informal_terms)
            formality_score = (formal_count - informal_count) / (formal_count + informal_count + 1e-10)
            results["formality"] = {
                "score": float(formality_score),
                "level": "formal" if formality_score > 0.3 else "informal" if formality_score < -0.3 else "neutral"
            }
            # 5. Generate suggestions based on analysis
            if len(results["warnings"]) == 0:
                results["suggestions"].append("No significant bias detected. Text appears balanced.")
            else:
                results["suggestions"].append(
                    "Consider reviewing the text for the identified potential biases."
                )
            return results
        except Exception as e:
            error_msg = f"Error during bias evaluation: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    def explain_decision(
        self, 
        text: str, 
        target_label: Optional[str] = None,
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Explain model's decision using feature importance.
        Args:
            text (str): Input text
            target_label (str, optional): Target label to explain
            method (str): Explanation method ('shap', 'lime', or 'attention')
        Returns:
            Dict containing explanation results
        """
        try:
            # This is a placeholder - in practice, you'd use SHAP, LIME, or attention weights
            # For demonstration, we'll return a simple word importance heuristic
            words = text.split()
            word_scores = {}
            # Simple heuristic: longer words and nouns/adjectives are more important
            for word in words:
                # Clean the word
                clean_word = ''.join(c for c in word if c.isalnum())
                if not clean_word:
                    continue
                # Simple scoring (in practice, use a proper explainability method)
                score = len(clean_word) / 5.0  # Longer words get higher scores
                # Boost score for nouns/adjectives (simple heuristic)
                if len(clean_word) > 5:
                    score *= 1.5
                word_scores[clean_word] = min(score, 1.0)  # Cap at 1.0
            # Sort words by importance
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            return {
                "text": text,
                "explanation_method": method,
                "feature_importance": dict(sorted_words[:10]),  # Top 10 words
                "suggestions": [
                    "This is a simplified explanation. For more accurate results, use SHAP or LIME.",
                    "Focus on the most important words when interpreting the model's decision."
                ]
            }
        except Exception as e:
            error_msg = f"Error generating explanation: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    def evaluate_rag_output(
        self,
        query: str,
        context: List[Document],
        generated_answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG model's output for quality, relevance, and faithfulness.
        Args:
            query (str): The original user query
            context (List[Document]): Retrieved context documents
            generated_answer (str): The answer generated by the RAG model
            ground_truth (str, optional): Ground truth answer for comparison
        Returns:
            Dict containing evaluation metrics and analysis
        """
        try:
            # Basic metrics
            word_count = len(generated_answer.split())
            char_count = len(generated_answer)
            # Check if answer is too short/long
            if word_count < 5:
                answer_quality = "too_short"
                quality_score = 0.2
            elif word_count > 500:
                answer_quality = "too_long"
                quality_score = 0.5
            else:
                answer_quality = "appropriate_length"
                quality_score = 0.8
            # Check if answer is relevant to the query (simple heuristic)
            query_terms = set(query.lower().split())
            answer_terms = set(generated_answer.lower().split())
            overlap = len(query_terms.intersection(answer_terms)) / max(1, len(query_terms))
            relevance_score = min(overlap * 2, 1.0)  # Scale to 0-1 range
            # Check if answer is supported by context (simple heuristic)
            context_text = " ".join(doc.page_content for doc in context)
            context_terms = set(context_text.lower().split())
            answer_context_overlap = len(set(answer_terms).intersection(context_terms)) / max(1, len(answer_terms))
            faithfulness_score = min(answer_context_overlap * 1.5, 1.0)  # Scale to 0-1 range
            # Calculate overall score (weighted average)
            overall_score = (
                0.3 * quality_score +
                0.4 * relevance_score +
                0.3 * faithfulness_score
            )
            # Generate suggestions
            suggestions = []
            if answer_quality == "too_short":
                suggestions.append("The answer is too short. Provide more detailed information.")
            elif answer_quality == "too_long":
                suggestions.append("The answer is too verbose. Consider making it more concise.")
            if relevance_score < 0.5:
                suggestions.append("The answer may not be fully relevant to the query.")
            if faithfulness_score < 0.5:
                suggestions.append("The answer may contain information not supported by the provided context.")
            if not suggestions:
                suggestions.append("The answer is well-formed and relevant to the query.")
            # Prepare results
            results = {
                "query": query,
                "answer": generated_answer,
                "metrics": {
                    "word_count": word_count,
                    "char_count": char_count,
                    "quality_score": quality_score,
                    "relevance_score": relevance_score,
                    "faithfulness_score": faithfulness_score,
                    "overall_score": overall_score
                },
                "suggestions": suggestions
            }
            # Add ground truth comparison if available
            if ground_truth:
                # Simple similarity metric (in practice, use BLEU, ROUGE, etc.)
                gt_terms = set(ground_truth.lower().split())
                precision = len(answer_terms.intersection(gt_terms)) / max(1, len(answer_terms))
                recall = len(answer_terms.intersection(gt_terms)) / max(1, len(gt_terms))
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                results["ground_truth_comparison"] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
            return results
        except Exception as e:
            error_msg = f"Error evaluating RAG output: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
# Example usage
if __name__ == "__main__":
    # Initialize the evaluator
    evaluator = EvaluatorAgent()
    # Example text to evaluate
    sample_text = """
    The CEO announced that all employees must return to the office full-time starting next month. 
    He emphasized that this decision is final and non-negotiable. The CEO believes that 
    in-person collaboration is crucial for innovation and company culture.
    """
    # Evaluate for bias
    print("Evaluating text for bias...")
    bias_results = evaluator.evaluate_bias(sample_text)
    print("\nBias Evaluation Results:")
    print("-" * 50)
    print(f"Sentiment: {bias_results['sentiment']['label']} (Score: {bias_results['sentiment']['score']:.2f})")
    print(f"Gender Balance: {bias_results['gender_bias']['male_mentions']} male vs {bias_results['gender_bias']['female_mentions']} female mentions")
    if bias_results['warnings']:
        print("\nPotential Issues:")
        for warning in bias_results['warnings']:
            print(f"- {warning}")
    # Generate explanation
    print("\nGenerating explanation...")
    explanation = evaluator.explain_decision(sample_text)
    print("\nKey Terms and Importance:")
    for term, score in explanation['feature_importance'].items():
        print(f"{term}: {score:.2f}")
