"""
RAG (Retrieval-Augmented Generation) Question Answering System
Advanced Q&A system with context retrieval and answer generation
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

# Import enhanced search components
from enhanced_semantic_search import EnhancedSearchResult, get_enhanced_search_engine

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Question answering result with sources."""

    question: str
    answer: str
    confidence: float
    sources: List[EnhancedSearchResult]
    processing_time: float
    method: str  # 'extractive', 'generative', 'hybrid'
    metadata: Dict[str, Any]


class RAGQuestionAnswering:
    """RAG-based question answering system."""

    def __init__(
        self,
        max_context_length: int = 2048,
        num_retrieval_results: int = 10,
        min_relevance_score: float = 0.4,
    ):
        """
        Initialize RAG Q&A system.

        Args:
            max_context_length: Maximum context length for QA
            num_retrieval_results: Number of search results to retrieve
            min_relevance_score: Minimum relevance score for search results
        """
        self.max_context_length = max_context_length
        self.num_retrieval_results = num_retrieval_results
        self.min_relevance_score = min_relevance_score

        # Get search engine
        self.search_engine = get_enhanced_search_engine()

        # QA models (will be loaded on demand)
        self.extractive_qa = None
        self.generative_qa = None

        # Statistics
        self.stats = {
            "total_questions": 0,
            "successful_answers": 0,
            "average_processing_time": 0,
            "retrieval_success_rate": 0,
            "methods_used": {"extractive": 0, "generative": 0, "hybrid": 0},
        }

        logger.info("RAG Q&A system initialized")

    def load_extractive_qa_model(self):
        """Load extractive QA model (e.g., BERT-based)."""
        if self.extractive_qa is not None:
            return

        try:
            from transformers import pipeline

            logger.info("Loading extractive QA model...")

            # Use a good extractive QA model
            self.extractive_qa = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                tokenizer="deepset/roberta-base-squad2",
                device=0 if torch.cuda.is_available() else -1,
            )

            logger.info("Extractive QA model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load extractive QA model: {e}")
            self.extractive_qa = None

    def load_generative_qa_model(self):
        """Load generative QA model (e.g., T5, GPT-based)."""
        if self.generative_qa is not None:
            return

        try:
            from transformers import pipeline

            logger.info("Loading generative QA model...")

            # Use a generative model for Q&A
            self.generative_qa = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                tokenizer="google/flan-t5-base",
                device=0 if torch.cuda.is_available() else -1,
            )

            logger.info("Generative QA model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load generative QA model: {e}")
            self.generative_qa = None

    def answer_question(
        self,
        question: str,
        method: str = "auto",
        include_sources: bool = True,
        filter_topics: Optional[List[str]] = None,
    ) -> QAResult:
        """
        Answer a question using RAG approach.

        Args:
            question: The question to answer
            method: QA method ('extractive', 'generative', 'hybrid', 'auto')
            include_sources: Include source segments in result
            filter_topics: Optional topic filters for retrieval

        Returns:
            QA result with answer and sources
        """
        start_time = time.time()

        try:
            # Step 1: Retrieve relevant context
            logger.info(f"Answering question: {question}")

            search_results = self.search_engine.search(
                query=question,
                k=self.num_retrieval_results,
                min_similarity=self.min_relevance_score,
                filter_topics=filter_topics,
            )

            if not search_results:
                logger.warning("No relevant context found for question")
                return QAResult(
                    question=question,
                    answer="I couldn't find relevant information to answer this question.",
                    confidence=0.0,
                    sources=[],
                    processing_time=time.time() - start_time,
                    method="none",
                    metadata={"error": "No relevant context found"},
                )

            # Step 2: Prepare context
            context = self._prepare_context(search_results)

            # Step 3: Choose QA method
            if method == "auto":
                method = self._choose_qa_method(question, context)

            # Step 4: Generate answer
            answer, confidence = self._generate_answer(question, context, method)

            # Step 5: Create result
            processing_time = time.time() - start_time

            result = QAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                sources=search_results if include_sources else [],
                processing_time=processing_time,
                method=method,
                metadata={
                    "context_length": len(context),
                    "num_sources": len(search_results),
                    "avg_source_confidence": sum(
                        r.confidence_score for r in search_results
                    )
                    / len(search_results),
                    "topics_found": list(
                        set(topic for r in search_results for topic in r.topic_tags)
                    ),
                    "processing_date": datetime.now().isoformat(),
                },
            )

            # Update statistics
            self._update_stats(result)

            logger.info(
                f"Question answered in {processing_time:.2f}s using {method} method"
            )
            return result

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return QAResult(
                question=question,
                answer="An error occurred while processing your question.",
                confidence=0.0,
                sources=[],
                processing_time=time.time() - start_time,
                method="error",
                metadata={"error": str(e)},
            )

    def _prepare_context(self, search_results: List[EnhancedSearchResult]) -> str:
        """
        Prepare context from search results for QA.

        Args:
            search_results: List of search results

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for result in search_results:
            # Create context snippet with metadata
            snippet = f"Video: {result.video_title}\n"
            snippet += f"Time: {result.start_time:.1f}s - {result.end_time:.1f}s\n"
            snippet += f"Content: {result.segment_text}\n"

            # Add context window if available and different
            if (
                result.context_window
                and result.context_window != result.segment_text
                and len(result.context_window) > len(result.segment_text)
            ):
                snippet += f"Context: {result.context_window}\n"

            snippet += f"Relevance: {result.confidence_score:.2f}\n"
            snippet += "---\n"

            # Check if adding this snippet would exceed max length
            if current_length + len(snippet) > self.max_context_length:
                break

            context_parts.append(snippet)
            current_length += len(snippet)

        return "\n".join(context_parts)

    def _choose_qa_method(self, question: str, context: str) -> str:
        """
        Choose the best QA method based on question and context.

        Args:
            question: The question
            context: The context

        Returns:
            Best QA method
        """
        # Simple heuristics for method selection
        question_lower = question.lower()

        # Extractive QA is good for factual questions
        extractive_keywords = [
            "who",
            "what",
            "when",
            "where",
            "which",
            "how many",
            "what is",
        ]

        # Generative QA is good for explanatory questions
        generative_keywords = [
            "why",
            "how",
            "explain",
            "describe",
            "compare",
            "summarize",
        ]

        if any(keyword in question_lower for keyword in extractive_keywords):
            return "extractive"
        elif any(keyword in question_lower for keyword in generative_keywords):
            return "generative"
        else:
            return "hybrid"

    def _generate_answer(
        self, question: str, context: str, method: str
    ) -> Tuple[str, float]:
        """
        Generate answer using specified method.

        Args:
            question: The question
            context: The context
            method: QA method

        Returns:
            Tuple of (answer, confidence)
        """
        if method == "extractive":
            return self._extractive_qa(question, context)
        elif method == "generative":
            return self._generative_qa(question, context)
        elif method == "hybrid":
            return self._hybrid_qa(question, context)
        else:
            return "Method not supported.", 0.0

    def _extractive_qa(self, question: str, context: str) -> Tuple[str, float]:
        """Extractive question answering."""
        if self.extractive_qa is None:
            self.load_extractive_qa_model()

        if self.extractive_qa is None:
            return "Extractive QA model not available.", 0.0

        try:
            result = self.extractive_qa(question=question, context=context)
            return result["answer"], result["score"]

        except Exception as e:
            logger.error(f"Extractive QA failed: {e}")
            return "Could not extract answer from context.", 0.0

    def _generative_qa(self, question: str, context: str) -> Tuple[str, float]:
        """Generative question answering."""
        if self.generative_qa is None:
            self.load_generative_qa_model()

        if self.generative_qa is None:
            return "Generative QA model not available.", 0.0

        try:
            # Format prompt for generative model
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

            result = self.generative_qa(
                prompt,
                max_length=256,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
            )

            answer = result[0]["generated_text"].replace(prompt, "").strip()
            confidence = 0.8  # Default confidence for generative models

            return answer, confidence

        except Exception as e:
            logger.error(f"Generative QA failed: {e}")
            return "Could not generate answer.", 0.0

    def _hybrid_qa(self, question: str, context: str) -> Tuple[str, float]:
        """Hybrid question answering combining extractive and generative."""
        # Try extractive first
        extractive_answer, extractive_confidence = self._extractive_qa(
            question, context
        )

        # If extractive confidence is high, use it
        if extractive_confidence > 0.7:
            return extractive_answer, extractive_confidence

        # Otherwise, try generative
        generative_answer, generative_confidence = self._generative_qa(
            question, context
        )

        # Choose the better answer
        if generative_confidence > extractive_confidence:
            return generative_answer, generative_confidence
        else:
            return extractive_answer, extractive_confidence

    def _update_stats(self, result: QAResult):
        """Update system statistics."""
        self.stats["total_questions"] += 1

        if result.confidence > 0.5:
            self.stats["successful_answers"] += 1

        if result.sources:
            self.stats["retrieval_success_rate"] = (
                self.stats["retrieval_success_rate"]
                * (self.stats["total_questions"] - 1)
                + 1
            ) / self.stats["total_questions"]

        self.stats["average_processing_time"] = (
            self.stats["average_processing_time"] * (self.stats["total_questions"] - 1)
            + result.processing_time
        ) / self.stats["total_questions"]

        if result.method in self.stats["methods_used"]:
            self.stats["methods_used"][result.method] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get Q&A system statistics."""
        success_rate = (
            self.stats["successful_answers"] / self.stats["total_questions"]
            if self.stats["total_questions"] > 0
            else 0
        )

        return {
            **self.stats,
            "success_rate": success_rate,
            "search_engine_available": self.search_engine.is_available,
            "search_engine_initialized": self.search_engine.is_initialized,
            "extractive_qa_loaded": self.extractive_qa is not None,
            "generative_qa_loaded": self.generative_qa is not None,
        }

    def batch_answer_questions(self, questions: List[str], **kwargs) -> List[QAResult]:
        """
        Answer multiple questions in batch.

        Args:
            questions: List of questions to answer
            **kwargs: Additional arguments passed to answer_question

        Returns:
            List of QA results
        """
        results = []

        for question in questions:
            result = self.answer_question(question, **kwargs)
            results.append(result)

        return results


# Initialize RAG Q&A system
rag_qa_system = RAGQuestionAnswering()


def get_rag_qa_system():
    """Get the global RAG Q&A system instance."""
    return rag_qa_system
