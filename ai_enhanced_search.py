"""
AI Enhanced Search System
Combines OpenAI and Hugging Face APIs for advanced video search capabilities
Falls back to regular search when AI tokens are not available
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with AI-generated metadata"""

    video_id: str
    title: str
    content: str
    timestamp: float
    relevance_score: float
    search_mode: str

    # AI enhancements (only populated if AI is available)
    generated_questions: List[str] = None
    topic_classification: str = None
    sentiment_score: float = None
    key_concepts: List[str] = None
    summary: str = None
    confidence_score: float = None
    ai_enhanced: bool = False


@dataclass
class AIConfig:
    """Configuration for AI services"""

    # OpenAI Configuration
    openai_api_key: str = None
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 150

    # Hugging Face Configuration
    huggingface_api_key: str = None
    hf_qa_model: str = "deepset/roberta-base-squad2"
    hf_summarization_model: str = "facebook/bart-large-cnn"
    hf_classification_model: str = "facebook/bart-large-mnli"
    hf_sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Rate limiting
    max_requests_per_minute: int = 60
    request_delay: float = 1.0

    @property
    def has_any_tokens(self) -> bool:
        """Check if any AI tokens are available"""
        return bool(self.openai_api_key or self.huggingface_api_key)

    @property
    def has_openai_token(self) -> bool:
        """Check if OpenAI token is available"""
        return bool(self.openai_api_key)

    @property
    def has_huggingface_token(self) -> bool:
        """Check if Hugging Face token is available"""
        return bool(self.huggingface_api_key)


class AIEnhancedSearchEngine:
    """Enhanced search engine with OpenAI and Hugging Face integration"""

    def __init__(self, config: AIConfig):
        self.config = config
        self.request_times = []

        # Log token availability status
        if config.has_any_tokens:
            available_services = []
            if config.has_openai_token:
                available_services.append("OpenAI")
            if config.has_huggingface_token:
                available_services.append("Hugging Face")
            logger.info(f"AI tokens available for: {', '.join(available_services)}")
        else:
            logger.info(
                "No AI tokens found - will fall back to regular search functionality"
            )

        # Topic classification candidates
        self.topic_labels = [
            "relationships and dating",
            "personal development",
            "career and business",
            "health and wellness",
            "spirituality and faith",
            "education and learning",
            "technology and innovation",
            "finance and money",
            "family and parenting",
            "entertainment and lifestyle",
        ]

    @property
    def is_ai_available(self) -> bool:
        """Check if any AI services are available"""
        return self.config.has_any_tokens

    async def _rate_limit(self):
        """Implement rate limiting"""
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.config.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_times.append(now)
        await asyncio.sleep(self.config.request_delay)

    async def _call_openai_api(
        self, prompt: str, max_tokens: int = None
    ) -> Optional[str]:
        """Call OpenAI API with error handling"""
        if not self.config.openai_api_key:
            return None

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.config.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.openai_max_tokens,
            "temperature": 0.7,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"OpenAI API error {response.status}: {error_text}"
                        )
                        return None
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return None

    async def _call_huggingface_api(self, model: str, inputs: Dict) -> Optional[Dict]:
        """Call Hugging Face API with error handling"""
        if not self.config.huggingface_api_key:
            return None

        await self._rate_limit()

        headers = {
            "Authorization": f"Bearer {self.config.huggingface_api_key}",
            "Content-Type": "application/json",
        }

        url = f"https://api-inference.huggingface.co/models/{model}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=inputs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Hugging Face API error {response.status}: {error_text}"
                        )
                        return None
        except Exception as e:
            logger.error(f"Hugging Face API call failed: {e}")
            return None

    async def generate_qa_pairs_openai(self, content: str) -> List[str]:
        """Generate Q/A pairs using OpenAI"""
        prompt = f"""
        Based on this video transcript content, generate 3-5 natural questions that viewers might ask about this content. 
        Focus on the main topics, insights, and practical advice mentioned.
        
        Content: {content[:1000]}...
        
        Format as a simple list of questions, one per line:
        """

        response = await self._call_openai_api(prompt, max_tokens=200)
        if response:
            questions = [
                q.strip() for q in response.split("\n") if q.strip() and "?" in q
            ]
            return questions[:5]
        return []

    async def generate_qa_pairs_huggingface(self, content: str) -> List[str]:
        """Generate questions using Hugging Face question generation"""
        # For now, we'll use a simple approach with the QA model
        # to generate likely questions based on content

        # Extract key sentences for question generation
        sentences = content.split(".")[:5]  # First 5 sentences
        questions = []

        for sentence in sentences:
            if len(sentence.strip()) > 20:
                # Use the QA model to generate questions
                inputs = {
                    "inputs": {
                        "question": "What is this about?",
                        "context": sentence.strip(),
                    }
                }

                result = await self._call_huggingface_api(
                    self.config.hf_qa_model, inputs
                )
                if result and "answer" in result:
                    # Generate a question based on the answer
                    question = f"What does this say about {result['answer'][:50]}?"
                    questions.append(question)

        return questions[:3]

    async def classify_topic_openai(self, content: str) -> str:
        """Classify content topic using OpenAI"""
        prompt = f"""
        Classify this video content into one of these categories:
        {', '.join(self.topic_labels)}
        
        Content: {content[:800]}...
        
        Respond with just the category name that best fits:
        """

        response = await self._call_openai_api(prompt, max_tokens=50)
        if response:
            response = response.lower().strip()
            for label in self.topic_labels:
                if label in response:
                    return label
        return "general content"

    async def classify_topic_huggingface(self, content: str) -> str:
        """Classify content topic using Hugging Face"""
        inputs = {
            "inputs": content[:512],
            "parameters": {"candidate_labels": self.topic_labels},
        }

        result = await self._call_huggingface_api(
            self.config.hf_classification_model, inputs
        )
        if result and "labels" in result and result["labels"]:
            return result["labels"][0]
        return "general content"

    async def analyze_sentiment_huggingface(self, content: str) -> float:
        """Analyze sentiment using Hugging Face"""
        inputs = {"inputs": content[:512]}

        result = await self._call_huggingface_api(
            self.config.hf_sentiment_model, inputs
        )
        if result and isinstance(result, list) and result:
            # Convert sentiment to score (-1 to 1)
            sentiment_data = result[0]
            if sentiment_data["label"] == "LABEL_2":  # Positive
                return sentiment_data["score"]
            elif sentiment_data["label"] == "LABEL_0":  # Negative
                return -sentiment_data["score"]
            else:  # Neutral
                return 0.0
        return 0.0

    async def summarize_content_openai(self, content: str) -> str:
        """Summarize content using OpenAI"""
        prompt = f"""
        Provide a concise 2-3 sentence summary of the key points from this video content:
        
        {content[:1200]}...
        
        Summary:
        """

        return await self._call_openai_api(prompt, max_tokens=100)

    async def summarize_content_huggingface(self, content: str) -> str:
        """Summarize content using Hugging Face"""
        inputs = {
            "inputs": content[:1024],
            "parameters": {"max_length": 100, "min_length": 30, "do_sample": False},
        }

        result = await self._call_huggingface_api(
            self.config.hf_summarization_model, inputs
        )
        if result and isinstance(result, list) and result:
            return result[0].get("summary_text", "")
        return ""

    async def extract_key_concepts_openai(self, content: str) -> List[str]:
        """Extract key concepts using OpenAI"""
        prompt = f"""
        Extract 3-5 key concepts, topics, or themes from this content.
        Return as a simple comma-separated list.
        
        Content: {content[:1000]}...
        
        Key concepts:
        """

        response = await self._call_openai_api(prompt, max_tokens=80)
        if response:
            concepts = [c.strip() for c in response.split(",") if c.strip()]
            return concepts[:5]
        return []

    async def enhance_search_results(
        self, search_results: List[Dict], use_openai: bool = True
    ) -> List[EnhancedSearchResult]:
        """Enhance search results with AI-generated metadata or return basic results if no AI available"""
        enhanced_results = []

        # If no AI tokens available, return basic enhanced results
        if not self.is_ai_available:
            logger.info("No AI tokens available - returning basic search results")
            for result in search_results:
                enhanced = EnhancedSearchResult(
                    video_id=result.get("video_id", ""),
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    timestamp=result.get("timestamp", 0),
                    relevance_score=result.get("relevance_score", 0),
                    search_mode=result.get("search_mode", "basic"),
                    ai_enhanced=False,
                    confidence_score=result.get(
                        "relevance_score", 0
                    ),  # Use original relevance as confidence
                )
                enhanced_results.append(enhanced)
            return enhanced_results

        # AI tokens available - proceed with AI enhancement
        for result in search_results:
            try:
                # Create base enhanced result
                enhanced = EnhancedSearchResult(
                    video_id=result.get("video_id", ""),
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    timestamp=result.get("timestamp", 0),
                    relevance_score=result.get("relevance_score", 0),
                    search_mode=result.get("search_mode", "basic"),
                    ai_enhanced=True,
                )

                content = result.get("content", "")
                if not content:
                    enhanced.ai_enhanced = False
                    enhanced_results.append(enhanced)
                    continue

                # Generate AI enhancements
                if use_openai and self.config.has_openai_token:
                    # Use OpenAI for enhancements
                    tasks = [
                        self.generate_qa_pairs_openai(content),
                        self.classify_topic_openai(content),
                        self.summarize_content_openai(content),
                        self.extract_key_concepts_openai(content),
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    enhanced.generated_questions = (
                        results[0] if not isinstance(results[0], Exception) else []
                    )
                    enhanced.topic_classification = (
                        results[1]
                        if not isinstance(results[1], Exception)
                        else "general"
                    )
                    enhanced.summary = (
                        results[2] if not isinstance(results[2], Exception) else ""
                    )
                    enhanced.key_concepts = (
                        results[3] if not isinstance(results[3], Exception) else []
                    )

                elif self.config.has_huggingface_token:
                    # Use Hugging Face for enhancements
                    tasks = [
                        self.generate_qa_pairs_huggingface(content),
                        self.classify_topic_huggingface(content),
                        self.analyze_sentiment_huggingface(content),
                        self.summarize_content_huggingface(content),
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    enhanced.generated_questions = (
                        results[0] if not isinstance(results[0], Exception) else []
                    )
                    enhanced.topic_classification = (
                        results[1]
                        if not isinstance(results[1], Exception)
                        else "general"
                    )
                    enhanced.sentiment_score = (
                        results[2] if not isinstance(results[2], Exception) else 0.0
                    )
                    enhanced.summary = (
                        results[3] if not isinstance(results[3], Exception) else ""
                    )

                # Calculate confidence score based on available data
                confidence_factors = []
                if enhanced.generated_questions:
                    confidence_factors.append(0.3)
                if (
                    enhanced.topic_classification
                    and enhanced.topic_classification != "general"
                ):
                    confidence_factors.append(0.3)
                if enhanced.summary:
                    confidence_factors.append(0.2)
                if enhanced.key_concepts:
                    confidence_factors.append(0.2)

                enhanced.confidence_score = sum(confidence_factors)
                enhanced_results.append(enhanced)

            except Exception as e:
                logger.error(f"Error enhancing result: {e}")
                # Add basic result without enhancements
                enhanced_results.append(
                    EnhancedSearchResult(
                        video_id=result.get("video_id", ""),
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        timestamp=result.get("timestamp", 0),
                        relevance_score=result.get("relevance_score", 0),
                        search_mode=result.get("search_mode", "basic"),
                        ai_enhanced=False,
                        confidence_score=0.0,
                    )
                )

        return enhanced_results

    async def answer_question(
        self, question: str, context_results: List[Dict]
    ) -> Optional[str]:
        """Answer a question based on search results context, or return None if no AI available"""
        if not context_results:
            return None

        # If no AI tokens available, cannot generate answers
        if not self.is_ai_available:
            logger.info("No AI tokens available - cannot generate answers")
            return None

        # Combine context from top results
        context = "\n".join([r.get("content", "")[:500] for r in context_results[:3]])

        if self.config.has_openai_token:
            prompt = f"""
            Based on the following video transcript content, answer this question concisely:
            
            Question: {question}
            
            Context:
            {context}
            
            Answer:
            """

            return await self._call_openai_api(prompt, max_tokens=150)

        elif self.config.has_huggingface_token:
            inputs = {"inputs": {"question": question, "context": context[:512]}}

            result = await self._call_huggingface_api(self.config.hf_qa_model, inputs)
            if result and "answer" in result:
                return result["answer"]

        return None


def create_ai_config() -> AIConfig:
    """Create AI configuration from environment variables"""
    return AIConfig(
        # OpenAI Configuration
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "150")),
        # Hugging Face Configuration
        huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY"),
        hf_qa_model=os.getenv("HF_QA_MODEL", "deepset/roberta-base-squad2"),
        hf_summarization_model=os.getenv(
            "HF_SUMMARIZATION_MODEL", "facebook/bart-large-cnn"
        ),
        hf_classification_model=os.getenv(
            "HF_CLASSIFICATION_MODEL", "facebook/bart-large-mnli"
        ),
        hf_sentiment_model=os.getenv(
            "HF_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ),
        # Rate limiting
        max_requests_per_minute=int(os.getenv("AI_MAX_REQUESTS_PER_MINUTE", "60")),
        request_delay=float(os.getenv("AI_REQUEST_DELAY", "1.0")),
    )


# Global AI engine instance
_ai_engine = None


def get_ai_engine() -> Optional[AIEnhancedSearchEngine]:
    """Get global AI engine instance - always returns engine, even without tokens"""
    global _ai_engine
    if _ai_engine is None:
        config = create_ai_config()
        # Always create the engine - it will handle token availability internally
        _ai_engine = AIEnhancedSearchEngine(config)
        if not config.has_any_tokens:
            logger.info(
                "AI engine created without tokens - will fall back to regular search"
            )
    return _ai_engine


async def enhance_search_with_ai(
    search_results: List[Dict], use_openai: bool = True
) -> List[EnhancedSearchResult]:
    """Convenience function to enhance search results with AI (falls back to regular results if no tokens)"""
    ai_engine = get_ai_engine()
    if ai_engine:
        return await ai_engine.enhance_search_results(search_results, use_openai)
    else:
        # Fallback: return basic enhanced results without AI
        logger.warning("AI engine not available - returning basic results")
        return [
            EnhancedSearchResult(
                video_id=r.get("video_id", ""),
                title=r.get("title", ""),
                content=r.get("content", ""),
                timestamp=r.get("timestamp", 0),
                relevance_score=r.get("relevance_score", 0),
                search_mode=r.get("search_mode", "basic"),
                ai_enhanced=False,
                confidence_score=0.0,
            )
            for r in search_results
        ]
