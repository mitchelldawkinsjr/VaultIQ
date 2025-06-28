"""
Enhanced Semantic Search Engine for VaultIQ Phase 2
Improved embedding models, better chunking, and advanced search capabilities
"""

import logging
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

# AI/ML imports
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSearchResult:
    """Enhanced search result with additional metadata."""
    video_id: str
    video_title: str
    segment_text: str
    start_time: float
    end_time: float
    confidence_score: float
    semantic_similarity: float
    context_window: str
    topic_tags: List[str]
    sentiment_score: Optional[float] = None

class EnhancedSemanticSearchEngine:
    """Enhanced semantic search engine with better embeddings and features."""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 cache_dir: str = "search_cache",
                 chunk_overlap: int = 50,
                 max_chunk_length: int = 512):
        """
        Initialize enhanced semantic search engine.
        
        Args:
            model_name: Better embedding model for improved accuracy
            cache_dir: Directory for caching indexes and metadata
            chunk_overlap: Overlap between text chunks for better context
            max_chunk_length: Maximum length of text chunks
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.chunk_overlap = chunk_overlap
        self.max_chunk_length = max_chunk_length
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.index = None
        self.segments = []
        self.metadata = {}
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'average_search_time': 0,
            'index_size': 0,
            'model_loaded': False
        }
        
        logger.info(f"Enhanced semantic search engine initialized with model: {model_name}")
    
    @property
    def is_available(self) -> bool:
        """Check if search engine dependencies are available."""
        try:
            import sentence_transformers
            import faiss
            return True
        except ImportError:
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if search engine is fully initialized."""
        return (self.model is not None and 
                self.index is not None and 
                len(self.segments) > 0)
    
    def load_model(self):
        """Load the enhanced embedding model."""
        if self.model is not None:
            return
        
        try:
            logger.info(f"Loading enhanced embedding model: {self.model_name}")
            
            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=str(self.cache_dir / "models")
            )
            
            self.stats['model_loaded'] = True
            logger.info(f"Enhanced model loaded successfully on device: {device}")
            
        except Exception as e:
            logger.warning(f"Failed to load enhanced model {self.model_name}, falling back to base model")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                self.stats['model_loaded'] = True
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise
    
    def create_enhanced_chunks(self, text: str, video_id: str, video_title: str, 
                             start_time: float = 0) -> List[Dict[str, Any]]:
        """
        Create enhanced text chunks with overlap and context.
        
        Args:
            text: Text to chunk
            video_id: Video identifier
            video_title: Video title
            start_time: Starting timestamp
            
        Returns:
            List of enhanced chunk dictionaries
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.max_chunk_length - self.chunk_overlap):
            chunk_words = words[i:i + self.max_chunk_length]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate approximate timing based on average speaking rate
            words_per_minute = 150  # Average speaking rate
            chunk_duration = len(chunk_words) / words_per_minute * 60
            chunk_start = start_time + (i / words_per_minute * 60)
            chunk_end = chunk_start + chunk_duration
            
            # Create context window (surrounding text)
            context_start = max(0, i - 50)
            context_end = min(len(words), i + self.max_chunk_length + 50)
            context_words = words[context_start:context_end]
            context_window = ' '.join(context_words)
            
            chunk = {
                'text': chunk_text,
                'video_id': video_id,
                'video_title': video_title,
                'start_time': chunk_start,
                'end_time': chunk_end,
                'context_window': context_window,
                'chunk_index': len(chunks),
                'word_count': len(chunk_words)
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def extract_topics(self, text: str) -> List[str]:
        """
        Extract topic tags from text using simple keyword extraction.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of topic tags
        """
        # Simple topic extraction based on common patterns
        topic_keywords = {
            'relationship': ['relationship', 'love', 'marriage', 'dating', 'partner'],
            'business': ['business', 'entrepreneur', 'company', 'startup', 'money'],
            'personal_growth': ['growth', 'development', 'mindset', 'goals', 'success'],
            'health': ['health', 'fitness', 'diet', 'exercise', 'wellness'],
            'technology': ['technology', 'tech', 'software', 'programming', 'AI'],
            'education': ['education', 'learning', 'study', 'knowledge', 'skill'],
            'lifestyle': ['lifestyle', 'travel', 'food', 'culture', 'hobby']
        }
        
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics[:3]  # Limit to top 3 topics
    
    def build_enhanced_index(self, video_segments: List[Dict[str, Any]]):
        """
        Build enhanced FAISS index with improved chunking and metadata.
        
        Args:
            video_segments: List of video segment dictionaries
        """
        if not self.is_available:
            raise RuntimeError("Required dependencies not available")
        
        self.load_model()
        
        logger.info("Building enhanced semantic search index...")
        start_time = time.time()
        
        # Create enhanced chunks
        all_chunks = []
        for segment in video_segments:
            chunks = self.create_enhanced_chunks(
                text=segment['text'],
                video_id=segment['video_id'],
                video_title=segment.get('video_title', 'Unknown'),
                start_time=segment.get('start_time', 0)
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No text chunks to index")
            return
        
        # Extract text and create embeddings
        texts = [chunk['text'] for chunk in all_chunks]
        
        logger.info(f"Creating embeddings for {len(texts)} enhanced chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Enhance chunks with topic tags
        for i, chunk in enumerate(all_chunks):
            chunk['topics'] = self.extract_topics(chunk['text'])
            chunk['embedding_index'] = i
        
        self.segments = all_chunks
        
        # Update metadata
        self.metadata = {
            'model_name': self.model_name,
            'total_segments': len(all_chunks),
            'dimension': dimension,
            'build_time': time.time() - start_time,
            'chunk_overlap': self.chunk_overlap,
            'max_chunk_length': self.max_chunk_length,
            'created_at': datetime.now().isoformat()
        }
        
        self.stats['index_size'] = len(all_chunks)
        
        # Save to cache
        self._save_index()
        
        build_time = time.time() - start_time
        logger.info(f"✅ Enhanced index built with {len(all_chunks)} segments in {build_time:.2f}s")
    
    def search(self, query: str, 
               k: int = 10,
               min_similarity: float = 0.3,
               filter_topics: Optional[List[str]] = None) -> List[EnhancedSearchResult]:
        """
        Enhanced semantic search with filtering and ranking.
        
        Args:
            query: Search query
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            filter_topics: Optional topic filters
            
        Returns:
            List of enhanced search results
        """
        if not self.is_initialized:
            logger.warning("Search engine not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search index
            similarities, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(k * 2, len(self.segments))  # Get more results for filtering
            )
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity < min_similarity:
                    continue
                
                segment = self.segments[idx]
                
                # Apply topic filtering
                if filter_topics:
                    if not any(topic in segment.get('topics', []) for topic in filter_topics):
                        continue
                
                result = EnhancedSearchResult(
                    video_id=segment['video_id'],
                    video_title=segment['video_title'],
                    segment_text=segment['text'],
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    confidence_score=float(similarity),
                    semantic_similarity=float(similarity),
                    context_window=segment['context_window'],
                    topic_tags=segment.get('topics', [])
                )
                
                results.append(result)
                
                if len(results) >= k:
                    break
            
            # Update statistics
            search_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['average_search_time'] = (
                (self.stats['average_search_time'] * (self.stats['total_searches'] - 1) + search_time) 
                / self.stats['total_searches']
            )
            
            logger.info(f"Enhanced search completed: {len(results)} results in {search_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []
    
    def get_semantic_similar_segments(self, segment_text: str, k: int = 5) -> List[EnhancedSearchResult]:
        """
        Find semantically similar segments to a given text.
        
        Args:
            segment_text: Reference text
            k: Number of similar segments to return
            
        Returns:
            List of similar segments
        """
        return self.search(segment_text, k=k, min_similarity=0.4)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced search engine statistics."""
        return {
            **self.stats,
            'is_available': self.is_available,
            'is_initialized': self.is_initialized,
            'model_name': self.model_name,
            'cache_dir': str(self.cache_dir),
            'metadata': self.metadata
        }
    
    def _save_index(self):
        """Save enhanced index and metadata to cache."""
        try:
            # Save FAISS index
            index_path = self.cache_dir / "enhanced_faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            # Save segments and metadata
            segments_path = self.cache_dir / "enhanced_segments_metadata.pkl"
            with open(segments_path, 'wb') as f:
                pickle.dump({
                    'segments': self.segments,
                    'metadata': self.metadata,
                    'stats': self.stats
                }, f)
            
            logger.info("Enhanced index saved to cache")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced index: {e}")
    
    def _load_index(self):
        """Load enhanced index from cache."""
        try:
            index_path = self.cache_dir / "enhanced_faiss_index.bin"
            segments_path = self.cache_dir / "enhanced_segments_metadata.pkl"
            
            if not (index_path.exists() and segments_path.exists()):
                logger.info("No enhanced cache found, will build new index")
                return False
            
            # Load model first
            self.load_model()
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load segments and metadata
            with open(segments_path, 'rb') as f:
                data = pickle.load(f)
                self.segments = data['segments']
                self.metadata = data['metadata']
                self.stats.update(data.get('stats', {}))
            
            logger.info(f"✅ Enhanced index loaded with {len(self.segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load enhanced index: {e}")
            return False


# Initialize enhanced search engine
enhanced_search_engine = EnhancedSemanticSearchEngine()

def get_enhanced_search_engine():
    """Get the global enhanced search engine instance."""
    return enhanced_search_engine 