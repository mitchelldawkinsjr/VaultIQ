"""
Semantic Search Engine for Video Content
Uses sentence transformers for embedding generation and FAISS for efficient similarity search.
"""

import logging
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pickle
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with score and metadata."""
    job_id: str
    video_name: str
    segment_index: int
    start_time: float
    end_time: float
    text: str
    score: float
    search_type: str  # 'keyword', 'semantic', 'hybrid'

class SemanticSearchEngine:
    """
    AI-powered semantic search engine for video transcriptions.
    
    Features:
    - Semantic similarity search using sentence transformers
    - Keyword-based search using TF-IDF
    - Hybrid search combining both approaches
    - Efficient vector search using FAISS
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = 'search_cache'):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
            cache_dir: Directory to cache embeddings and indices
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.index = None
        self.segments_metadata = []
        self.is_initialized = False
        
        if not SEMANTIC_SEARCH_AVAILABLE:
            logger.warning("Semantic search dependencies not available. Install sentence-transformers and faiss-cpu.")
            return
        
        try:
            if self._initialize_model():
                # Try to load existing index
                self._load_index()
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        if not SEMANTIC_SEARCH_AVAILABLE:
            return False
        
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Semantic search model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load semantic search model: {e}")
            return False
    
    def encode_segments(self, segments: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for text segments.
        
        Args:
            segments: List of segment dictionaries with 'text' field
            
        Returns:
            numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Semantic search model not initialized")
        
        texts = [segment.get('text', '').strip() for segment in segments]
        texts = [text for text in texts if text]  # Filter empty texts
        
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            logger.info(f"Generated embeddings for {len(texts)} segments")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def build_index(self, video_segments: List[Dict]):
        """
        Build FAISS index from video segments.
        
        Args:
            video_segments: List of video segment data with metadata
        """
        if not self.model:
            logger.warning("Cannot build index - semantic search model not available")
            return
        
        logger.info(f"Building semantic search index for {len(video_segments)} segments")
        
        # Clear existing data
        self.segments_metadata = []
        all_embeddings = []
        
        for video_data in video_segments:
            segments = video_data.get('segments', [])
            if not segments:
                continue
            
            # Generate embeddings for this video's segments
            embeddings = self.encode_segments(segments)
            if embeddings.size == 0:
                continue
            
            # Store metadata for each segment
            for i, segment in enumerate(segments):
                if i < len(embeddings):  # Ensure we have embedding for this segment
                    self.segments_metadata.append({
                        'job_id': video_data['job_id'],
                        'video_name': video_data['video_name'],
                        'segment_index': i,
                        'start_time': segment.get('start', 0),
                        'end_time': segment.get('end', 0),
                        'text': segment.get('text', '').strip()
                    })
            
            all_embeddings.append(embeddings)
        
        if not all_embeddings:
            logger.warning("No embeddings generated - index not built")
            return
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Build FAISS index
        dimension = combined_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(combined_embeddings)
        self.index.add(combined_embeddings)
        
        self.is_initialized = True
        logger.info(f"FAISS index built with {self.index.ntotal} segments, dimension {dimension}")
        
        # Save index and metadata
        self._save_index()
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                index_path = self.cache_dir / 'faiss_index.bin'
                faiss.write_index(self.index, str(index_path))
            
            metadata_path = self.cache_dir / 'segments_metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.segments_metadata, f)
            
            logger.info("Search index and metadata saved to cache")
        except Exception as e:
            logger.error(f"Failed to save search index: {e}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            index_path = self.cache_dir / 'faiss_index.bin'
            metadata_path = self.cache_dir / 'segments_metadata.pkl'
            
            logger.info(f"Checking for cached index at {index_path}")
            logger.info(f"Index exists: {index_path.exists()}, Metadata exists: {metadata_path.exists()}")
            
            if index_path.exists() and metadata_path.exists():
                logger.info("Loading FAISS index...")
                self.index = faiss.read_index(str(index_path))
                
                logger.info("Loading metadata...")
                with open(metadata_path, 'rb') as f:
                    self.segments_metadata = pickle.load(f)
                
                self.is_initialized = True
                logger.info(f"✅ Loaded search index with {len(self.segments_metadata)} segments")
                return True
            else:
                logger.warning("Cache files not found")
        except Exception as e:
            logger.error(f"Failed to load search index: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return False
    
    def semantic_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        Perform semantic similarity search.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.is_initialized:
            # Try to load cached index
            logger.info("Search engine not initialized, attempting to load cached index...")
            if not self._load_index():
                logger.warning("Semantic search not available - index not built")
                return []
        
        if not self.model or not self.index:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search similar segments
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.segments_metadata):
                    metadata = self.segments_metadata[idx]
                    results.append(SearchResult(
                        job_id=metadata['job_id'],
                        video_name=metadata['video_name'],
                        segment_index=metadata['segment_index'],
                        start_time=metadata['start_time'],
                        end_time=metadata['end_time'],
                        text=metadata['text'],
                        score=float(score),
                        search_type='semantic'
                    ))
            
            logger.info(f"Semantic search found {len(results)} results for: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def keyword_search(self, query: str, segments_data: List[Dict]) -> List[SearchResult]:
        """
        Perform traditional keyword-based search.
        
        Args:
            query: Search query
            segments_data: Video segments data
            
        Returns:
            List of SearchResult objects
        """
        results = []
        query_lower = query.lower()
        
        for video_data in segments_data:
            segments = video_data.get('segments', [])
            
            for i, segment in enumerate(segments):
                text = segment.get('text', '').lower()
                if query_lower in text:
                    # Simple relevance scoring based on query frequency and position
                    count = text.count(query_lower)
                    position_score = 1.0 if text.startswith(query_lower) else 0.5
                    score = count * position_score
                    
                    results.append(SearchResult(
                        job_id=video_data['job_id'],
                        video_name=video_data['video_name'],
                        segment_index=i,
                        start_time=segment.get('start', 0),
                        end_time=segment.get('end', 0),
                        text=segment.get('text', ''),
                        score=score,
                        search_type='keyword'
                    ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Keyword search found {len(results)} results for: '{query}'")
        return results
    
    def hybrid_search(self, query: str, segments_data: List[Dict], 
                     top_k: int = 20, semantic_weight: float = 0.7) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query: Search query
            segments_data: Video segments data
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0.0 to 1.0)
            
        Returns:
            List of SearchResult objects with combined scores
        """
        # Get results from both approaches
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, segments_data)
        
        # Normalize scores to 0-1 range
        if semantic_results:
            max_semantic = max(r.score for r in semantic_results)
            if max_semantic > 0:
                for r in semantic_results:
                    r.score = r.score / max_semantic
        
        if keyword_results:
            max_keyword = max(r.score for r in keyword_results)
            if max_keyword > 0:
                for r in keyword_results:
                    r.score = r.score / max_keyword
        
        # Combine results with unique segments
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            key = (result.job_id, result.segment_index)
            combined_results[key] = result
            result.score = result.score * semantic_weight
            result.search_type = 'hybrid'
        
        # Add/update with keyword results
        keyword_weight = 1.0 - semantic_weight
        for result in keyword_results:
            key = (result.job_id, result.segment_index)
            if key in combined_results:
                # Combine scores
                combined_results[key].score += result.score * keyword_weight
            else:
                # Add as new result
                result.score = result.score * keyword_weight
                result.search_type = 'hybrid'
                combined_results[key] = result
        
        # Sort and return top results
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Hybrid search found {len(final_results)} combined results for: '{query}'")
        return final_results[:top_k]
    
    def get_stats(self) -> Dict:
        """Get search engine statistics."""
        return {
            'is_available': SEMANTIC_SEARCH_AVAILABLE,
            'is_initialized': self.is_initialized,
            'model_name': self.model_name,
            'total_segments': len(self.segments_metadata) if self.is_initialized else 0,
            'index_size': self.index.ntotal if self.index else 0
        }

# Global search engine instance
search_engine = SemanticSearchEngine() 