"""
Enhanced Whisper Pipeline for VaultIQ Phase 2
Improved transcription with better accuracy and error handling
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Enhanced transcription segment with additional metadata."""

    id: int
    start: float
    end: float
    text: str
    confidence: float
    language: str
    no_speech_prob: float
    words: List[Dict[str, Any]]
    speaker_id: Optional[str] = None
    sentiment: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""

    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    processing_time: float
    model_name: str
    confidence_stats: Dict[str, float]
    word_count: int
    metadata: Dict[str, Any]


class EnhancedWhisperPipeline:
    """Enhanced Whisper pipeline with improved features."""

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
    ):
        """
        Initialize enhanced Whisper pipeline.

        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (auto-detected if None)
            compute_type: Computation type for optimization
        """
        self.model_name = model_name
        self.compute_type = compute_type

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.stats = {
            "total_transcriptions": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "model_loaded": False,
            "supported_languages": [],
        }

        logger.info(
            f"Enhanced Whisper pipeline initialized: {model_name} on {self.device}"
        )

    def load_model(self):
        """Load the Whisper model with optimizations."""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            start_time = time.time()

            # Load model with device specification
            self.model = whisper.load_model(self.model_name, device=self.device)

            # Get supported languages
            self.stats["supported_languages"] = list(whisper.tokenizer.LANGUAGES.keys())
            self.stats["model_loaded"] = True

            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect the language of audio content.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (language_code, confidence)
        """
        if self.model is None:
            self.load_model()

        try:
            # Load audio and detect language
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            confidence = probs[detected_language]

            logger.info(
                f"Detected language: {detected_language} (confidence: {confidence:.2f})"
            )
            return detected_language, confidence

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return "en", 0.5  # Default to English

    def transcribe_with_enhancement(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
        vad_filter: bool = True,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """
        Enhanced transcription with improved accuracy and features.

        Args:
            audio_path: Path to audio file
            language: Language code (auto-detected if None)
            task: 'transcribe' or 'translate'
            word_timestamps: Include word-level timestamps
            vad_filter: Use voice activity detection
            temperature: Temperature for sampling (0.0 for deterministic)

        Returns:
            Enhanced transcription result
        """
        if self.model is None:
            self.load_model()

        start_time = time.time()

        try:
            # Detect language if not specified
            if language is None:
                language, lang_confidence = self.detect_language(audio_path)
            else:
                lang_confidence = 1.0

            logger.info(f"Transcribing with enhanced pipeline: {audio_path}")

            # Enhanced transcription options
            options = {
                "language": language,
                "task": task,
                "word_timestamps": word_timestamps,
                "temperature": temperature,
                "best_of": 5 if temperature > 0 else 1,
                "beam_size": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "suppress_tokens": "-1",
                "initial_prompt": None,
                "condition_on_previous_text": True,
                "fp16": self.compute_type == "float16",
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }

            # Perform transcription
            result = self.model.transcribe(audio_path, **options)

            # Process segments with enhanced metadata
            enhanced_segments = []
            for i, segment in enumerate(result["segments"]):
                # Calculate confidence from avg_logprob
                confidence = self._calculate_confidence(
                    segment.get("avg_logprob", -1.0)
                )

                # Extract word-level information if available
                words = []
                if "words" in segment:
                    words = [
                        {
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "probability": word.get("probability", 0.0),
                        }
                        for word in segment["words"]
                    ]

                enhanced_segment = TranscriptionSegment(
                    id=segment["id"],
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"].strip(),
                    confidence=confidence,
                    language=language,
                    no_speech_prob=segment.get("no_speech_prob", 0.0),
                    words=words,
                )

                enhanced_segments.append(enhanced_segment)

            # Calculate statistics
            processing_time = time.time() - start_time
            word_count = len(result["text"].split())

            # Confidence statistics
            segment_confidences = [seg.confidence for seg in enhanced_segments]
            confidence_stats = {
                "mean": np.mean(segment_confidences) if segment_confidences else 0.0,
                "median": (
                    np.median(segment_confidences) if segment_confidences else 0.0
                ),
                "min": np.min(segment_confidences) if segment_confidences else 0.0,
                "max": np.max(segment_confidences) if segment_confidences else 0.0,
                "std": np.std(segment_confidences) if segment_confidences else 0.0,
            }

            # Create enhanced result
            enhanced_result = TranscriptionResult(
                text=result["text"],
                segments=enhanced_segments,
                language=language,
                duration=enhanced_segments[-1].end if enhanced_segments else 0.0,
                processing_time=processing_time,
                model_name=self.model_name,
                confidence_stats=confidence_stats,
                word_count=word_count,
                metadata={
                    "language_confidence": lang_confidence,
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "task": task,
                    "vad_filter": vad_filter,
                    "word_timestamps": word_timestamps,
                    "processing_date": datetime.now().isoformat(),
                },
            )

            # Update statistics
            self.stats["total_transcriptions"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_transcriptions"]
            )

            logger.info(
                f"Enhanced transcription completed in {processing_time:.2f}s "
                f"({word_count} words, confidence: {confidence_stats['mean']:.2f})"
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Enhanced transcription failed: {e}")
            raise

    def _calculate_confidence(self, avg_logprob: float) -> float:
        """
        Calculate confidence score from average log probability.

        Args:
            avg_logprob: Average log probability from Whisper

        Returns:
            Confidence score between 0 and 1
        """
        # Convert log probability to confidence (empirical mapping)
        # Whisper avg_logprob typically ranges from -1.0 to 0.0
        if avg_logprob >= -0.5:
            return 0.95
        elif avg_logprob >= -1.0:
            return 0.8 + (avg_logprob + 1.0) * 0.3  # Linear mapping
        elif avg_logprob >= -1.5:
            return 0.6 + (avg_logprob + 1.5) * 0.4
        elif avg_logprob >= -2.0:
            return 0.3 + (avg_logprob + 2.0) * 0.6
        else:
            return max(0.1, 0.3 + (avg_logprob + 2.0) * 0.2)

    def enhance_transcript_quality(
        self,
        segments: List[TranscriptionSegment],
        remove_filler_words: bool = True,
        fix_capitalization: bool = True,
        merge_short_segments: bool = True,
    ) -> List[TranscriptionSegment]:
        """
        Post-process transcription to improve quality.

        Args:
            segments: List of transcription segments
            remove_filler_words: Remove common filler words
            fix_capitalization: Fix capitalization issues
            merge_short_segments: Merge very short segments

        Returns:
            Enhanced segments
        """
        enhanced_segments = segments.copy()

        if remove_filler_words:
            enhanced_segments = self._remove_filler_words(enhanced_segments)

        if fix_capitalization:
            enhanced_segments = self._fix_capitalization(enhanced_segments)

        if merge_short_segments:
            enhanced_segments = self._merge_short_segments(enhanced_segments)

        return enhanced_segments

    def _remove_filler_words(
        self, segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """Remove common filler words from transcription."""
        filler_words = {"um", "uh", "er", "ah", "like", "you know", "so", "well"}

        for segment in segments:
            words = segment.text.split()
            filtered_words = [
                word for word in words if word.lower().strip(".,!?") not in filler_words
            ]
            segment.text = " ".join(filtered_words)

        return segments

    def _fix_capitalization(
        self, segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """Fix basic capitalization issues."""
        for segment in segments:
            # Capitalize first letter of each segment
            if segment.text:
                segment.text = segment.text[0].upper() + segment.text[1:]

            # Capitalize after sentence-ending punctuation
            import re

            segment.text = re.sub(
                r"([.!?]\s+)([a-z])",
                lambda m: m.group(1) + m.group(2).upper(),
                segment.text,
            )

        return segments

    def _merge_short_segments(
        self, segments: List[TranscriptionSegment], min_duration: float = 2.0
    ) -> List[TranscriptionSegment]:
        """Merge segments that are very short."""
        if len(segments) <= 1:
            return segments

        merged_segments = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            duration = current_segment.end - current_segment.start

            if (
                duration < min_duration
                and next_segment.start - current_segment.end < 1.0
            ):
                # Merge with next segment
                current_segment.text += " " + next_segment.text
                current_segment.end = next_segment.end
                current_segment.confidence = min(
                    current_segment.confidence, next_segment.confidence
                )
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment

        merged_segments.append(current_segment)
        return merged_segments

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "model_name": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
        }

    def export_transcript(
        self,
        result: TranscriptionResult,
        format: str = "srt",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export transcript in various formats.

        Args:
            result: Transcription result
            format: Export format ('srt', 'vtt', 'txt', 'json')
            output_path: Output file path (optional)

        Returns:
            Formatted transcript string
        """
        if format == "srt":
            return self._export_srt(result, output_path)
        elif format == "vtt":
            return self._export_vtt(result, output_path)
        elif format == "txt":
            return self._export_txt(result, output_path)
        elif format == "json":
            return self._export_json(result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_srt(
        self, result: TranscriptionResult, output_path: Optional[str] = None
    ) -> str:
        """Export transcript in SRT format."""
        srt_content = []

        for i, segment in enumerate(result.segments, 1):
            start_time = self._format_timestamp(segment.start)
            end_time = self._format_timestamp(segment.end)

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text)
            srt_content.append("")

        content = "\n".join(srt_content)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT/VTT format."""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return (
            f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
        )


# Initialize enhanced Whisper pipeline
enhanced_whisper = EnhancedWhisperPipeline()


def get_enhanced_whisper():
    """Get the global enhanced Whisper pipeline instance."""
    return enhanced_whisper
