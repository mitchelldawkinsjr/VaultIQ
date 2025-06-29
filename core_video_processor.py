#!/usr/bin/env python3
"""
Core Video Processing Module

A clean, well-structured video processing system following SOLID principles.
Provides robust video validation, metadata extraction, audio processing, and transcription.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

# Configure logging with descriptive format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies with clear handling
try:
    import cv2

    OPENCV_AVAILABLE = True
    logger.info("OpenCV successfully imported")
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available - video frame analysis will be limited")

try:
    import whisper

    WHISPER_AVAILABLE = True
    logger.info("Whisper successfully imported")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available - transcription will be disabled")


class ProcessingStatus(Enum):
    """Enumeration of possible processing statuses."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


class VideoFormat(Enum):
    """Supported video formats."""

    MP4 = ".mp4"
    AVI = ".avi"
    MOV = ".mov"
    MKV = ".mkv"
    WEBM = ".webm"


@dataclass
class VideoMetadata:
    """
    Comprehensive video metadata information.

    Contains all relevant information about a video file including
    technical specifications and file system properties.
    """

    file_path: str
    duration_seconds: float
    width_pixels: int
    height_pixels: int
    frames_per_second: float
    format_extension: str
    file_size_bytes: int
    creation_timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        metadata_dict = asdict(self)
        metadata_dict["creation_timestamp"] = self.creation_timestamp.isoformat()
        return metadata_dict

    @property
    def file_size_megabytes(self) -> float:
        """Get file size in megabytes for human readability."""
        return self.file_size_bytes / (1024 * 1024)

    @property
    def resolution_string(self) -> str:
        """Get resolution as a formatted string."""
        return f"{self.width_pixels}x{self.height_pixels}"


@dataclass
class ProcessingResult:
    """
    Result of any processing operation.

    Provides consistent structure for all processing operations
    with success status, timing, and error information.
    """

    status: ProcessingStatus
    output_file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_duration_seconds: Optional[float] = None

    @property
    def is_successful(self) -> bool:
        """Check if the processing was successful."""
        return self.status == ProcessingStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result_dict = asdict(self)
        result_dict["status"] = self.status.value
        return result_dict


@dataclass
class TranscriptionResult:
    """
    Result of video/audio transcription operation.

    Contains the transcribed text, timing information, and metadata
    about the transcription process.
    """

    status: ProcessingStatus
    transcribed_text: Optional[str] = None
    text_segments: Optional[List[Dict[str, Any]]] = None
    detected_language: Optional[str] = None
    confidence_score: Optional[float] = None
    processing_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Check if transcription was successful."""
        return self.status == ProcessingStatus.SUCCESS

    @property
    def word_count(self) -> int:
        """Get approximate word count from transcribed text."""
        if not self.transcribed_text:
            return 0
        return len(self.transcribed_text.split())


# Protocol definitions for dependency inversion
class VideoValidator(Protocol):
    """Protocol for video file validation."""

    def validate_video_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate a video file and return (is_valid, error_message)."""
        ...


class MetadataExtractor(Protocol):
    """Protocol for video metadata extraction."""

    def extract_metadata(self, file_path: str) -> Optional[VideoMetadata]:
        """Extract metadata from a video file."""
        ...


class AudioProcessor(Protocol):
    """Protocol for audio processing operations."""

    def extract_audio_from_video(
        self, video_path: str, output_path: Optional[str] = None
    ) -> ProcessingResult:
        """Extract audio from video file."""
        ...


class TranscriptionService(Protocol):
    """Protocol for transcription services."""

    def transcribe_audio_file(
        self, audio_path: str, language: str = "auto"
    ) -> TranscriptionResult:
        """Transcribe an audio file to text."""
        ...


class VideoFileValidator:
    """
    Responsible for validating video files.

    Implements single responsibility principle by focusing solely
    on video file validation logic.
    """

    SUPPORTED_FORMATS = {format.value for format in VideoFormat}
    MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB default limit

    def __init__(self, max_file_size_bytes: int = MAX_FILE_SIZE_BYTES):
        """Initialize validator with configurable file size limit."""
        self.max_file_size_bytes = max_file_size_bytes
        logger.debug(
            f"VideoFileValidator initialized with {max_file_size_bytes / (1024*1024):.1f}MB limit"
        )

    def validate_video_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a video file for processing.

        Checks file existence, format support, and size limits.

        Args:
            file_path: Path to the video file to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        video_file_path = Path(file_path)

        # Check file existence
        if not video_file_path.exists():
            return False, f"Video file does not exist: {file_path}"

        # Check file format
        file_extension = video_file_path.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            supported_formats_list = ", ".join(self.SUPPORTED_FORMATS)
            return (
                False,
                f"Unsupported video format '{file_extension}'. Supported: {supported_formats_list}",
            )

        # Check file size
        file_size_bytes = video_file_path.stat().st_size
        if file_size_bytes > self.max_file_size_bytes:
            file_size_mb = file_size_bytes / (1024 * 1024)
            max_size_mb = self.max_file_size_bytes / (1024 * 1024)
            return (
                False,
                f"File too large: {file_size_mb:.1f}MB (maximum: {max_size_mb:.1f}MB)",
            )

        # Basic file integrity check
        if file_size_bytes == 0:
            return False, "Video file is empty"

        logger.debug(f"Video file validation successful: {file_path}")
        return True, None


class VideoMetadataExtractor:
    """
    Responsible for extracting metadata from video files.

    Uses OpenCV when available, falls back to basic file system information.
    Implements single responsibility principle.
    """

    def __init__(self):
        """Initialize metadata extractor with available tools."""
        self.opencv_available = OPENCV_AVAILABLE
        self.ffmpeg_available = self._check_ffmpeg_availability()

        logger.info(
            f"MetadataExtractor initialized - OpenCV: {self.opencv_available}, FFmpeg: {self.ffmpeg_available}"
        )

    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg is available on the system."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def extract_metadata(self, file_path: str) -> Optional[VideoMetadata]:
        """
        Extract comprehensive metadata from a video file.

        Uses the best available tool (OpenCV > FFmpeg > basic file info).

        Args:
            file_path: Path to the video file

        Returns:
            VideoMetadata object or None if extraction fails
        """
        try:
            video_path = Path(file_path)

            # Basic file system metadata
            file_stats = video_path.stat()
            basic_metadata = {
                "file_path": str(video_path.absolute()),
                "format_extension": video_path.suffix.lower()[1:],  # Remove the dot
                "file_size_bytes": file_stats.st_size,
                "creation_timestamp": datetime.now(),
                "duration_seconds": 0.0,
                "width_pixels": 0,
                "height_pixels": 0,
                "frames_per_second": 0.0,
            }

            # Try to get detailed video metadata
            if self.opencv_available:
                video_metadata = self._extract_metadata_with_opencv(file_path)
                basic_metadata.update(video_metadata)
            elif self.ffmpeg_available:
                video_metadata = self._extract_metadata_with_ffmpeg(file_path)
                basic_metadata.update(video_metadata)
            else:
                logger.warning(
                    "No video analysis tools available - using basic file metadata only"
                )

            return VideoMetadata(**basic_metadata)

        except Exception as extraction_error:
            logger.error(f"Failed to extract video metadata: {extraction_error}")
            return None

    def _extract_metadata_with_opencv(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using OpenCV."""
        try:
            video_capture = cv2.VideoCapture(str(file_path))

            if not video_capture.isOpened():
                logger.warning(f"OpenCV could not open video file: {file_path}")
                return {}

            # Extract video properties
            width_pixels = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_pixels = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
            total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate duration from frame count and FPS
            duration_seconds = (
                total_frame_count / frames_per_second if frames_per_second > 0 else 0.0
            )

            video_capture.release()

            logger.debug(
                f"OpenCV metadata extracted: {width_pixels}x{height_pixels}, {frames_per_second}fps, {duration_seconds:.2f}s"
            )

            return {
                "width_pixels": width_pixels,
                "height_pixels": height_pixels,
                "frames_per_second": frames_per_second,
                "duration_seconds": duration_seconds,
            }

        except Exception as opencv_error:
            logger.warning(f"OpenCV metadata extraction failed: {opencv_error}")
            return {}

    def _extract_metadata_with_ffmpeg(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata using FFmpeg's ffprobe."""
        try:
            ffprobe_command = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                file_path,
            ]

            result = subprocess.run(
                ffprobe_command, capture_output=True, text=True, check=True
            )
            ffprobe_data = json.loads(result.stdout)

            # Find the video stream
            video_stream = None
            for stream in ffprobe_data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                logger.warning("No video stream found in FFmpeg output")
                return {}

            # Extract metadata from video stream
            width_pixels = int(video_stream.get("width", 0))
            height_pixels = int(video_stream.get("height", 0))

            # Parse frame rate (can be a fraction like "30/1")
            frame_rate_string = video_stream.get("r_frame_rate", "0/1")
            try:
                numerator, denominator = map(int, frame_rate_string.split("/"))
                frames_per_second = numerator / denominator if denominator > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                frames_per_second = 0.0

            # Get duration from format section
            duration_seconds = float(ffprobe_data.get("format", {}).get("duration", 0))

            logger.debug(
                f"FFmpeg metadata extracted: {width_pixels}x{height_pixels}, {frames_per_second}fps, {duration_seconds:.2f}s"
            )

            return {
                "width_pixels": width_pixels,
                "height_pixels": height_pixels,
                "frames_per_second": frames_per_second,
                "duration_seconds": duration_seconds,
            }

        except Exception as ffmpeg_error:
            logger.warning(f"FFmpeg metadata extraction failed: {ffmpeg_error}")
            return {}


class AudioExtractor:
    """
    Responsible for extracting audio from video files.

    Uses FFmpeg for reliable audio extraction with proper error handling.
    Implements single responsibility principle.
    """

    def __init__(self, temp_directory: Optional[str] = None):
        """Initialize audio extractor with temporary directory configuration."""
        self.temp_directory = temp_directory or tempfile.gettempdir()
        self.ffmpeg_available = self._check_ffmpeg_availability()

        if not self.ffmpeg_available:
            logger.warning("FFmpeg not available - audio extraction will fail")

        logger.debug(
            f"AudioExtractor initialized with temp directory: {self.temp_directory}"
        )

    def _check_ffmpeg_availability(self) -> bool:
        """Check if FFmpeg is available on the system."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def extract_audio_from_video(
        self, video_path: str, output_path: Optional[str] = None
    ) -> ProcessingResult:
        """
        Extract audio from a video file using FFmpeg.

        Args:
            video_path: Path to the input video file
            output_path: Optional path for output audio file

        Returns:
            ProcessingResult with extraction status and output path
        """
        start_time = datetime.now()

        try:
            if not self.ffmpeg_available:
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error_message="FFmpeg not available for audio extraction",
                )

            # Determine output path if not provided
            if output_path is None:
                video_filename = Path(video_path).stem
                output_path = os.path.join(
                    self.temp_directory, f"{video_filename}_extracted_audio.mp3"
                )

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Build FFmpeg command for high-quality audio extraction
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                video_path,  # Input video file
                "-q:a",
                "0",  # Best audio quality
                "-map",
                "a",  # Map only audio streams
                "-y",  # Overwrite output file if exists
                output_path,
            ]

            logger.info(f"Extracting audio: {video_path} -> {output_path}")

            # Execute FFmpeg command
            subprocess.run(ffmpeg_command, check=True, capture_output=True)

            # Verify output file was created
            if not Path(output_path).exists():
                return ProcessingResult(
                    status=ProcessingStatus.FAILED,
                    error_message="Audio extraction completed but output file not found",
                )

            processing_duration = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Audio extraction successful in {processing_duration:.2f} seconds"
            )

            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                output_file_path=output_path,
                metadata={
                    "input_video_path": video_path,
                    "extraction_timestamp": datetime.now().isoformat(),
                    "extraction_tool": "ffmpeg",
                },
                processing_duration_seconds=processing_duration,
            )

        except subprocess.CalledProcessError as ffmpeg_error:
            error_message = f"FFmpeg error: {ffmpeg_error.stderr.decode() if ffmpeg_error.stderr else str(ffmpeg_error)}"
            logger.error(error_message)

            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message=error_message,
                processing_duration_seconds=(
                    datetime.now() - start_time
                ).total_seconds(),
            )

        except Exception as extraction_error:
            error_message = f"Audio extraction failed: {extraction_error}"
            logger.error(error_message)

            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message=error_message,
                processing_duration_seconds=(
                    datetime.now() - start_time
                ).total_seconds(),
            )


class WhisperTranscriptionService:
    """
    Responsible for transcribing audio using OpenAI Whisper.

    Handles Whisper model loading, audio transcription, and result processing.
    Implements single responsibility principle.
    """

    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper transcription service.

        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_name = model_name
        self.whisper_model = None
        self.whisper_available = WHISPER_AVAILABLE

        if self.whisper_available:
            self._load_whisper_model()
        else:
            logger.warning("Whisper not available - transcription will be disabled")

    def _load_whisper_model(self):
        """Load the Whisper model for transcription."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.whisper_model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as model_error:
            logger.error(f"Failed to load Whisper model: {model_error}")
            self.whisper_model = None
            self.whisper_available = False

    def transcribe_audio_file(
        self, audio_path: str, language: str = "auto"
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text using Whisper.

        Args:
            audio_path: Path to the audio file to transcribe
            language: Language code for transcription ("auto" for auto-detection)

        Returns:
            TranscriptionResult with transcription text and metadata
        """
        start_time = datetime.now()

        try:
            if not self.whisper_available or not self.whisper_model:
                return TranscriptionResult(
                    status=ProcessingStatus.FAILED,
                    error_message="Whisper model not available for transcription",
                )

            audio_file_path = Path(audio_path)
            if not audio_file_path.exists():
                return TranscriptionResult(
                    status=ProcessingStatus.FAILED,
                    error_message=f"Audio file not found: {audio_path}",
                )

            logger.info(f"Starting transcription of audio file: {audio_path}")

            # Prepare transcription options
            transcription_options = {}
            if language != "auto":
                transcription_options["language"] = language

            # Perform transcription
            transcription_result = self.whisper_model.transcribe(
                audio_path, **transcription_options
            )

            processing_duration = (datetime.now() - start_time).total_seconds()

            # Calculate confidence score if available
            confidence_score = None
            if "segments" in transcription_result and transcription_result["segments"]:
                segment_confidences = [
                    segment.get("confidence", 0)
                    for segment in transcription_result["segments"]
                ]
                confidence_score = (
                    sum(segment_confidences) / len(segment_confidences)
                    if segment_confidences
                    else None
                )

            logger.info(f"Transcription completed in {processing_duration:.2f} seconds")

            return TranscriptionResult(
                status=ProcessingStatus.SUCCESS,
                transcribed_text=transcription_result.get("text", ""),
                text_segments=transcription_result.get("segments", []),
                detected_language=transcription_result.get("language", language),
                confidence_score=confidence_score,
                processing_duration_seconds=processing_duration,
            )

        except Exception as transcription_error:
            error_message = f"Transcription failed: {transcription_error}"
            logger.error(error_message)

            return TranscriptionResult(
                status=ProcessingStatus.FAILED,
                error_message=error_message,
                processing_duration_seconds=(
                    datetime.now() - start_time
                ).total_seconds(),
            )


class CoreVideoProcessor:
    """
    Main video processing coordinator.

    Orchestrates video validation, metadata extraction, audio processing,
    and transcription using dependency injection for loose coupling.
    Follows the facade pattern and dependency inversion principle.
    """

    def __init__(
        self,
        temporary_directory: Optional[str] = None,
        whisper_model_name: str = "base",
        max_file_size_bytes: int = 500 * 1024 * 1024,
        cleanup_temporary_files: bool = True,
    ):
        """
        Initialize the core video processor with all dependencies.

        Args:
            temporary_directory: Directory for temporary files
            whisper_model_name: Whisper model size for transcription
            max_file_size_bytes: Maximum allowed video file size
            cleanup_temporary_files: Whether to automatically clean up temp files
        """
        self.temporary_directory = temporary_directory or tempfile.gettempdir()
        self.cleanup_temporary_files = cleanup_temporary_files

        # Ensure temporary directory exists
        os.makedirs(self.temporary_directory, exist_ok=True)

        # Initialize all processing components using dependency injection
        self.video_validator = VideoFileValidator(max_file_size_bytes)
        self.metadata_extractor = VideoMetadataExtractor()
        self.audio_extractor = AudioExtractor(self.temporary_directory)
        self.transcription_service = WhisperTranscriptionService(whisper_model_name)

        # Log initialization status
        self._log_initialization_status()

    def _log_initialization_status(self):
        """Log the initialization status of all components."""
        logger.info("CoreVideoProcessor initialized with components:")
        logger.info("  - Video validation: Ready")
        logger.info(
            f"  - Metadata extraction: {'OpenCV' if OPENCV_AVAILABLE else 'FFmpeg/Basic'}"
        )
        logger.info(
            f"  - Audio extraction: {'FFmpeg' if self.audio_extractor.ffmpeg_available else 'Not available'}"
        )
        logger.info(
            f"  - Transcription: {'Whisper' if self.transcription_service.whisper_available else 'Not available'}"
        )
        logger.info(f"  - Temporary directory: {self.temporary_directory}")

    def validate_video_file(self, video_file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a video file for processing.

        Args:
            video_file_path: Path to the video file to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.video_validator.validate_video_file(video_file_path)

    def extract_video_metadata(self, video_file_path: str) -> Optional[VideoMetadata]:
        """
        Extract comprehensive metadata from a video file.

        Args:
            video_file_path: Path to the video file

        Returns:
            VideoMetadata object or None if extraction fails
        """
        # Validate file first
        is_valid, validation_error = self.validate_video_file(video_file_path)
        if not is_valid:
            logger.error(f"Video validation failed: {validation_error}")
            return None

        return self.metadata_extractor.extract_metadata(video_file_path)

    def extract_audio_from_video(
        self, video_file_path: str, audio_output_path: Optional[str] = None
    ) -> ProcessingResult:
        """
        Extract audio from a video file.

        Args:
            video_file_path: Path to the input video file
            audio_output_path: Optional path for output audio file

        Returns:
            ProcessingResult with extraction status
        """
        # Validate video file first
        is_valid, validation_error = self.validate_video_file(video_file_path)
        if not is_valid:
            return ProcessingResult(
                status=ProcessingStatus.FAILED,
                error_message=f"Video validation failed: {validation_error}",
            )

        return self.audio_extractor.extract_audio_from_video(
            video_file_path, audio_output_path
        )

    def transcribe_audio_file(
        self, audio_file_path: str, language_code: str = "auto"
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_file_path: Path to the audio file
            language_code: Language code for transcription

        Returns:
            TranscriptionResult with transcription data
        """
        return self.transcription_service.transcribe_audio_file(
            audio_file_path, language_code
        )

    def transcribe_video_file(
        self, video_file_path: str, language_code: str = "auto"
    ) -> TranscriptionResult:
        """
        Extract audio from video and transcribe it to text.

        This is a convenience method that combines audio extraction and transcription.

        Args:
            video_file_path: Path to the video file
            language_code: Language code for transcription

        Returns:
            TranscriptionResult with transcription data
        """
        try:
            # Step 1: Extract audio from video
            logger.info(f"Starting video transcription pipeline for: {video_file_path}")
            audio_extraction_result = self.extract_audio_from_video(video_file_path)

            if not audio_extraction_result.is_successful:
                return TranscriptionResult(
                    status=ProcessingStatus.FAILED,
                    error_message=f"Audio extraction failed: {audio_extraction_result.error_message}",
                )

            # Step 2: Transcribe the extracted audio
            transcription_result = self.transcribe_audio_file(
                audio_extraction_result.output_file_path, language_code
            )

            # Step 3: Clean up temporary audio file if configured
            if (
                self.cleanup_temporary_files
                and audio_extraction_result.output_file_path
            ):
                try:
                    os.remove(audio_extraction_result.output_file_path)
                    logger.debug(
                        f"Cleaned up temporary audio file: {audio_extraction_result.output_file_path}"
                    )
                except Exception as cleanup_error:
                    logger.warning(
                        f"Could not clean up temporary audio file: {cleanup_error}"
                    )

            return transcription_result

        except Exception as pipeline_error:
            error_message = f"Video transcription pipeline failed: {pipeline_error}"
            logger.error(error_message)

            return TranscriptionResult(
                status=ProcessingStatus.FAILED, error_message=error_message
            )

    def create_comprehensive_video_summary(
        self, video_file_path: str, language_code: str = "auto"
    ) -> Dict[str, Any]:
        """
        Create a comprehensive analysis summary of a video file.

        Combines metadata extraction and transcription for complete video analysis.

        Args:
            video_file_path: Path to the video file to analyze
            language_code: Language code for transcription

        Returns:
            Dictionary containing all analysis results
        """
        analysis_summary = {
            "video_file_path": video_file_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "metadata": None,
            "transcription": None,
            "processing_errors": [],
        }

        try:
            # Extract video metadata
            logger.info(f"Creating comprehensive summary for: {video_file_path}")
            video_metadata = self.extract_video_metadata(video_file_path)

            if video_metadata:
                analysis_summary["metadata"] = video_metadata.to_dict()
                logger.info(
                    f"Metadata extracted: {video_metadata.resolution_string}, {video_metadata.duration_seconds:.2f}s"
                )
            else:
                analysis_summary["processing_errors"].append(
                    "Failed to extract video metadata"
                )

            # Generate transcription if service is available
            if self.transcription_service.whisper_available:
                transcription_result = self.transcribe_video_file(
                    video_file_path, language_code
                )

                if transcription_result.is_successful:
                    analysis_summary["transcription"] = {
                        "text": transcription_result.transcribed_text,
                        "text_segments": transcription_result.text_segments,
                        "language": transcription_result.detected_language,
                        "confidence_score": transcription_result.confidence_score,
                        "word_count": transcription_result.word_count,
                        "processing_duration_seconds": transcription_result.processing_duration_seconds,
                    }
                    word_count = transcription_result.word_count
                    language = transcription_result.detected_language
                    logger.info(
                        f"Transcription completed: {word_count} words in {language}"
                    )
                else:
                    analysis_summary["processing_errors"].append(
                        f"Transcription failed: {transcription_result.error_message}"
                    )
            else:
                analysis_summary["processing_errors"].append(
                    "Transcription service not available"
                )

            logger.info(
                f"Video analysis completed with {len(analysis_summary['processing_errors'])} errors"
            )

        except Exception as analysis_error:
            error_message = f"Video analysis failed: {analysis_error}"
            logger.error(error_message)
            analysis_summary["processing_errors"].append(error_message)

        return analysis_summary


def main():
    """
    Example usage and command-line interface for the core video processor.

    Demonstrates the clean architecture and provides a simple CLI for testing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Core Video Processor - Clean, SOLID-compliant video processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python core_video_processor.py metadata video.mp4
  python core_video_processor.py transcribe video.mp4 --language en
  python core_video_processor.py summary video.mp4 --output results.json
        """,
    )

    parser.add_argument(
        "action",
        choices=["metadata", "transcribe", "summary"],
        help="Action to perform on the video file",
    )
    parser.add_argument("video_file", help="Path to the video file to process")
    parser.add_argument("--output", help="Output file path for results (JSON format)")
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code for transcription (default: auto-detect)",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription (default: base)",
    )

    args = parser.parse_args()

    # Initialize the processor with user configuration
    processor = CoreVideoProcessor(whisper_model_name=args.whisper_model)

    # Execute the requested action
    if args.action == "metadata":
        metadata = processor.extract_video_metadata(args.video_file)
        if metadata:
            result_data = metadata.to_dict()
            print(json.dumps(result_data, indent=2))

            if args.output:
                with open(args.output, "w") as output_file:
                    json.dump(result_data, output_file, indent=2)
                print(f"Metadata saved to: {args.output}")
        else:
            print("Failed to extract video metadata")
            sys.exit(1)

    elif args.action == "transcribe":
        transcription = processor.transcribe_video_file(args.video_file, args.language)
        if transcription.is_successful:
            result_data = {
                "text": transcription.transcribed_text,
                "language": transcription.detected_language,
                "confidence": transcription.confidence_score,
                "word_count": transcription.word_count,
                "processing_duration": transcription.processing_duration_seconds,
            }

            print(json.dumps(result_data, indent=2))

            if args.output:
                with open(args.output, "w") as output_file:
                    json.dump(result_data, output_file, indent=2)
                print(f"Transcription saved to: {args.output}")
        else:
            print(f"Transcription failed: {transcription.error_message}")
            sys.exit(1)

    elif args.action == "summary":
        summary = processor.create_comprehensive_video_summary(
            args.video_file, args.language
        )

        print(json.dumps(summary, indent=2))

        if args.output:
            with open(args.output, "w") as output_file:
                json.dump(summary, output_file, indent=2)
            print(f"Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
 