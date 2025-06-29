import uuid
from datetime import datetime
from pathlib import Path
import re

from django.contrib.auth.models import User
from django.db import models


class JobStatus(models.TextChoices):
    """Job status choices."""

    PENDING = "pending", "Pending"
    PROCESSING = "processing", "Processing"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class VideoJob(models.Model):
    """
    Model for video processing jobs.

    Replaces the JSON-based job management with proper Django ORM.
    """

    # Use UUID as primary key to match existing system
    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Multi-tenancy: Add user ownership
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)

    # Video information
    video_path = models.CharField(max_length=500, help_text="Path to the video file")
    video_name = models.CharField(max_length=255, help_text="Original video filename")
    youtube_url = models.URLField(
        null=True,
        blank=True,
        help_text="Original YouTube URL if this was a YouTube video",
    )
    file_size_bytes = models.BigIntegerField(null=True, blank=True)

    # Job status and timing
    status = models.CharField(
        max_length=20, choices=JobStatus.choices, default=JobStatus.PENDING
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    processing_time = models.FloatField(
        null=True, blank=True, help_text="Processing time in seconds"
    )

    # Error handling
    error_message = models.TextField(null=True, blank=True)

    # Processing results - using JSONField for flexibility
    metadata = models.JSONField(null=True, blank=True, help_text="Video metadata")
    transcription = models.JSONField(
        null=True, blank=True, help_text="Transcription results"
    )
    processing_errors = models.JSONField(default=list, blank=True)

    # New fields
    title = models.CharField(max_length=200, blank=True)
    transcript = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.title or self.video_path}"

    @property
    def duration_seconds(self):
        """Get video duration from metadata."""
        if self.metadata and "duration_seconds" in self.metadata:
            return self.metadata["duration_seconds"]
        return 0

    @property
    def resolution(self):
        """Get video resolution from metadata."""
        if self.metadata:
            width = self.metadata.get("width_pixels", 0)
            height = self.metadata.get("height_pixels", 0)
            return f"{width}x{height}"
        return "Unknown"

    @property
    def transcription_text(self):
        """Get transcription text."""
        if self.transcription and "text" in self.transcription:
            return self.transcription["text"]
        return ""

    @property
    def text_segments(self):
        """Get text segments with timestamps."""
        if self.transcription and "text_segments" in self.transcription:
            return self.transcription["text_segments"]
        return []

    @property
    def word_count(self):
        """Get word count from transcription."""
        if self.transcription and "word_count" in self.transcription:
            return self.transcription["word_count"]
        return 0

    @property
    def language(self):
        """Get detected language."""
        if self.transcription and "language" in self.transcription:
            return self.transcription["language"]
        return "unknown"

    def is_youtube_video(self):
        """Check if this video was downloaded from YouTube."""
        # First check if we have a stored YouTube URL
        if self.youtube_url:
            return True

        # Check if the video path contains typical YouTube patterns
        video_path = str(self.video_path).lower()

        # Common patterns that indicate YouTube videos
        youtube_patterns = [
            "youtube",
            "youtu.be",
            "yt_dlp",
            # YouTube video IDs are 11 characters
            # Common pattern: title [ID].ext
        ]

        # Check for YouTube patterns in path
        for pattern in youtube_patterns:
            if pattern in video_path:
                return True

        # Check for YouTube video ID pattern in filename
        # YouTube video IDs are 11 characters of letters, numbers, hyphens, underscores
        filename = Path(self.video_path).name
        youtube_id_pattern = r"[a-zA-Z0-9_-]{11}"
        if re.search(youtube_id_pattern, filename):
            # Additional check: if filename has typical YouTube download pattern
            if any(
                pattern in filename.lower() for pattern in ["[", "]", "_", "youtube"]
            ):
                return True

        return False

    def get_youtube_video_id(self):
        """Extract YouTube video ID from the stored URL or filename."""
        # First try to extract from stored YouTube URL
        if self.youtube_url:
            patterns = [
                r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
                r"youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})",
            ]

            for pattern in patterns:
                match = re.search(pattern, self.youtube_url)
                if match:
                    return match.group(1)

        # If no URL or no match, try to extract from filename/path
        if self.is_youtube_video() and self.video_path:
            filename = Path(self.video_path).name

            # Common patterns for YouTube downloaded files:
            # 1. [TITLE] [VIDEO_ID].ext
            # 2. TITLE [VIDEO_ID].ext
            # 3. VIDEO_ID.ext
            # 4. Any 11-character alphanumeric string with hyphens/underscores

            patterns = [
                # Pattern for files like "[Title] [VIDEO_ID].mp4" or "Title [VIDEO_ID].mp4"
                r"\[([a-zA-Z0-9_-]{11})\]",
                # Pattern for files ending with " VIDEO_ID.ext"
                r"\s([a-zA-Z0-9_-]{11})\.",
                # Pattern for files starting with "VIDEO_ID "
                r"^([a-zA-Z0-9_-]{11})\s",
                # Pattern for standalone video ID as filename
                r"^([a-zA-Z0-9_-]{11})\.",
                # General pattern - any 11-character YouTube ID in the filename
                r"([a-zA-Z0-9_-]{11})",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, filename)
                for match in matches:
                    # Validate that this looks like a YouTube video ID
                    # YouTube IDs are 11 characters, mix of letters, numbers, hyphens, underscores
                    if len(match) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", match):
                        return match

        return None

    def search_segments(self, query):
        """
        Search for text in transcript segments.

        Args:
            query: Search term

        Returns:
            List of matching segments with timestamps
        """
        if not self.text_segments or not query:
            return []

        matching_segments = []
        query_lower = query.lower()

        for segment in self.text_segments:
            text = segment.get("text", "").lower()
            if query_lower in text:
                matching_segments.append(
                    {
                        "start_time": segment.get("start", 0),
                        "end_time": segment.get("end", 0),
                        "text": segment.get("text", "").strip(),
                    }
                )

        return matching_segments


class VideoSearchQuery(models.Model):
    """
    Model to track search queries for analytics.
    """

    query = models.CharField(max_length=255)
    results_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"'{self.query}' ({self.results_count} results)"
