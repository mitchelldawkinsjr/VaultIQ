import asyncio
import json
import logging
import mimetypes
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import yt_dlp
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse_lazy
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView

from core_video_processor import CoreVideoProcessor

from .models import JobStatus, VideoJob, VideoSearchQuery

# Import semantic search engine (the initialized instance)
try:
    from semantic_search import SemanticSearchEngine, search_engine
except ImportError:
    search_engine = None
    SemanticSearchEngine = None

# Import AI enhancement system
try:
    from ai_enhanced_search import (
        create_ai_config,
        enhance_search_with_ai,
        get_ai_engine,
    )

    AI_AVAILABLE = True
    logger = logging.getLogger(__name__)
except ImportError:
    AI_AVAILABLE = False
    logger = None

# Phase 2: Import enhanced AI components
try:
    from enhanced_semantic_search import get_enhanced_search_engine
    from enhanced_whisper_pipeline import get_enhanced_whisper
    from rag_qa_system import get_rag_qa_system

    enhanced_search = get_enhanced_search_engine()
    enhanced_whisper = get_enhanced_whisper()
    rag_qa = get_rag_qa_system()

    PHASE2_AI_AVAILABLE = True
    logger.info("Phase 2 enhanced AI components loaded successfully")

except ImportError as e:
    logger.warning(f"Phase 2 AI components not available: {e}")
    enhanced_search = None
    enhanced_whisper = None
    rag_qa = None
    PHASE2_AI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Initialize the video processor
video_processor = CoreVideoProcessor()


# User Authentication Views
def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get("username")
            messages.success(request, f"Account created for {username}! Please log in.")
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "registration/register.html", {"form": form})


# Multi-tenant Video Library View
@method_decorator(login_required, name="dispatch")
class VideoLibraryView(LoginRequiredMixin, ListView):
    model = VideoJob
    template_name = "video_processor/library.html"
    context_object_name = "videos"
    login_url = reverse_lazy("login")

    def get_queryset(self):
        # Filter videos by current user
        return VideoJob.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add user-specific statistics
        user_videos = self.get_queryset()
        context.update(
            {
                "video_count": user_videos.count(),
                "completed_count": user_videos.filter(status="completed").count(),
                "processing_count": user_videos.filter(status="processing").count(),
                "failed_count": user_videos.filter(status="failed").count(),
            }
        )
        return context


@login_required
def search_interface_view(request):
    return render(request, "video_processor/search_interface.html")


# Function removed - using the more comprehensive upload_video function below


def search_videos(request):
    """Enhanced search through video transcriptions with semantic capabilities."""
    if request.method != "POST":
        return redirect("video_library")

    query = request.POST.get("query", "").strip()
    search_mode = request.POST.get("search_mode", "hybrid")  # keyword, semantic, hybrid

    if not query:
        return redirect("video_library")

    # Track search query
    VideoSearchQuery.objects.create(query=query, results_count=0)

    # Perform search based on selected mode
    search_results = []

    try:
        if search_mode == "semantic" and search_engine.is_initialized:
            # Pure semantic search
            semantic_results = search_engine.semantic_search(query, top_k=50)
            search_results = convert_semantic_results_to_display_format(
                semantic_results
            )

        elif search_mode == "hybrid" and search_engine.is_initialized:
            # Hybrid search (combines semantic + keyword)
            video_segments_data = get_video_segments_for_search()
            hybrid_results = search_engine.hybrid_search(
                query, video_segments_data, top_k=50
            )
            search_results = convert_semantic_results_to_display_format(hybrid_results)

        else:
            # Fallback to keyword search
            search_results = perform_keyword_search(query)

    except Exception as e:
        logger.error(f"Search error: {e}")
        # Fallback to keyword search on any error
        search_results = perform_keyword_search(query)

    # Update search query results count
    VideoSearchQuery.objects.filter(query=query).update(
        results_count=len(search_results)
    )

    # Calculate statistics (same as library view)
    total_videos = VideoJob.objects.filter(status=JobStatus.COMPLETED).count()
    total_processing_time = sum(
        job.processing_time or 0
        for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
    )
    total_words = sum(
        job.word_count or 0
        for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
    )
    pending_jobs = VideoJob.objects.filter(status=JobStatus.PENDING).count()

    stats = {
        "total_videos": total_videos,
        "total_processing_time": total_processing_time,
        "total_words": total_words,
        "pending_jobs": pending_jobs,
    }

    # Check search engine status for template
    search_available = False
    try:
        stats_search = search_engine.get_stats()
        if stats_search["is_available"] and not stats_search["is_initialized"]:
            search_engine._load_index()
            stats_search = search_engine.get_stats()
        search_available = stats_search["is_initialized"]
    except Exception as e:
        logger.warning(f"Could not check search engine status: {e}")

    return render(
        request,
        "video_processor/library.html",
        {
            "videos": VideoJob.objects.filter(status=JobStatus.COMPLETED).order_by(
                "-created_at"
            ),
            "search_results": search_results,
            "query": query,
            "search_mode": search_mode,
            "semantic_available": search_available,
            "stats": stats,
        },
    )


def perform_keyword_search(query):
    """Perform traditional keyword-based search."""
    search_results = []
    videos = VideoJob.objects.filter(
        status=JobStatus.COMPLETED, transcription__isnull=False
    )

    for video in videos:
        transcription_text = video.transcription_text.lower()
        if query.lower() in transcription_text:
            # Get matching segments
            matching_segments = video.search_segments(query)
            if matching_segments:
                search_results.append({"video": video, "segments": matching_segments})

    return search_results


def get_video_segments_for_search():
    """Get video segments data formatted for semantic search."""
    videos = VideoJob.objects.filter(
        status=JobStatus.COMPLETED, transcription__isnull=False
    )

    video_segments_data = []
    for video in videos:
        segments = video.text_segments
        if segments:
            video_segments_data.append(
                {
                    "job_id": str(video.job_id),
                    "video_name": video.video_name,
                    "segments": segments,
                }
            )

    return video_segments_data


def convert_semantic_results_to_display_format(semantic_results):
    """Convert semantic search results to the format expected by the template."""
    # Group results by video
    video_groups = {}

    for result in semantic_results:
        job_id = result.job_id
        if job_id not in video_groups:
            try:
                video = VideoJob.objects.get(job_id=job_id)
                video_groups[job_id] = {"video": video, "segments": []}
            except VideoJob.DoesNotExist:
                continue

        # Add segment with relevance score
        segment = {
            "start_time": result.start_time,
            "end_time": result.end_time,
            "text": result.text,
            "relevance_score": result.score,
            "search_type": result.search_type,
        }
        video_groups[job_id]["segments"].append(segment)

    return list(video_groups.values())


def convert_semantic_results_to_api_format(semantic_results):
    """Convert semantic search results to JSON-serializable format for API."""
    api_results = []

    for result in semantic_results:
        try:
            video = VideoJob.objects.get(job_id=result.job_id)
            api_results.append(
                {
                    "video_id": str(result.job_id),
                    "video_name": video.video_name,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "text": result.text,
                    "relevance_score": result.score,
                    "search_type": result.search_type,
                }
            )
        except VideoJob.DoesNotExist:
            continue

    return api_results


def convert_keyword_results_to_api_format(keyword_results):
    """Convert keyword search results to JSON-serializable format for API."""
    api_results = []

    for result_group in keyword_results:
        video = result_group["video"]
        segments = result_group["segments"]

        for segment in segments:
            api_results.append(
                {
                    "video_id": str(video.job_id),
                    "video_name": video.video_name,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": segment["text"],
                    "relevance_score": segment.get("relevance_score", 0.0),
                    "search_type": "keyword",
                }
            )

    return api_results


@csrf_exempt
def upload_video(request):
    """Handle video upload and processing (files or YouTube URLs)."""
    if request.method != "POST":
        return redirect("video_library")

    # Check if it's a YouTube URL submission
    youtube_url = request.POST.get("youtube_url", "").strip()

    if youtube_url:
        return handle_youtube_upload(request, youtube_url)

    # Handle file upload
    if "video" not in request.FILES:
        messages.error(request, "No video file provided")
        return redirect("video_library")

    video_file = request.FILES["video"]
    if not video_file.name:
        messages.error(request, "Invalid video file")
        return redirect("video_library")

    try:
        # Save video file to media directory
        media_videos_dir = settings.MEDIA_ROOT / "videos"
        media_videos_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename to avoid conflicts
        import uuid

        unique_id = uuid.uuid4()
        file_extension = Path(video_file.name).suffix
        safe_filename = f"{unique_id}_{video_file.name}"
        file_path = media_videos_dir / safe_filename

        # Save the file
        with open(file_path, "wb+") as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Create video job
        job = VideoJob.objects.create(
            job_id=unique_id,  # Use the same UUID for consistency
            video_path=str(file_path),
            video_name=video_file.name,
            file_size_bytes=video_file.size,
            status=JobStatus.PENDING,
        )

        # Process video in background thread
        thread = threading.Thread(target=process_video_job, args=(job.job_id,))
        thread.daemon = True
        thread.start()

        messages.success(
            request, f'Video "{video_file.name}" uploaded and queued for processing'
        )

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        messages.error(request, f"Error uploading video: {e}")

    return redirect("video_library")


def handle_youtube_upload(request, youtube_url):
    """Handle YouTube URL submission."""
    try:
        # Validate YouTube URL
        if not is_youtube_url(youtube_url):
            messages.error(
                request, "Invalid YouTube URL. Please provide a valid YouTube link."
            )
            return redirect("video_library")

        # Set up download directory
        media_videos_dir = settings.MEDIA_ROOT / "videos"

        messages.info(
            request, "Downloading YouTube video... This may take a few minutes."
        )

        # Download video
        success, video_path, video_info, error_msg = download_youtube_video(
            youtube_url, media_videos_dir
        )

        if not success:
            messages.error(request, f"Failed to download YouTube video: {error_msg}")
            return redirect("video_library")

        # Get file size
        file_size = Path(video_path).stat().st_size

        # Create video job
        import uuid

        job = VideoJob.objects.create(
            video_path=video_path,
            video_name=video_info["title"],
            youtube_url=youtube_url,  # Store the original YouTube URL
            file_size_bytes=file_size,
            status=JobStatus.PENDING,
        )

        # Process video in background thread
        thread = threading.Thread(target=process_video_job, args=(job.job_id,))
        thread.daemon = True
        thread.start()

        messages.success(
            request,
            f'YouTube video "{video_info["title"]}" downloaded and queued for processing',
        )

    except Exception as e:
        logger.error(f"Error processing YouTube URL: {e}")
        messages.error(request, f"Error processing YouTube video: {e}")

    return redirect("video_library")


def process_video_job(job_id):
    """Process a video job in the background."""
    try:
        job = VideoJob.objects.get(job_id=job_id)
        job.status = JobStatus.PROCESSING
        job.started_at = timezone.now()
        job.save()

        start_time = time.time()

        # Process video using our core processor
        result = video_processor.create_comprehensive_video_summary(job.video_path)

        processing_time = time.time() - start_time

        # Update job with results
        job.metadata = result.get("metadata")
        job.transcription = result.get("transcription")
        job.processing_errors = result.get("processing_errors", [])
        job.processing_time = processing_time
        job.completed_at = timezone.now()

        if result.get("processing_errors"):
            job.status = JobStatus.FAILED
            job.error_message = "; ".join(result["processing_errors"])
        else:
            # Only mark as completed if we have everything needed for searching
            transcription_data = result.get("transcription", {})
            has_text = bool(transcription_data.get("text", "").strip())
            has_segments = bool(transcription_data.get("text_segments", []))

            if has_text and has_segments:
                job.status = JobStatus.COMPLETED
                logger.info(
                    f"Job {job_id} marked as COMPLETED - ready for searching with {len(transcription_data.get('text_segments', []))} segments"
                )
            else:
                job.status = JobStatus.FAILED
                job.error_message = (
                    "Transcription incomplete - missing text or segments"
                )
                logger.warning(
                    f"Job {job_id} failed completion check - has_text: {has_text}, has_segments: {has_segments}"
                )

        job.save()

        logger.info(f"Job {job_id} completed in {processing_time:.2f}s")

        # Clean up YouTube videos after successful processing to save storage
        # Check if this is a YouTube video (either by naming pattern or video path containing YouTube IDs)
        is_youtube = (
            job.video_name.startswith(("YouTube_", "yt_"))
            or "media/videos/" in job.video_path
            and any(char in job.video_path for char in ["_", "-"])
            and len(Path(job.video_path).stem.split("_")[0]) == 11  # YouTube ID length
        )

        if job.status == JobStatus.COMPLETED and is_youtube:
            try:
                video_path = Path(job.video_path)
                if video_path.exists():
                    file_size_mb = video_path.stat().st_size / (1024 * 1024)
                    video_path.unlink()
                    logger.info(
                        f"Cleaned up YouTube video file: {job.video_name} ({file_size_mb:.1f}MB freed)"
                    )
                    # Update path to indicate file was cleaned up
                    job.video_path = f"[CLEANED_UP] {job.video_path}"
                    job.save()
            except Exception as cleanup_error:
                logger.warning(f"Could not clean up YouTube video: {cleanup_error}")

        # Update semantic search index if video was successfully processed
        if job.status == JobStatus.COMPLETED:
            try:
                rebuild_search_index()
            except Exception as e:
                logger.warning(f"Failed to update search index: {e}")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        try:
            job = VideoJob.objects.get(job_id=job_id)
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save()
        except:
            pass


def rebuild_search_index():
    """Rebuild the semantic search index with all completed videos."""
    try:
        # Get all completed videos with transcription data
        videos = VideoJob.objects.filter(
            status=JobStatus.COMPLETED, transcription__isnull=False
        )

        if not videos.exists():
            logger.info("No videos available for search indexing")
            return

        # Prepare video segments data for indexing
        video_segments = []
        for video in videos:
            segments = video.text_segments
            if segments:
                video_segments.append(
                    {
                        "job_id": str(video.job_id),
                        "video_name": video.video_name,
                        "segments": segments,
                    }
                )

        if video_segments:
            # Build the search index
            search_engine.build_index(video_segments)
            logger.info(f"Search index rebuilt with {len(video_segments)} videos")
        else:
            logger.warning("No video segments found for indexing")

    except Exception as e:
        logger.error(f"Error rebuilding search index: {e}")


def download_youtube_video(url, output_path):
    """
    Download a YouTube video using yt-dlp.

    Args:
        url: YouTube URL
        output_path: Directory to save the video

    Returns:
        tuple: (success, video_path, video_info, error_message)
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Configure yt-dlp options
        ydl_opts = {
            "format": "best[ext=mp4]/best",  # Prefer mp4 format
            "outtmpl": str(output_path / "%(id)s_%(title)s.%(ext)s"),
            "restrictfilenames": True,  # Remove special characters from filename
            "noplaylist": True,  # Only download single video, not playlist
            "extractaudio": False,
            "audioformat": "mp3",
            "quiet": True,  # Reduce output verbosity
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info first
            info = ydl.extract_info(url, download=False)

            if not info:
                return False, None, None, "Could not extract video information"

            # Check video duration (optional limit)
            duration = info.get("duration", 0)
            if duration > 3600:  # 1 hour limit
                return (
                    False,
                    None,
                    None,
                    f"Video too long: {duration//60:.1f} minutes (max 60 minutes)",
                )

            # Download the video
            ydl.download([url])

            # Find the downloaded file
            video_id = info.get("id", "unknown")
            title = info.get("title", "Unknown")

            # Look for the downloaded file
            downloaded_files = list(output_path.glob(f"{video_id}_*"))
            if not downloaded_files:
                return False, None, None, "Downloaded file not found"

            video_path = downloaded_files[0]

            # Return success with file info
            return (
                True,
                str(video_path),
                {
                    "title": title,
                    "duration": duration,
                    "uploader": info.get("uploader", "Unknown"),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "description": info.get("description", "")[
                        :500
                    ],  # Truncate description
                    "url": url,
                    "video_id": video_id,
                },
                None,
            )

    except Exception as e:
        logger.error(f"Error downloading YouTube video: {e}")
        return False, None, None, str(e)


def is_youtube_url(url):
    """Check if a URL is a valid YouTube URL."""
    youtube_patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+",
        r"(?:https?://)?(?:www\.)?youtu\.be/[\w-]+",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+",
        r"(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+",
    ]

    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


def api_jobs(request):
    """API endpoint for job data."""
    jobs = VideoJob.objects.all().order_by("-created_at")
    job_data = []

    for job in jobs:
        job_info = {
            "job_id": str(job.job_id),
            "video_path": job.video_path,
            "video_name": job.video_name,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time": job.processing_time,
            "has_transcript": bool(job.transcription_text),
            "word_count": job.word_count,
            "language": job.language,
        }
        job_data.append(job_info)

    return JsonResponse(job_data, safe=False)


def api_latest_job(request):
    """API endpoint for the latest job status."""
    try:
        latest_job = VideoJob.objects.order_by("-created_at").first()

        if not latest_job:
            return JsonResponse({"error": "No jobs found"}, status=404)

        job_info = {
            "job_id": str(latest_job.job_id),
            "video_path": latest_job.video_path,
            "video_name": latest_job.video_name,
            "status": latest_job.status,
            "created_at": latest_job.created_at.isoformat(),
            "completed_at": (
                latest_job.completed_at.isoformat() if latest_job.completed_at else None
            ),
            "processing_time": latest_job.processing_time,
            "has_transcript": bool(latest_job.transcription_text),
            "has_segments": bool(latest_job.text_segments),
            "segment_count": (
                len(latest_job.text_segments) if latest_job.text_segments else 0
            ),
            "word_count": latest_job.word_count,
            "language": latest_job.language,
            "error_message": latest_job.error_message,
            "is_searchable": latest_job.status == JobStatus.COMPLETED
            and bool(latest_job.transcription_text)
            and bool(latest_job.text_segments),
        }

        return JsonResponse(job_info)

    except Exception as e:
        logger.error(f"Error in API latest job: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def api_video_details(request, job_id):
    """API endpoint for detailed video information."""
    try:
        job = get_object_or_404(VideoJob, job_id=job_id)

        if job.status != JobStatus.COMPLETED or not job.transcription:
            return JsonResponse({"error": "Video not processed yet"}, status=404)

        # Prepare segments data
        segments = []
        for i, segment in enumerate(job.text_segments):
            segments.append(
                {
                    "index": i,
                    "start_time": segment.get("start", 0),
                    "end_time": segment.get("end", 0),
                    "duration": segment.get("end", 0) - segment.get("start", 0),
                    "text": segment.get("text", "").strip(),
                    "confidence": segment.get("confidence", 0),
                }
            )

        video_details = {
            "job_id": str(job.job_id),
            "video_name": job.video_name,
            "video_path": job.video_path,
            "youtube_url": job.youtube_url,
            "youtube_video_id": job.get_youtube_video_id(),
            "is_youtube": job.is_youtube_video(),
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "processing_time": job.processing_time,
            "metadata": {
                "duration_seconds": job.duration_seconds,
                "width_pixels": (
                    job.metadata.get("width_pixels", 0) if job.metadata else 0
                ),
                "height_pixels": (
                    job.metadata.get("height_pixels", 0) if job.metadata else 0
                ),
                "frames_per_second": (
                    job.metadata.get("frames_per_second", 0) if job.metadata else 0
                ),
                "file_size_bytes": (
                    job.metadata.get("file_size_bytes", 0) if job.metadata else 0
                ),
            },
            "transcription": {
                "text": job.transcription_text,
                "language": job.language,
                "word_count": job.word_count,
                "confidence_score": (
                    job.transcription.get("confidence_score", 0)
                    if job.transcription
                    else 0
                ),
                "processing_duration_seconds": (
                    job.transcription.get("processing_duration_seconds", 0)
                    if job.transcription
                    else 0
                ),
            },
            "segments": segments,
            "segment_count": len(segments),
        }

        return JsonResponse(video_details)

    except Exception as e:
        logger.error(f"Error in API video details: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def video_player_page(request, job_id):
    """Direct link to video player page with optional timestamp."""
    # timestamp = request.GET.get("t", 0)  # Future use
    # job = get_object_or_404(VideoJob, job_id=job_id)  # Future use

    # For now, redirect to main page
    # In a full implementation, this would show a dedicated video player
    return redirect("video_library")


def delete_video(request, job_id):
    """Delete a video job."""
    if request.method == "POST":
        job = get_object_or_404(VideoJob, job_id=job_id)
        video_name = job.video_name

        # Clean up video file if it exists
        try:
            video_path = Path(job.video_path)
            if video_path.exists():
                video_path.unlink()
        except Exception as e:
            logger.warning(f"Could not delete video file: {e}")

        job.delete()
        messages.success(request, f'Video "{video_name}" has been deleted')

    return redirect("video_library")


def serve_video(request, job_id):
    """Serve video files with proper streaming support."""
    job = get_object_or_404(VideoJob, job_id=job_id)

    # Get the video file path
    video_path = Path(job.video_path)

    # If the original path doesn't exist, check in media directory
    if not video_path.exists():
        media_path = settings.MEDIA_ROOT / "videos" / f"{job_id}_{job.video_name}"
        if media_path.exists():
            video_path = media_path
        else:
            raise Http404("Video file not found")

    if not video_path.exists():
        raise Http404("Video file not found")

    # Get content type
    content_type, _ = mimetypes.guess_type(str(video_path))
    if content_type is None:
        content_type = "video/mp4"

    # Handle range requests for video streaming
    range_header = request.META.get("HTTP_RANGE")
    file_size = video_path.stat().st_size

    if range_header:
        # Parse range header
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1

        # Ensure valid range
        if start >= file_size:
            start = file_size - 1
        if end >= file_size:
            end = file_size - 1

        chunk_size = end - start + 1

        # Create response with partial content
        response = HttpResponse(status=206)  # Partial Content
        response["Content-Type"] = content_type
        response["Content-Length"] = str(chunk_size)
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        response["Accept-Ranges"] = "bytes"

        # Read and return the requested chunk
        with open(video_path, "rb") as video_file:
            video_file.seek(start)
            response.write(video_file.read(chunk_size))

        return response
    else:
        # Return entire file
        response = FileResponse(
            open(video_path, "rb"), content_type=content_type, filename=job.video_name
        )
        response["Content-Length"] = str(file_size)
        response["Accept-Ranges"] = "bytes"
        return response


def api_search_status(request):
    """API endpoint to check search engine status."""
    try:
        stats = search_engine.get_stats()
        return JsonResponse(
            {
                "available": stats["is_available"],
                "initialized": stats["is_initialized"],
                "model_name": stats.get("model_name", "Unknown"),
                "indexed_segments": stats.get("indexed_segments", 0),
            }
        )
    except Exception as e:
        return JsonResponse({"available": False, "initialized": False, "error": str(e)})


@csrf_exempt
def api_rebuild_search_index(request):
    """API endpoint to rebuild search index."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        # Get all completed videos for indexing
        videos = VideoJob.objects.filter(
            status=JobStatus.COMPLETED, transcription__isnull=False
        )

        if not videos.exists():
            return JsonResponse(
                {"error": "No completed videos found to index"}, status=400
            )

        # Prepare segments for indexing
        all_segments = []
        for video in videos:
            segments = video.text_segments
            if segments:
                for segment in segments:
                    all_segments.append(
                        {
                            "video_id": str(video.job_id),
                            "video_name": video.video_name,
                            "text": segment["text"],
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                        }
                    )

        if not all_segments:
            return JsonResponse(
                {"error": "No text segments found to index"}, status=400
            )

        # Build the index
        search_engine.build_index(all_segments)

        # Get updated stats
        stats = search_engine.get_stats()

        return JsonResponse(
            {
                "success": True,
                "message": f"Search index rebuilt successfully with {len(all_segments)} segments",
                "segments_count": stats.get("indexed_segments", len(all_segments)),
            }
        )

    except Exception as e:
        logger.error(f"Error rebuilding search index: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_cleanup_youtube(request):
    """API endpoint to cleanup YouTube videos."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        dry_run = request.POST.get("dry_run", "false").lower() == "true"

        # Find YouTube videos that can be cleaned up
        youtube_videos = []
        total_size = 0

        for video in VideoJob.objects.filter(status=JobStatus.COMPLETED):
            video_path = Path(video.video_path)
            if video_path.exists() and video.is_youtube_video():
                size_mb = video_path.stat().st_size / (1024 * 1024)
                youtube_videos.append(
                    {
                        "job_id": str(video.job_id),
                        "name": video.video_name,
                        "path": str(video_path),
                        "size_mb": round(size_mb, 1),
                    }
                )
                total_size += size_mb

        if not youtube_videos:
            return JsonResponse(
                {
                    "success": True,
                    "message": "No YouTube videos found to clean up",
                    "files_processed": 0,
                    "space_freed_mb": 0,
                    "dry_run": dry_run,
                }
            )

        # Perform cleanup if not dry run
        if not dry_run:
            for video_info in youtube_videos:
                video_path = Path(video_info["path"])
                if video_path.exists():
                    video_path.unlink()
                    logger.info(f"Deleted YouTube video: {video_path}")

        return JsonResponse(
            {
                "success": True,
                "message": f'{"Would delete" if dry_run else "Deleted"} {len(youtube_videos)} YouTube video files',
                "files_processed": len(youtube_videos),
                "space_freed_mb": round(total_size, 1),
                "dry_run": dry_run,
                "files": youtube_videos,
            }
        )

    except Exception as e:
        logger.error(f"Error cleaning up YouTube videos: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_process_job(request):
    """API endpoint to process a specific job."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        job_id = request.POST.get("job_id")
        if not job_id:
            return JsonResponse({"error": "Job ID required"}, status=400)

        try:
            job = VideoJob.objects.get(job_id=job_id)
        except VideoJob.DoesNotExist:
            return JsonResponse({"error": f"Job not found: {job_id}"}, status=404)

        if job.status != JobStatus.PENDING:
            return JsonResponse(
                {"error": f"Job {job_id} is not pending (status: {job.status})"},
                status=400,
            )

        # Start processing in background thread
        def process_job():
            try:
                process_video_job(job.job_id)
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}")

        thread = threading.Thread(target=process_job)
        thread.daemon = True
        thread.start()

        return JsonResponse(
            {
                "success": True,
                "message": f"Processing started for job {job_id}",
                "job_id": job_id,
            }
        )

    except Exception as e:
        logger.error(f"Error starting job processing: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def api_timestamp_content(request, job_id):
    """API endpoint to get content at specific timestamp."""
    try:
        timestamp = float(request.GET.get("timestamp", 0))

        try:
            job = VideoJob.objects.get(job_id=job_id)
        except VideoJob.DoesNotExist:
            return JsonResponse({"error": f"Job not found: {job_id}"}, status=404)

        if job.status != JobStatus.COMPLETED or not job.transcription:
            return JsonResponse(
                {"error": "Video not processed or transcription not available"},
                status=400,
            )

        # Find the segment containing this timestamp
        segments = job.text_segments or []
        current_segment = None

        for segment in segments:
            if segment["start_time"] <= timestamp <= segment["end_time"]:
                current_segment = segment
                break

        if not current_segment:
            return JsonResponse(
                {"error": f"No content found at timestamp {timestamp}s"}, status=404
            )

        return JsonResponse(
            {
                "timestamp": timestamp,
                "segment": current_segment,
                "video_name": job.video_name,
                "job_id": str(job.job_id),
            }
        )

    except ValueError:
        return JsonResponse({"error": "Invalid timestamp"}, status=400)
    except Exception as e:
        logger.error(f"Error getting timestamp content: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def api_detailed_stats(request):
    """API endpoint for detailed statistics."""
    try:
        # Basic stats
        total_videos = VideoJob.objects.filter(status=JobStatus.COMPLETED).count()
        pending_jobs = VideoJob.objects.filter(status=JobStatus.PENDING).count()
        processing_jobs = VideoJob.objects.filter(status=JobStatus.PROCESSING).count()
        failed_jobs = VideoJob.objects.filter(status=JobStatus.FAILED).count()

        # Processing time stats
        completed_jobs = VideoJob.objects.filter(
            status=JobStatus.COMPLETED, processing_time__isnull=False
        )
        total_processing_time = sum(job.processing_time for job in completed_jobs)
        avg_processing_time = (
            total_processing_time / completed_jobs.count()
            if completed_jobs.count() > 0
            else 0
        )

        # Content stats
        total_words = sum(
            job.word_count or 0
            for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
        )
        total_duration = sum(
            job.duration_seconds or 0
            for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
        )

        # Language distribution
        language_stats = {}
        for job in VideoJob.objects.filter(status=JobStatus.COMPLETED):
            lang = job.language or "unknown"
            language_stats[lang] = language_stats.get(lang, 0) + 1

        # Recent activity (last 7 days)
        from datetime import timedelta

        week_ago = timezone.now() - timedelta(days=7)
        recent_jobs = VideoJob.objects.filter(created_at__gte=week_ago).count()

        # Storage stats
        total_file_size = sum(
            job.file_size_bytes or 0 for job in VideoJob.objects.all()
        )

        # Search stats
        search_stats = {}
        try:
            search_stats = search_engine.get_stats()
        except Exception as e:
            search_stats = {"error": str(e)}

        return JsonResponse(
            {
                "video_stats": {
                    "total_videos": total_videos,
                    "pending_jobs": pending_jobs,
                    "processing_jobs": processing_jobs,
                    "failed_jobs": failed_jobs,
                    "recent_jobs_7_days": recent_jobs,
                },
                "processing_stats": {
                    "total_processing_time": round(total_processing_time, 2),
                    "average_processing_time": round(avg_processing_time, 2),
                    "total_duration_hours": round(total_duration / 3600, 2),
                },
                "content_stats": {
                    "total_words": total_words,
                    "language_distribution": language_stats,
                },
                "storage_stats": {
                    "total_file_size_gb": round(total_file_size / (1024**3), 2)
                },
                "search_stats": search_stats,
            }
        )

    except Exception as e:
        logger.error(f"Error getting detailed stats: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def api_pending_jobs(request):
    """API endpoint to get pending jobs."""
    try:
        pending_jobs = VideoJob.objects.filter(status=JobStatus.PENDING).order_by(
            "-created_at"
        )

        jobs_data = []
        for job in pending_jobs:
            jobs_data.append(
                {
                    "job_id": str(job.job_id),
                    "video_name": job.video_name,
                    "created_at": job.created_at.isoformat(),
                    "file_size_mb": (
                        round((job.file_size_bytes or 0) / (1024 * 1024), 1)
                        if job.file_size_bytes
                        else 0
                    ),
                }
            )

        return JsonResponse({"pending_jobs": jobs_data, "count": len(jobs_data)})

    except Exception as e:
        logger.error(f"Error getting pending jobs: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def clean_search_interface(request):
    """Clean, user-facing search interface."""
    return render(request, "video_processor/search_interface.html")


def public_enhanced_search_interface(request):
    """Public enhanced search interface with Phase 2 AI capabilities - no authentication required."""
    return render(request, "video_processor/public_enhanced_search.html")


def library_interface(request):
    """Legacy library interface with full admin functionality."""
    # Calculate statistics
    total_videos = VideoJob.objects.filter(status=JobStatus.COMPLETED).count()
    total_processing_time = sum(
        job.processing_time or 0
        for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
    )
    total_words = sum(
        job.word_count or 0
        for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
    )
    pending_jobs = VideoJob.objects.filter(status=JobStatus.PENDING).count()

    # Check and initialize search engine if needed
    search_available = False
    try:
        stats = search_engine.get_stats()
        if stats["is_available"] and not stats["is_initialized"]:
            logger.info("Initializing search engine for library interface...")
            search_engine._load_index()
            stats = search_engine.get_stats()
        search_available = stats["is_initialized"]
    except Exception as e:
        logger.warning(f"Could not initialize search engine: {e}")

    context = {
        "videos": VideoJob.objects.filter(status=JobStatus.COMPLETED).order_by(
            "-created_at"
        ),
        "stats": {
            "total_videos": total_videos,
            "total_processing_time": total_processing_time,
            "total_words": total_words,
            "pending_jobs": pending_jobs,
        },
        "query": None,
        "search_results": None,
        "semantic_available": search_available,
    }

    return render(request, "video_processor/library.html", context)


@csrf_exempt
def api_video_info(request, video_id):
    """API endpoint to get video info for lightbox player."""
    try:
        video = VideoJob.objects.get(job_id=video_id)

        response_data = {
            "video_id": str(video.job_id),
            "video_name": video.video_name,
            "youtube_url": video.youtube_url or "",
            "status": video.status,
            "is_youtube": video.is_youtube_video(),
        }

        # Add YouTube video ID if it's a YouTube video
        if video.is_youtube_video():
            youtube_video_id = video.get_youtube_video_id()
            response_data["youtube_video_id"] = youtube_video_id

        return JsonResponse(response_data)
    except VideoJob.DoesNotExist:
        return JsonResponse({"error": "Video not found"}, status=404)


def serve_video_file(request, video_id):
    """Serve local video files."""
    try:
        video = VideoJob.objects.get(job_id=video_id)

        # Check if this is a YouTube video
        if video.is_youtube_video():
            return JsonResponse(
                {"error": "This is a YouTube video - use the YouTube URL instead"},
                status=400,
            )

        # Check if local file exists
        if video.video_path and os.path.exists(video.video_path):
            return FileResponse(
                open(video.video_path, "rb"),
                content_type="video/mp4",
                filename=os.path.basename(video.video_path),
            )
        else:
            return HttpResponse("Video file not found", status=404)
    except VideoJob.DoesNotExist:
        return HttpResponse("Video not found", status=404)


@csrf_exempt
def api_clean_search(request):
    """API endpoint for the clean search interface."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        search_mode = data.get("mode", "hybrid")

        if not query:
            return JsonResponse({"error": "Search query is required"}, status=400)

        logger.info(f"Clean search API: '{query}' (mode: {search_mode})")

        # Perform search based on mode
        if search_mode == "keyword":
            results = convert_keyword_results_to_api_format(
                perform_keyword_search(query)
            )
        elif (
            search_mode == "semantic" and search_engine and search_engine.is_initialized
        ):
            semantic_results = search_engine.semantic_search(query, top_k=20)
            results = convert_semantic_results_to_api_format(semantic_results)
        elif search_mode == "hybrid" and search_engine and search_engine.is_initialized:
            video_segments_data = get_video_segments_for_search()
            semantic_results = search_engine.hybrid_search(
                query, video_segments_data, top_k=20
            )
            results = convert_semantic_results_to_api_format(semantic_results)
        else:
            # Fall back to keyword search if semantic/hybrid not available
            logger.info(
                f"Search engine not available (engine: {search_engine}, mode: {search_mode}), falling back to keyword search"
            )
            results = convert_keyword_results_to_api_format(
                perform_keyword_search(query)
            )
            search_mode = "keyword"

        # Save search query
        VideoSearchQuery.objects.create(query=query, results_count=len(results))

        return JsonResponse(
            {
                "results": results,
                "count": len(results),
                "query": query,
                "mode": search_mode,
                "message": f'Found {len(results)} results for "{query}"',
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in clean search API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_public_enhanced_search(request):
    """Public enhanced search API - searches across all videos without authentication."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()

        if not query:
            return JsonResponse({"error": "Search query is required"}, status=400)

        logger.info(f"Public enhanced search: '{query}'")

        # Check if Phase 2 enhanced search is available
        if PHASE2_AI_AVAILABLE and enhanced_search:
            try:
                # Use enhanced search engine
                results = enhanced_search.search(query, k=20)

                # Convert to API format
                api_results = []
                for result in results:
                    api_results.append(
                        {
                            "video_id": result.video_id,
                            "title": result.video_title,
                            "content": result.segment_text,
                            "timestamp": result.start_time,
                            "confidence_score": result.confidence_score,
                            "topic": result.topic_tags,
                            "enhanced": True,
                        }
                    )

                return JsonResponse(
                    {
                        "results": api_results,
                        "count": len(api_results),
                        "query": query,
                        "enhanced": True,
                        "message": f'Found {len(api_results)} enhanced results for "{query}"',
                    }
                )

            except Exception as e:
                logger.error(f"Enhanced search error: {e}")
                # Fall back to regular search

        # Fall back to regular semantic search
        if search_engine and search_engine.is_initialized:
            semantic_results = search_engine.semantic_search(query, top_k=20)
            results = convert_semantic_results_to_api_format(semantic_results)
        else:
            # Fall back to keyword search
            logger.info(
                f"Search engine not available for public search, falling back to keyword search"
            )
            results = convert_keyword_results_to_api_format(
                perform_keyword_search(query)
            )

        # Save search query
        VideoSearchQuery.objects.create(query=query, results_count=len(results))

        return JsonResponse(
            {
                "results": results,
                "count": len(results),
                "query": query,
                "enhanced": False,
                "message": f'Found {len(results)} results for "{query}"',
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in public enhanced search API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_public_rag_question_answer(request):
    """Public RAG-based Q&A API - answers questions using all video content without authentication."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()

        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)

        logger.info(f"Public RAG Q&A: '{question}'")

        # Check if RAG system is available
        if PHASE2_AI_AVAILABLE and rag_qa:
            try:
                # Use RAG Q&A system
                result = rag_qa.answer_question(question)

                return JsonResponse(
                    {
                        "question": question,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "sources": [
                            {
                                "video_id": source.video_id,
                                "title": source.title,
                                "content": source.content,
                                "timestamp": source.timestamp,
                                "relevance_score": source.relevance_score,
                            }
                            for source in result.sources
                        ],
                        "method": result.method,
                        "enhanced": True,
                    }
                )

            except Exception as e:
                logger.error(f"RAG Q&A error: {e}")
                # Fall back to simple search

        # Fall back to simple search-based answer
        if search_engine and search_engine.is_initialized:
            search_results = search_engine.semantic_search(question, top_k=5)
            api_results = convert_semantic_results_to_api_format(search_results)
        else:
            logger.info(
                f"Search engine not available for Q&A, falling back to keyword search"
            )
            api_results = convert_keyword_results_to_api_format(
                perform_keyword_search(question)
            )

        # Create a simple answer from search results
        answer = (
            "Based on the available video content, here are the most relevant segments:"
        )
        if api_results:
            answer += f" Found {len(api_results)} relevant segments."
        else:
            answer = "I couldn't find relevant information to answer your question."

        return JsonResponse(
            {
                "question": question,
                "answer": answer,
                "confidence": 0.5 if api_results else 0.1,
                "sources": api_results[:3],  # Top 3 sources
                "method": "search_fallback",
                "enhanced": False,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in public RAG Q&A API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# AI Enhancement Endpoints
@csrf_exempt
def api_ai_enhanced_search(request):
    """AI-enhanced search with Q/A pairs, topic classification, and summaries (falls back to regular search)"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        search_mode = data.get("mode", "hybrid")
        use_openai = data.get("use_openai", True)

        if not query:
            return JsonResponse({"error": "Query required"}, status=400)

        logger.info(
            f"AI Enhanced search: '{query}' (mode: {search_mode}, openai: {use_openai})"
        )

        # Check if AI is available
        ai_available = AI_AVAILABLE
        if AI_AVAILABLE:
            try:
                config = create_ai_config()
                ai_available = config.has_any_tokens
                if not ai_available:
                    logger.info("No AI tokens found - falling back to regular search")
            except Exception as e:
                logger.warning(
                    f"AI config check failed: {e} - falling back to regular search"
                )
                ai_available = False

        # First, perform the regular search to get base results
        search_results = []

        try:
            if search_mode == "semantic" and search_engine.is_initialized:
                semantic_results = search_engine.semantic_search(query, top_k=20)
                search_results = convert_semantic_results_to_display_format(
                    semantic_results
                )
            elif search_mode == "hybrid" and search_engine.is_initialized:
                video_segments_data = get_video_segments_for_search()
                hybrid_results = search_engine.hybrid_search(
                    query, video_segments_data, top_k=20
                )
                search_results = convert_semantic_results_to_display_format(
                    hybrid_results
                )
            else:
                search_results = perform_keyword_search(query)

        except Exception as e:
            logger.error(f"Base search error: {e}")
            search_results = perform_keyword_search(query)

        # If AI not available, return regular results in enhanced format
        if not ai_available:
            enhanced_api_results = []
            for result in search_results[:20]:
                enhanced_api_results.append(
                    {
                        "video_id": result.get("video_id", ""),
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "timestamp": result.get("timestamp", 0),
                        "relevance_score": result.get("relevance_score", 0),
                        "search_mode": search_mode,
                        # Empty AI enhancements
                        "ai_enhanced": False,
                        "generated_questions": [],
                        "topic_classification": None,
                        "sentiment_score": None,
                        "key_concepts": [],
                        "summary": None,
                        "confidence_score": result.get("relevance_score", 0),
                    }
                )

            # Track search query
            VideoSearchQuery.objects.create(
                query=query, results_count=len(enhanced_api_results)
            )

            return JsonResponse(
                {
                    "results": enhanced_api_results,
                    "total": len(enhanced_api_results),
                    "query": query,
                    "mode": search_mode,
                    "ai_enhanced": False,
                    "fallback_reason": "No AI tokens available",
                    "ai_provider": None,
                }
            )

        # Convert to format expected by AI enhancement
        ai_input_results = []
        for result in search_results[:10]:  # Limit to top 10 for AI processing
            ai_input_results.append(
                {
                    "video_id": result.get("video_id", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "timestamp": result.get("timestamp", 0),
                    "relevance_score": result.get("relevance_score", 0),
                    "search_mode": search_mode,
                }
            )

        # Apply AI enhancements asynchronously
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            enhanced_results = loop.run_until_complete(
                enhance_search_with_ai(ai_input_results, use_openai)
            )
        except Exception as e:
            logger.error(f"AI enhancement error: {e}")
            # Return regular results if AI fails
            enhanced_api_results = []
            for result in search_results[:20]:
                enhanced_api_results.append(
                    {
                        "video_id": result.get("video_id", ""),
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "timestamp": result.get("timestamp", 0),
                        "relevance_score": result.get("relevance_score", 0),
                        "search_mode": search_mode,
                        "ai_enhanced": False,
                        "generated_questions": [],
                        "topic_classification": None,
                        "sentiment_score": None,
                        "key_concepts": [],
                        "summary": None,
                        "confidence_score": result.get("relevance_score", 0),
                    }
                )

            return JsonResponse(
                {
                    "results": enhanced_api_results,
                    "total": len(enhanced_api_results),
                    "query": query,
                    "mode": search_mode,
                    "ai_enhanced": False,
                    "fallback_reason": f"AI enhancement failed: {str(e)}",
                    "ai_provider": None,
                }
            )

        # Convert enhanced results back to API format
        enhanced_api_results = []
        for enhanced in enhanced_results:
            result = {
                "video_id": enhanced.video_id,
                "title": enhanced.title,
                "content": enhanced.content,
                "timestamp": enhanced.timestamp,
                "relevance_score": enhanced.relevance_score,
                "search_mode": enhanced.search_mode,
                # AI enhancements
                "ai_enhanced": enhanced.ai_enhanced,
                "generated_questions": enhanced.generated_questions or [],
                "topic_classification": enhanced.topic_classification,
                "sentiment_score": enhanced.sentiment_score,
                "key_concepts": enhanced.key_concepts or [],
                "summary": enhanced.summary,
                "confidence_score": enhanced.confidence_score or 0.0,
            }
            enhanced_api_results.append(result)

        # Track search query
        VideoSearchQuery.objects.create(
            query=query, results_count=len(enhanced_api_results)
        )

        # Determine if any results were actually AI enhanced
        ai_enhanced_count = sum(1 for r in enhanced_results if r.ai_enhanced)

        return JsonResponse(
            {
                "results": enhanced_api_results,
                "total": len(enhanced_api_results),
                "query": query,
                "mode": search_mode,
                "ai_enhanced": ai_enhanced_count > 0,
                "ai_enhanced_count": ai_enhanced_count,
                "ai_provider": (
                    "openai"
                    if use_openai
                    else "huggingface" if ai_enhanced_count > 0 else None
                ),
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"AI enhanced search error: {e}")
        return JsonResponse({"error": f"AI search failed: {str(e)}"}, status=500)


@csrf_exempt
def api_ai_question_answer(request):
    """Answer questions based on video content using AI (returns context if no AI available)"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()
        context_limit = data.get("context_limit", 3)

        if not question:
            return JsonResponse({"error": "Question required"}, status=400)

        logger.info(f"AI Q/A: '{question}'")

        # Check if AI is available
        ai_available = AI_AVAILABLE
        if AI_AVAILABLE:
            try:
                config = create_ai_config()
                ai_available = config.has_any_tokens
                if not ai_available:
                    logger.info("No AI tokens found - returning context only")
            except Exception as e:
                logger.warning(f"AI config check failed: {e} - returning context only")
                ai_available = False

        # Search for relevant context
        try:
            if search_engine.is_initialized:
                video_segments_data = get_video_segments_for_search()
                context_results = search_engine.hybrid_search(
                    question, video_segments_data, top_k=context_limit
                )
                context_data = convert_semantic_results_to_display_format(
                    context_results
                )
            else:
                context_data = perform_keyword_search(question)[:context_limit]
        except Exception as e:
            logger.error(f"Context search error: {e}")
            context_data = perform_keyword_search(question)[:context_limit]

        # If AI not available, return context only
        if not ai_available:
            return JsonResponse(
                {
                    "question": question,
                    "answer": None,
                    "ai_available": False,
                    "fallback_reason": "No AI tokens available",
                    "context_results": context_data,
                    "context_count": len(context_data),
                    "suggestion": "Based on the context above, you can find relevant information about your question.",
                }
            )

        # Convert to format for AI
        context_for_ai = []
        for result in context_data:
            context_for_ai.append(
                {
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "timestamp": result.get("timestamp", 0),
                }
            )

        # Get AI answer
        try:
            ai_engine = get_ai_engine()
            if ai_engine:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                answer = loop.run_until_complete(
                    ai_engine.answer_question(question, context_for_ai)
                )

                if answer:
                    return JsonResponse(
                        {
                            "question": question,
                            "answer": answer,
                            "ai_available": True,
                            "ai_enhanced": True,
                            "context_results": context_data,
                            "context_count": len(context_data),
                        }
                    )
                else:
                    return JsonResponse(
                        {
                            "question": question,
                            "answer": None,
                            "ai_available": True,
                            "ai_enhanced": False,
                            "fallback_reason": "Could not generate answer",
                            "context_results": context_data,
                            "context_count": len(context_data),
                            "suggestion": "The AI could not generate an answer. Please check the context above for relevant information.",
                        }
                    )
            else:
                return JsonResponse(
                    {
                        "question": question,
                        "answer": None,
                        "ai_available": False,
                        "fallback_reason": "AI engine not available",
                        "context_results": context_data,
                        "context_count": len(context_data),
                    }
                )

        except Exception as e:
            logger.error(f"AI Q/A error: {e}")
            return JsonResponse(
                {
                    "question": question,
                    "answer": None,
                    "ai_available": True,
                    "ai_enhanced": False,
                    "fallback_reason": f"Answer generation failed: {str(e)}",
                    "context_results": context_data,
                    "context_count": len(context_data),
                    "suggestion": "There was an error generating the AI answer. Please check the context above for relevant information.",
                }
            )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        logger.error(f"Q/A API error: {e}")
        return JsonResponse({"error": f"Q/A failed: {str(e)}"}, status=500)


def api_ai_status(request):
    """Get AI system status and configuration"""
    if not AI_AVAILABLE:
        return JsonResponse(
            {
                "available": False,
                "system_installed": False,
                "tokens_available": False,
                "error": "AI enhancement system not installed",
            }
        )

    try:
        config = create_ai_config()

        return JsonResponse(
            {
                "available": True,
                "system_installed": True,
                "tokens_available": config.has_any_tokens,
                "openai_token_available": config.has_openai_token,
                "huggingface_token_available": config.has_huggingface_token,
                "fallback_mode": not config.has_any_tokens,
                "configuration": {
                    "openai_model": config.openai_model,
                    "openai_max_tokens": config.openai_max_tokens,
                    "hf_qa_model": config.hf_qa_model,
                    "hf_summarization_model": config.hf_summarization_model,
                    "hf_classification_model": config.hf_classification_model,
                    "hf_sentiment_model": config.hf_sentiment_model,
                    "max_requests_per_minute": config.max_requests_per_minute,
                    "request_delay": config.request_delay,
                },
                "status": (
                    "AI tokens available - full functionality"
                    if config.has_any_tokens
                    else "No AI tokens - fallback to regular search"
                ),
            }
        )

    except Exception as e:
        return JsonResponse(
            {
                "available": True,
                "system_installed": True,
                "tokens_available": False,
                "configuration_error": str(e),
                "status": "AI system installed but configuration failed - fallback to regular search",
            }
        )


@login_required
@csrf_exempt
def edit_transcript(request, job_id):
    """Allow users to edit transcript content."""
    # Ensure user can only edit their own videos
    job = get_object_or_404(VideoJob, job_id=job_id, user=request.user)

    try:

        if request.method == "GET":
            # Return current transcript for editing
            transcript_text = ""
            if job.transcription and "segments" in job.transcription:
                # Extract text from segments
                segments = job.transcription["segments"]
                transcript_text = "\n".join(
                    [seg.get("text", "").strip() for seg in segments]
                )
            elif job.transcript:
                transcript_text = job.transcript

            return JsonResponse(
                {
                    "status": "success",
                    "transcript": transcript_text,
                    "job_id": str(job.job_id),
                    "title": job.video_name,
                }
            )

        elif request.method == "POST":
            # Save edited transcript
            data = json.loads(request.body)
            new_transcript = data.get("transcript", "").strip()

            # Update the transcript field
            job.transcript = new_transcript

            # Update transcription JSON structure if it exists
            if job.transcription and "segments" in job.transcription:
                # Split the edited text back into segments (simple approach)
                lines = new_transcript.split("\n")
                segments = job.transcription["segments"]

                # Update segments with new text (preserve timing if available)
                for i, line in enumerate(lines[: len(segments)]):
                    if i < len(segments):
                        segments[i]["text"] = line.strip()

                # Add new segments for additional lines
                for i in range(len(segments), len(lines)):
                    segments.append(
                        {
                            "text": lines[i].strip(),
                            "start": segments[-1]["end"] if segments else 0,
                            "end": segments[-1]["end"] + 5 if segments else 5,
                        }
                    )

                job.transcription["segments"] = segments

            job.save()

            # Rebuild search index if needed
            try:
                if search_engine.is_initialized:
                    logger.info(
                        f"Rebuilding search index after transcript edit for job {job_id}"
                    )
                    search_engine.rebuild_index()
            except Exception as e:
                logger.warning(f"Could not rebuild search index: {e}")

            return JsonResponse(
                {"status": "success", "message": "Transcript updated successfully"}
            )

    except json.JSONDecodeError:
        return JsonResponse(
            {"status": "error", "message": "Invalid JSON data"}, status=400
        )
    except Exception as e:
        logger.error(f"Error editing transcript for job {job_id}: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@login_required
def transcript_editor_view(request, job_id):
    """Render the transcript editor page."""
    job = get_object_or_404(VideoJob, job_id=job_id, user=request.user)
    return render(
        request,
        "video_processor/transcript_editor.html",
        {"job": job, "job_id": str(job.job_id)},
    )


def health_check(request):
    """Health check endpoint for Docker container monitoring."""
    try:
        # Check database connectivity
        VideoJob.objects.count()

        # Check if core services are available (safely)
        search_engine_status = "unavailable"
        try:
            # Check if search_engine is defined and available
            if (
                "search_engine" in globals()
                and search_engine is not None
                and hasattr(search_engine, "is_available")
            ):
                search_engine_status = (
                    "available" if search_engine.is_available else "unavailable"
                )
            else:
                search_engine_status = "not_initialized"
        except NameError:
            search_engine_status = "not_defined"

        # Check Phase 2 AI availability (safely)
        phase2_status = "unavailable"
        try:
            if "PHASE2_AI_AVAILABLE" in globals():
                phase2_status = "available" if PHASE2_AI_AVAILABLE else "unavailable"
        except NameError:
            phase2_status = "not_defined"

        # Check enhanced search (safely)
        enhanced_search_status = "unavailable"
        try:
            if "enhanced_search" in globals() and enhanced_search is not None:
                enhanced_search_status = "available"
        except NameError:
            enhanced_search_status = "not_defined"

        # Check RAG Q&A (safely)
        rag_qa_status = "unavailable"
        try:
            if "rag_qa" in globals() and rag_qa is not None:
                rag_qa_status = "available"
        except NameError:
            rag_qa_status = "not_defined"

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "search_engine": search_engine_status,
            "whisper": "available",  # If we got here, imports worked
            "phase2_ai": phase2_status,
            "enhanced_search": enhanced_search_status,
            "rag_qa": rag_qa_status,
        }

        return JsonResponse(health_status)

    except Exception as e:
        health_status = {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }
        return JsonResponse(health_status, status=503)


@login_required
@csrf_exempt
def api_enhanced_search(request):
    """Enhanced semantic search with better embeddings and filtering."""
    if not PHASE2_AI_AVAILABLE or enhanced_search is None:
        return JsonResponse(
            {"status": "error", "message": "Enhanced search not available"}
        )

    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"})

    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()

        if not query:
            return JsonResponse({"status": "error", "message": "Query required"})

        # Enhanced search parameters
        k = min(int(data.get("k", 10)), 50)
        min_similarity = float(data.get("min_similarity", 0.3))
        filter_topics = data.get("filter_topics", [])

        logger.info(f"Enhanced search query: '{query}' by user {request.user.username}")

        # Check if enhanced search engine is initialized
        if not enhanced_search.is_initialized:
            # Try to load from cache or build new index
            if not enhanced_search._load_index():
                # Build new enhanced index
                user_videos = VideoJob.objects.filter(
                    user=request.user, status=JobStatus.COMPLETED
                )

                segments = []
                for video in user_videos:
                    if video.transcription and "text_segments" in video.transcription:
                        for segment in video.transcription["text_segments"]:
                            segments.append(
                                {
                                    "text": segment.get("text", ""),
                                    "video_id": str(video.job_id),
                                    "video_title": video.video_name,
                                    "start_time": segment.get("start", 0),
                                    "end_time": segment.get("end", 0),
                                }
                            )

                if segments:
                    enhanced_search.build_enhanced_index(segments)

        # Perform enhanced search
        results = enhanced_search.search(
            query=query,
            k=k,
            min_similarity=min_similarity,
            filter_topics=filter_topics if filter_topics else None,
        )

        # Convert results to API format
        search_results = []
        for result in results:
            search_results.append(
                {
                    "video_id": result.video_id,
                    "video_title": result.video_title,
                    "text": result.segment_text,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "confidence": result.confidence_score,
                    "semantic_similarity": result.semantic_similarity,
                    "context_window": result.context_window,
                    "topic_tags": result.topic_tags,
                    "timestamp_formatted": f"{int(result.start_time // 60):02d}:{int(result.start_time % 60):02d}",
                }
            )

        return JsonResponse(
            {
                "status": "success",
                "results": search_results,
                "total_results": len(results),
                "query": query,
                "search_type": "enhanced_semantic",
                "processing_time": enhanced_search.stats.get("average_search_time", 0),
                "model_info": {
                    "model_name": enhanced_search.model_name,
                    "total_segments": enhanced_search.stats.get("index_size", 0),
                },
            }
        )

    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        return JsonResponse(
            {"status": "error", "message": f"Enhanced search failed: {str(e)}"},
            status=500,
        )


@login_required
@csrf_exempt
def api_rag_question_answer(request):
    """RAG-based question answering using enhanced search and QA models."""
    if not PHASE2_AI_AVAILABLE or rag_qa is None:
        return JsonResponse(
            {"status": "error", "message": "RAG Q&A system not available"}
        )

    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"})

    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()

        if not question:
            return JsonResponse({"status": "error", "message": "Question required"})

        # QA parameters
        method = data.get("method", "auto")  # auto, extractive, generative, hybrid
        include_sources = data.get("include_sources", True)
        filter_topics = data.get("filter_topics", [])

        logger.info(f"RAG Q&A question: '{question}' by user {request.user.username}")

        # Ensure enhanced search is initialized for the user
        if not enhanced_search.is_initialized:
            return JsonResponse(
                {
                    "status": "error",
                    "message": "Enhanced search index not available. Please build index first.",
                }
            )

        # Answer question using RAG
        qa_result = rag_qa.answer_question(
            question=question,
            method=method,
            include_sources=include_sources,
            filter_topics=filter_topics if filter_topics else None,
        )

        # Format sources for API response
        sources = []
        if qa_result.sources:
            for source in qa_result.sources:
                sources.append(
                    {
                        "video_id": source.video_id,
                        "video_title": source.video_title,
                        "text": source.segment_text,
                        "start_time": source.start_time,
                        "end_time": source.end_time,
                        "confidence": source.confidence_score,
                        "topic_tags": source.topic_tags,
                        "timestamp_formatted": f"{int(source.start_time // 60):02d}:{int(source.start_time % 60):02d}",
                    }
                )

        return JsonResponse(
            {
                "status": "success",
                "question": qa_result.question,
                "answer": qa_result.answer,
                "confidence": qa_result.confidence,
                "method": qa_result.method,
                "sources": sources,
                "processing_time": qa_result.processing_time,
                "metadata": qa_result.metadata,
            }
        )

    except Exception as e:
        logger.error(f"RAG Q&A failed: {e}")
        return JsonResponse(
            {"status": "error", "message": f"Question answering failed: {str(e)}"},
            status=500,
        )


@login_required
def api_enhanced_transcription_status(request):
    """Check status of enhanced AI transcription capabilities."""
    try:
        status = {
            "phase2_available": PHASE2_AI_AVAILABLE,
            "enhanced_search": {
                "available": enhanced_search is not None,
                "initialized": (
                    enhanced_search.is_initialized if enhanced_search else False
                ),
                "stats": enhanced_search.get_stats() if enhanced_search else {},
            },
            "enhanced_whisper": {
                "available": enhanced_whisper is not None,
                "stats": enhanced_whisper.get_stats() if enhanced_whisper else {},
            },
            "rag_qa": {
                "available": rag_qa is not None,
                "stats": rag_qa.get_stats() if rag_qa else {},
            },
        }

        return JsonResponse({"status": "success", "ai_status": status})

    except Exception as e:
        logger.error(f"AI status check failed: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@login_required
@csrf_exempt
def api_rebuild_enhanced_index(request):
    """Rebuild enhanced search index with improved embeddings."""
    if not PHASE2_AI_AVAILABLE or enhanced_search is None:
        return JsonResponse(
            {"status": "error", "message": "Enhanced search not available"}
        )

    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"})

    try:
        logger.info(
            f"Rebuilding enhanced search index for user {request.user.username}"
        )

        # Get user's completed videos
        user_videos = VideoJob.objects.filter(
            user=request.user, status=JobStatus.COMPLETED
        )

        segments = []
        video_count = 0

        for video in user_videos:
            if video.transcription and "text_segments" in video.transcription:
                video_count += 1
                for segment in video.transcription["text_segments"]:
                    segments.append(
                        {
                            "text": segment.get("text", ""),
                            "video_id": str(video.job_id),
                            "video_title": video.video_name,
                            "start_time": segment.get("start", 0),
                            "end_time": segment.get("end", 0),
                        }
                    )

        if not segments:
            return JsonResponse(
                {"status": "error", "message": "No transcribed videos found to index"}
            )

        # Build enhanced index
        enhanced_search.build_enhanced_index(segments)

        return JsonResponse(
            {
                "status": "success",
                "message": f"Enhanced index rebuilt successfully",
                "videos_processed": video_count,
                "segments_indexed": len(segments),
                "model_name": enhanced_search.model_name,
                "build_time": enhanced_search.metadata.get("build_time", 0),
            }
        )

    except Exception as e:
        logger.error(f"Enhanced index rebuild failed: {e}")
        return JsonResponse(
            {"status": "error", "message": f"Index rebuild failed: {str(e)}"},
            status=500,
        )


@login_required
def enhanced_search_interface(request):
    """Render enhanced search interface for Phase 2 AI capabilities."""
    return render(
        request,
        "video_processor/enhanced_search.html",
        {
            "phase2_available": PHASE2_AI_AVAILABLE,
            "enhanced_search_ready": (
                enhanced_search.is_initialized if enhanced_search else False
            ),
            "user": request.user,
        },
    )


# PUBLIC USER-SPECIFIC SEARCH VIEWS
# These allow public access to search a specific user's video library


def public_user_search_interface(request, username):
    """Public search interface for a specific user's video library."""
    from django.contrib.auth.models import User
    from django.shortcuts import get_object_or_404

    # Check if user exists
    try:
        target_user = User.objects.get(username=username)
    except User.DoesNotExist:
        return render(
            request,
            "video_processor/user_not_found.html",
            {"username": username},
            status=404,
        )

    # Get user's video count for display
    video_count = VideoJob.objects.filter(
        user=target_user, status=JobStatus.COMPLETED
    ).count()

    return render(
        request,
        "video_processor/public_user_search.html",
        {
            "target_user": target_user,
            "username": username,
            "video_count": video_count,
            "search_mode": "basic",
        },
    )


def public_user_enhanced_search_interface(request, username):
    """Public enhanced search interface for a specific user's video library."""
    from django.contrib.auth.models import User
    from django.shortcuts import get_object_or_404

    # Check if user exists
    try:
        target_user = User.objects.get(username=username)
    except User.DoesNotExist:
        return render(
            request,
            "video_processor/user_not_found.html",
            {"username": username},
            status=404,
        )

    # Get user's video count for display
    video_count = VideoJob.objects.filter(
        user=target_user, status=JobStatus.COMPLETED
    ).count()

    return render(
        request,
        "video_processor/public_user_enhanced_search.html",
        {
            "target_user": target_user,
            "username": username,
            "video_count": video_count,
            "search_mode": "enhanced",
            "phase2_available": PHASE2_AI_AVAILABLE,
            "enhanced_search_ready": (
                enhanced_search.is_initialized if enhanced_search else False
            ),
        },
    )


# PUBLIC USER-SPECIFIC SEARCH APIs


@csrf_exempt
def api_public_user_search(request, username):
    """Public search API for a specific user's video library - basic search."""
    from django.contrib.auth.models import User

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        # Check if user exists
        try:
            target_user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({"error": f'User "{username}" not found'}, status=404)

        data = json.loads(request.body)
        query = data.get("query", "").strip()

        if not query:
            return JsonResponse({"error": "Search query is required"}, status=400)

        logger.info(f"Public user search: '{query}' for user '{username}'")

        # Get user's videos for search
        user_videos = VideoJob.objects.filter(
            user=target_user, status=JobStatus.COMPLETED
        )

        if not user_videos.exists():
            return JsonResponse(
                {
                    "results": [],
                    "count": 0,
                    "query": query,
                    "username": username,
                    "message": f'No videos found for user "{username}"',
                }
            )

        # Use semantic search if available, otherwise fall back to keyword search
        if search_engine and search_engine.is_initialized:
            # Create user-specific segments for search
            user_segments = []
            for video in user_videos:
                if video.transcription and "text_segments" in video.transcription:
                    for segment in video.transcription["text_segments"]:
                        user_segments.append(
                            {
                                "text": segment.get("text", ""),
                                "video_id": str(video.job_id),
                                "video_title": video.video_name,
                                "start_time": segment.get("start", 0),
                                "end_time": segment.get("end", 0),
                                "user": target_user.username,
                            }
                        )

            # Perform hybrid search on user's content
            if user_segments:
                hybrid_results = search_engine.hybrid_search(
                    query, user_segments, top_k=20
                )
                results = convert_semantic_results_to_api_format(hybrid_results)
            else:
                results = []
        else:
            # Fall back to keyword search in user's content
            results = perform_user_keyword_search(query, target_user)

        return JsonResponse(
            {
                "results": results,
                "count": len(results),
                "query": query,
                "username": username,
                "enhanced": False,
                "message": f"Found {len(results)} results in {target_user.username}'s videos",
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in public user search API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_public_user_enhanced_search(request, username):
    """Public enhanced search API for a specific user's video library."""
    from django.contrib.auth.models import User

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        # Check if user exists
        try:
            target_user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({"error": f'User "{username}" not found'}, status=404)

        data = json.loads(request.body)
        query = data.get("query", "").strip()

        if not query:
            return JsonResponse({"error": "Search query is required"}, status=400)

        logger.info(f"Public user enhanced search: '{query}' for user '{username}'")

        # Get user's videos
        user_videos = VideoJob.objects.filter(
            user=target_user, status=JobStatus.COMPLETED
        )

        if not user_videos.exists():
            return JsonResponse(
                {
                    "results": [],
                    "count": 0,
                    "query": query,
                    "username": username,
                    "enhanced": False,
                    "message": f'No videos found for user "{username}"',
                }
            )

        # Check if Phase 2 enhanced search is available
        if PHASE2_AI_AVAILABLE and enhanced_search:
            try:
                # Build user-specific segments for enhanced search
                user_segments = []
                for video in user_videos:
                    if video.transcription and "text_segments" in video.transcription:
                        for segment in video.transcription["text_segments"]:
                            user_segments.append(
                                {
                                    "text": segment.get("text", ""),
                                    "video_id": str(video.job_id),
                                    "video_title": video.video_name,
                                    "start_time": segment.get("start", 0),
                                    "end_time": segment.get("end", 0),
                                }
                            )

                if user_segments:
                    # Temporarily build index for this user's content
                    enhanced_search.build_enhanced_index(user_segments)
                    results = enhanced_search.search(query, k=20)

                    # Convert to API format
                    api_results = []
                    for result in results:
                        api_results.append(
                            {
                                "video_id": result.video_id,
                                "title": result.video_title,
                                "content": result.segment_text,
                                "timestamp": result.start_time,
                                "confidence_score": result.confidence_score,
                                "topic": result.topic_tags,
                                "enhanced": True,
                            }
                        )

                    return JsonResponse(
                        {
                            "results": api_results,
                            "count": len(api_results),
                            "query": query,
                            "username": username,
                            "enhanced": True,
                            "message": f"Found {len(api_results)} enhanced results in {target_user.username}'s videos",
                        }
                    )

            except Exception as e:
                logger.error(f"Enhanced search error for user {username}: {e}")
                # Fall back to regular search

        # Fall back to regular search
        if search_engine and search_engine.is_initialized:
            user_segments = []
            for video in user_videos:
                if video.transcription and "text_segments" in video.transcription:
                    for segment in video.transcription["text_segments"]:
                        user_segments.append(
                            {
                                "text": segment.get("text", ""),
                                "video_id": str(video.job_id),
                                "video_title": video.video_name,
                                "start_time": segment.get("start", 0),
                                "end_time": segment.get("end", 0),
                            }
                        )

            if user_segments:
                semantic_results = search_engine.hybrid_search(
                    query, user_segments, top_k=20
                )
                results = convert_semantic_results_to_api_format(semantic_results)
            else:
                results = []
        else:
            # Fall back to keyword search
            results = perform_user_keyword_search(query, target_user)

        return JsonResponse(
            {
                "results": results,
                "count": len(results),
                "query": query,
                "username": username,
                "enhanced": False,
                "message": f"Found {len(results)} results in {target_user.username}'s videos",
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in public user enhanced search API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_public_user_rag_qa(request, username):
    """Public RAG-based Q&A API for a specific user's video library."""
    from django.contrib.auth.models import User

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        # Check if user exists
        try:
            target_user = User.objects.get(username=username)
        except User.DoesNotExist:
            return JsonResponse({"error": f'User "{username}" not found'}, status=404)

        data = json.loads(request.body)
        question = data.get("question", "").strip()

        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)

        logger.info(f"Public user RAG Q&A: '{question}' for user '{username}'")

        # Get user's videos
        user_videos = VideoJob.objects.filter(
            user=target_user, status=JobStatus.COMPLETED
        )

        if not user_videos.exists():
            return JsonResponse(
                {
                    "question": question,
                    "answer": f"No videos found for user '{username}' to answer questions about.",
                    "confidence": 0.0,
                    "sources": [],
                    "method": "no_content",
                    "username": username,
                    "enhanced": False,
                }
            )

        # Check if RAG system is available
        if PHASE2_AI_AVAILABLE and rag_qa:
            try:
                # Build user-specific segments for RAG
                user_segments = []
                for video in user_videos:
                    if video.transcription and "text_segments" in video.transcription:
                        for segment in video.transcription["text_segments"]:
                            user_segments.append(
                                {
                                    "text": segment.get("text", ""),
                                    "video_id": str(video.job_id),
                                    "video_title": video.video_name,
                                    "start_time": segment.get("start", 0),
                                    "end_time": segment.get("end", 0),
                                }
                            )

                if user_segments:
                    # Temporarily build index for this user's content
                    enhanced_search.build_enhanced_index(user_segments)
                    result = rag_qa.answer_question(question)

                    return JsonResponse(
                        {
                            "question": question,
                            "answer": result.answer,
                            "confidence": result.confidence,
                            "sources": [
                                {
                                    "video_id": source.video_id,
                                    "title": source.video_title,
                                    "content": source.segment_text,
                                    "timestamp": source.start_time,
                                    "relevance_score": source.confidence_score,
                                }
                                for source in result.sources
                            ],
                            "method": result.method,
                            "username": username,
                            "enhanced": True,
                        }
                    )

            except Exception as e:
                logger.error(f"RAG Q&A error for user {username}: {e}")
                # Fall back to simple search

        # Fall back to simple search-based answer
        user_segments = []
        for video in user_videos:
            if video.transcription and "text_segments" in video.transcription:
                for segment in video.transcription["text_segments"]:
                    user_segments.append(
                        {
                            "text": segment.get("text", ""),
                            "video_id": str(video.job_id),
                            "video_title": video.video_name,
                            "start_time": segment.get("start", 0),
                            "end_time": segment.get("end", 0),
                        }
                    )

        if search_engine and search_engine.is_initialized and user_segments:
            search_results = search_engine.hybrid_search(
                question, user_segments, top_k=5
            )
            api_results = convert_semantic_results_to_api_format(search_results)
        else:
            api_results = perform_user_keyword_search(question, target_user)[:5]

        # Create a simple answer from search results
        answer = f"Based on {target_user.username}'s video content, here are the most relevant segments:"
        if api_results:
            answer += f" Found {len(api_results)} relevant segments."
        else:
            answer = f"I couldn't find relevant information in {target_user.username}'s videos to answer your question."

        return JsonResponse(
            {
                "question": question,
                "answer": answer,
                "confidence": 0.5 if api_results else 0.1,
                "sources": api_results[:3],  # Top 3 sources
                "method": "search_fallback",
                "username": username,
                "enhanced": False,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON data"}, status=400)
    except Exception as e:
        logger.error(f"Error in public user RAG Q&A API: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# HELPER FUNCTION FOR USER-SPECIFIC KEYWORD SEARCH


def perform_user_keyword_search(query, target_user):
    """Perform keyword search in a specific user's video library."""
    results = []
    user_videos = VideoJob.objects.filter(user=target_user, status=JobStatus.COMPLETED)

    query_lower = query.lower()

    for video in user_videos:
        if video.transcription and "text_segments" in video.transcription:
            for segment in video.transcription["text_segments"]:
                text = segment.get("text", "").lower()
                if query_lower in text:
                    results.append(
                        {
                            "video_id": str(video.job_id),
                            "title": video.video_name,
                            "content": segment.get("text", ""),
                            "timestamp": segment.get("start", 0),
                            "relevance_score": 0.8,  # Basic relevance score for keyword match
                            "search_type": "keyword",
                        }
                    )

    # Sort by relevance (for now, just by timestamp)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:20]  # Limit to top 20 results
