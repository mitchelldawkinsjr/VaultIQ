import time
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core_video_processor import CoreVideoProcessor
from semantic_search import search_engine
from video_processor.models import JobStatus, VideoJob


class Command(BaseCommand):
    help = "Manage video processing jobs"

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # List command
        list_parser = subparsers.add_parser("list", help="List all video jobs")
        list_parser.add_argument(
            "--status",
            choices=["pending", "processing", "completed", "failed"],
            help="Filter by status",
        )

        # Submit command
        submit_parser = subparsers.add_parser(
            "submit", help="Submit a video for processing"
        )
        submit_parser.add_argument("video_path", help="Path to the video file")

        # Process command
        process_parser = subparsers.add_parser("process", help="Process a specific job")
        process_parser.add_argument("--job-id", required=True, help="Job ID to process")

        # Search command
        search_parser = subparsers.add_parser(
            "search", help="Search video transcriptions"
        )
        search_parser.add_argument("query", help="Search query")

        # Timestamp command
        timestamp_parser = subparsers.add_parser(
            "timestamp", help="Get content at specific timestamp"
        )
        timestamp_parser.add_argument("job_id", help="Job ID")
        timestamp_parser.add_argument(
            "timestamp", type=float, help="Timestamp in seconds"
        )

        # Delete command
        delete_parser = subparsers.add_parser("delete", help="Delete a video job")
        delete_parser.add_argument("job_id", help="Job ID to delete")

        # Stats command
        stats_parser = subparsers.add_parser("stats", help="Show processing statistics")  # noqa: F841

        # Semantic search commands
        rebuild_index_parser = subparsers.add_parser(  # noqa: F841
            "rebuild-index", help="Rebuild semantic search index"
        )
        search_stats_parser = subparsers.add_parser(  # noqa: F841
            "search-stats", help="Show search engine statistics"
        )
        ai_search_parser = subparsers.add_parser(
            "ai-search", help="Perform AI-powered semantic search"
        )
        ai_search_parser.add_argument("query", help="Natural language search query")
        ai_search_parser.add_argument(
            "--mode",
            choices=["semantic", "hybrid"],
            default="hybrid",
            help="Search mode (semantic or hybrid)",
        )

        # Storage management commands
        cleanup_parser = subparsers.add_parser(
            "cleanup-youtube", help="Clean up YouTube video files to save storage"
        )
        cleanup_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without actually deleting",
        )

        # AI Enhancement commands
        ai_status_parser = subparsers.add_parser(  # noqa: F841
            "ai-status", help="Check AI system status and configuration"
        )

        ai_enhanced_search_parser = subparsers.add_parser(
            "ai-enhanced-search",
            help="Perform AI-enhanced search with Q/A pairs and topic classification",
        )
        ai_enhanced_search_parser.add_argument("query", help="Search query")
        ai_enhanced_search_parser.add_argument(
            "--use-openai",
            action="store_true",
            default=False,
            help="Use OpenAI instead of Hugging Face",
        )
        ai_enhanced_search_parser.add_argument(
            "--mode",
            choices=["semantic", "hybrid"],
            default="hybrid",
            help="Search mode",
        )

        ai_qa_parser = subparsers.add_parser(
            "ai-qa", help="Ask a question about your videos using AI"
        )
        ai_qa_parser.add_argument("question", help="Question to ask")
        ai_qa_parser.add_argument(
            "--context-limit",
            type=int,
            default=3,
            help="Maximum number of video segments to use as context",
        )

    def handle(self, *args, **options):
        command = options.get("command")

        if command == "list":
            self.handle_list(options)
        elif command == "submit":
            self.handle_submit(options)
        elif command == "process":
            self.handle_process(options)
        elif command == "search":
            self.handle_search(options)
        elif command == "timestamp":
            self.handle_timestamp(options)
        elif command == "delete":
            self.handle_delete(options)
        elif command == "stats":
            self.handle_stats(options)
        elif command == "rebuild-index":
            self.handle_rebuild_index(options)
        elif command == "search-stats":
            self.handle_search_stats(options)
        elif command == "ai-search":
            self.handle_ai_search(options)
        elif command == "cleanup-youtube":
            self.handle_cleanup_youtube(options)
        elif command == "ai-status":
            self.handle_ai_status(options)
        elif command == "ai-enhanced-search":
            self.handle_ai_enhanced_search(options)
        elif command == "ai-qa":
            self.handle_ai_qa(options)
        else:
            self.print_help()

    def handle_list(self, options):
        """List all video jobs."""
        queryset = VideoJob.objects.all().order_by("-created_at")

        if options.get("status"):
            queryset = queryset.filter(status=options["status"])

        jobs = list(queryset)

        if not jobs:
            self.stdout.write("No video jobs found.")
            return

        self.stdout.write("\n📚 Video Library ({len(jobs)} jobs)")
        self.stdout.write("=" * 50)

        for job in jobs:
            status_emoji = {
                "pending": "⏳",
                "processing": "⚙️",
                "completed": "✅",
                "failed": "❌",
            }.get(job.status, "❓")

            self.stdout.write(f"\n{status_emoji} {job.video_name}")
            self.stdout.write(f"   ID: {job.job_id}")
            self.stdout.write(f"   Status: {job.status}")
            self.stdout.write(
                f"   Created: {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if job.status == JobStatus.COMPLETED:
                self.stdout.write("   Duration: {job.duration_seconds:.1f}s")
                self.stdout.write("   Words: {job.word_count}")
                self.stdout.write("   Language: {job.language}")
                if job.processing_time:
                    self.stdout.write("   Processing Time: {job.processing_time:.2f}s")

            if job.error_message:
                self.stdout.write("   Error: {job.error_message}")

    def handle_submit(self, options):
        """Submit a video for processing."""
        video_path = Path(options["video_path"])

        if not video_path.exists():
            raise CommandError(f"Video file not found: {video_path}")

        # Create job
        job = VideoJob.objects.create(
            video_path=str(video_path),
            video_name=video_path.name,
            file_size_bytes=video_path.stat().st_size,
            status=JobStatus.PENDING,
        )

        self.stdout.write("✅ Video submitted for processing")
        self.stdout.write("   Job ID: {job.job_id}")
        self.stdout.write("   Video: {job.video_name}")
        self.stdout.write("   Status: {job.status}")

        # Ask if user wants to process immediately
        if self.confirm("Process video now?"):
            self.process_video_job(job.job_id)

    def handle_process(self, options):
        """Process a specific job."""
        job_id = options["job_id"]

        try:
            job = VideoJob.objects.get(job_id=job_id)
        except VideoJob.DoesNotExist:
            raise CommandError(f"Job not found: {job_id}")

        if job.status != JobStatus.PENDING:
            raise CommandError(f"Job {job_id} is not pending (status: {job.status})")

        self.process_video_job(job.job_id)

    def handle_search(self, options):
        """Search video transcriptions."""
        query = options["query"]

        # Find videos with matching transcription text
        videos = VideoJob.objects.filter(
            status=JobStatus.COMPLETED, transcription__isnull=False
        )

        results = []
        for video in videos:
            transcription_text = video.transcription_text.lower()
            if query.lower() in transcription_text:
                matching_segments = video.search_segments(query)
                if matching_segments:
                    results.append((video, matching_segments))

        if not results:
            self.stdout.write(f"❌ No results found for '{query}'")
            return

        self.stdout.write(f"\n🎯 Search Results for '{query}' ({len(results)} videos)")
        self.stdout.write("=" * 60)

        for video, segments in results:
            self.stdout.write(f"\n📹 {video.video_name}")
            self.stdout.write(f"   Job ID: {video.job_id}")
            self.stdout.write(f"   Duration: {video.duration_seconds:.1f}s")

            for segment in segments:
                self.stdout.write(
                    f"   ⏰ {segment['start_time']:.1f}s - {segment['end_time']:.1f}s:"
                )
                self.stdout.write(f"      {segment['text']}")

    def handle_timestamp(self, options):
        """Get content at specific timestamp."""
        job_id = options["job_id"]
        timestamp = options["timestamp"]

        try:
            job = VideoJob.objects.get(job_id=job_id)
        except VideoJob.DoesNotExist:
            raise CommandError(f"Job not found: {job_id}")

        if job.status != JobStatus.COMPLETED:
            raise CommandError(f"Job {job_id} is not completed")

        # Find segment containing the timestamp
        segments = job.text_segments
        if not segments:
            self.stdout.write("❌ No segments found for this video")
            return

        matching_segment = None
        for segment in segments:
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)

            if start_time <= timestamp <= end_time:
                matching_segment = segment
                break

        if matching_segment:
            self.stdout.write(f"\n⏰ Content at {timestamp}s in {job.video_name}")
            self.stdout.write("=" * 50)
            self.stdout.write(
                f"Segment: {matching_segment['start']:.1f}s - {matching_segment['end']:.1f}s"
            )
            self.stdout.write(f"Text: {matching_segment['text']}")
        else:
            self.stdout.write(f"❌ No content found at {timestamp}s")

    def handle_delete(self, options):
        """Delete a video job."""
        job_id = options["job_id"]

        try:
            job = VideoJob.objects.get(job_id=job_id)
        except VideoJob.DoesNotExist:
            raise CommandError(f"Job not found: {job_id}")

        video_name = job.video_name

        if self.confirm(f"Delete '{video_name}'?"):
            job.delete()
            self.stdout.write(f"✅ Deleted job: {video_name}")
        else:
            self.stdout.write("❌ Delete cancelled")

    def handle_stats(self, options):
        """Show processing statistics."""
        total_jobs = VideoJob.objects.count()
        completed_jobs = VideoJob.objects.filter(status=JobStatus.COMPLETED).count()
        pending_jobs = VideoJob.objects.filter(status=JobStatus.PENDING).count()
        failed_jobs = VideoJob.objects.filter(status=JobStatus.FAILED).count()

        total_words = sum(
            job.word_count or 0
            for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
        )

        total_processing_time = sum(
            job.processing_time or 0
            for job in VideoJob.objects.filter(status=JobStatus.COMPLETED)
        )

        self.stdout.write("\n📊 Video Processing Statistics")
        self.stdout.write("=" * 35)
        self.stdout.write(f"Total Jobs: {total_jobs}")
        self.stdout.write(f"  ✅ Completed: {completed_jobs}")
        self.stdout.write(f"  ⏳ Pending: {pending_jobs}")
        self.stdout.write(f"  ❌ Failed: {failed_jobs}")
        self.stdout.write("\nProcessing Stats:")
        self.stdout.write(f"  📝 Total Words: {total_words:,}")
        self.stdout.write(f"  ⏱️ Total Processing Time: {total_processing_time:.1f}s")

        if completed_jobs > 0:
            avg_processing_time = total_processing_time / completed_jobs
            avg_words_per_video = total_words / completed_jobs
            self.stdout.write(f"  📈 Avg Processing Time: {avg_processing_time:.2f}s")
            self.stdout.write(f"  📈 Avg Words per Video: {avg_words_per_video:.0f}")

    def handle_rebuild_index(self, options):
        """Rebuild the semantic search index."""
        try:
            from video_processor.views import rebuild_search_index

            self.stdout.write("🔄 Rebuilding semantic search index...")
            rebuild_search_index()

            stats = search_engine.get_stats()
            if stats["is_initialized"]:
                self.stdout.write("✅ Search index rebuilt successfully")
                self.stdout.write(f"   📊 Indexed {stats['total_segments']} segments")
                self.stdout.write(f"   🧠 Model: {stats['model_name']}")
            else:
                self.stdout.write("❌ Failed to rebuild search index")

        except Exception as e:
            self.stdout.write(f"❌ Error rebuilding index: {e}")

    def handle_search_stats(self, options):
        """Show search engine statistics."""
        stats = search_engine.get_stats()

        self.stdout.write("\n🧠 Semantic Search Engine Status")
        self.stdout.write("=" * 40)
        self.stdout.write(f"Available: {'✅' if stats['is_available'] else '❌'}")
        self.stdout.write(f"Initialized: {'✅' if stats['is_initialized'] else '❌'}")

        if stats["is_available"]:
            self.stdout.write(f"Model: {stats['model_name']}")
            self.stdout.write(f"Total Segments: {stats['total_segments']:,}")
            self.stdout.write(f"Index Size: {stats['index_size']:,}")

            if not stats["is_initialized"]:
                self.stdout.write("\n💡 Run 'rebuild-index' to enable semantic search")
        else:
            self.stdout.write("\n💡 Install semantic search dependencies:")
            self.stdout.write("   pip install sentence-transformers faiss-cpu")

    def handle_ai_search(self, options):
        """Perform AI-powered semantic search."""
        query = options["query"]
        mode = options["mode"]

        if not search_engine.is_initialized:
            self.stdout.write(
                "❌ Semantic search not available. Run 'rebuild-index' first."
            )
            return

        try:
            self.stdout.write(f"🧠 Performing {mode} search for: '{query}'")

            if mode == "semantic":
                results = search_engine.semantic_search(query, top_k=20)
            else:  # hybrid
                from video_processor.views import get_video_segments_for_search

                video_segments = get_video_segments_for_search()
                results = search_engine.hybrid_search(query, video_segments, top_k=20)

            if not results:
                self.stdout.write(f"❌ No results found for '{query}'")
                return

            self.stdout.write(f"\n🎯 AI Search Results ({len(results)} segments)")
            self.stdout.write("=" * 60)

            # Group results by video
            video_groups = {}
            for result in results:
                if result.job_id not in video_groups:
                    try:
                        job = VideoJob.objects.get(job_id=result.job_id)
                        video_groups[result.job_id] = {"video": job, "segments": []}
                    except VideoJob.DoesNotExist:
                        continue

                video_groups[result.job_id]["segments"].append(result)

            for video_data in video_groups.values():
                video = video_data["video"]
                segments = video_data["segments"]

                self.stdout.write(f"\n📹 {video.video_name}")
                self.stdout.write(f"   Job ID: {video.job_id}")

                for result in segments:
                    score_display = f"{result.score:.3f}"
                    self.stdout.write(
                        f"   ⏰ {result.start_time:.1f}s - {result.end_time:.1f}s "
                        f"(relevance: {score_display}) [{result.search_type}]"
                    )
                    self.stdout.write(f"      {result.text}")

        except Exception as e:
            self.stdout.write(f"❌ Search error: {e}")

    def handle_cleanup_youtube(self, options):
        """Clean up YouTube video files to save storage."""
        dry_run = options.get("dry_run", False)

        # Find completed YouTube videos
        youtube_jobs = VideoJob.objects.filter(status=JobStatus.COMPLETED).exclude(
            video_path__startswith="[CLEANED_UP]"
        )

        total_size = 0
        cleaned_count = 0

        self.stdout.write(
            f"\n🧹 YouTube Video Cleanup {'(DRY RUN)' if dry_run else ''}"
        )
        self.stdout.write("=" * 50)

        for job in youtube_jobs:
            # Check if this looks like a YouTube video
            is_youtube = (
                "media/videos/" in job.video_path
                and any(char in job.video_path for char in ["_", "-"])
                and len(Path(job.video_path).stem.split("_")[0])
                == 11  # YouTube ID length
            )

            if is_youtube:
                try:
                    video_path = Path(job.video_path)
                    if video_path.exists():
                        file_size = video_path.stat().st_size
                        file_size_mb = file_size / (1024 * 1024)
                        total_size += file_size

                        self.stdout.write(f"📹 {job.video_name}")
                        self.stdout.write(f"   Size: {file_size_mb:.1f}MB")
                        self.stdout.write(f"   Path: {job.video_path}")

                        if not dry_run:
                            video_path.unlink()
                            job.video_path = f"[CLEANED_UP] {job.video_path}"
                            job.save()
                            self.stdout.write("   ✅ Deleted")
                        else:
                            self.stdout.write("   🔍 Would delete")

                        cleaned_count += 1

                except Exception as e:
                    self.stdout.write(f"   ❌ Error: {e}")

        total_size_mb = total_size / (1024 * 1024)

        if cleaned_count > 0:
            action = "Would free" if dry_run else "Freed"
            self.stdout.write("\n📊 Summary:")
            self.stdout.write("   Videos processed: {cleaned_count}")
            self.stdout.write("   Storage {action.lower()}: {total_size_mb:.1f}MB")

            if dry_run:
                self.stdout.write(
                    "\n💡 Run without --dry-run to actually delete the files"
                )
        else:
            self.stdout.write("\n✅ No YouTube videos found to clean up")

    def process_video_job(self, job_id):
        """Process a video job."""
        try:
            job = VideoJob.objects.get(job_id=job_id)

            self.stdout.write(f"🚀 Processing video: {job.video_name}")

            job.status = JobStatus.PROCESSING
            job.started_at = timezone.now()
            job.save()

            start_time = time.time()

            # Process video using core processor
            processor = CoreVideoProcessor()
            result = processor.create_comprehensive_video_summary(job.video_path)

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
                self.stdout.write(f"❌ Processing failed: {job.error_message}")
            else:
                job.status = JobStatus.COMPLETED
                self.stdout.write(f"✅ Processing completed in {processing_time:.2f}s")
                self.stdout.write(f"   📝 Transcribed {job.word_count} words")
                self.stdout.write(f"   🗣️ Language: {job.language}")

            job.save()

        except Exception as e:
            self.stdout.write(f"❌ Error processing job {job_id}: {e}")
            try:
                job = VideoJob.objects.get(job_id=job_id)
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = timezone.now()
                job.save()
            except:
                pass

    def confirm(self, question):
        """Ask for user confirmation."""
        try:
            answer = input(f"{question} (y/N): ").lower().strip()
            return answer in ["y", "yes"]
        except KeyboardInterrupt:
            return False

    def handle_ai_status(self, options):
        """Check AI system status and configuration."""
        try:
            from ai_enhanced_search import create_ai_config, get_ai_engine

            config = create_ai_config()
            ai_engine = get_ai_engine()

            self.stdout.write("\n🤖 AI Enhancement System Status")
            self.stdout.write("=" * 50)

            # API Configuration
            self.stdout.write("\n🔑 API Configuration:")
            self.stdout.write(
                f"   OpenAI API Key: {'✅ Configured' if config.openai_api_key else '❌ Not configured'}"
            )
            self.stdout.write(
                f"   Hugging Face Token: {'✅ Configured' if config.huggingface_api_key else '❌ Not configured'}"
            )

            if not config.openai_api_key and not config.huggingface_api_key:
                self.stdout.write("\n❌ No API keys configured!")
                self.stdout.write(
                    "   Please check AI_SETUP_INSTRUCTIONS.md for setup details."
                )
                return

            # Model Configuration
            self.stdout.write("\n🧠 Model Configuration:")
            if config.openai_api_key:
                self.stdout.write(f"   OpenAI Model: {config.openai_model}")
                self.stdout.write(f"   Max Tokens: {config.openai_max_tokens}")

            if config.huggingface_api_key:
                self.stdout.write(f"   HF Q/A Model: {config.hf_qa_model}")
                self.stdout.write(
                    f"   HF Summarization: {config.hf_summarization_model}"
                )

            # Engine Status
            self.stdout.write("\n🔧 Engine Status:")
            self.stdout.write(
                f"   AI Engine: {'✅ Initialized' if ai_engine else '❌ Not initialized'}"
            )

            if ai_engine:
                self.stdout.write("\n🎯 Available Features:")
                if config.openai_api_key:
                    self.stdout.write("   • Q/A Pair Generation (OpenAI)")
                    self.stdout.write("   • Topic Classification (OpenAI)")
                    self.stdout.write("   • Content Summarization (OpenAI)")

                if config.huggingface_api_key:
                    self.stdout.write("   • Advanced Q/A (Hugging Face)")
                    self.stdout.write("   • Sentiment Analysis (Hugging Face)")

                self.stdout.write("\n💡 Test Commands:")
                self.stdout.write(
                    "   python manage.py video_manager ai-enhanced-search 'relationship advice'"
                )
                self.stdout.write(
                    "   python manage.py video_manager ai-qa 'How to build confidence?'"
                )

        except ImportError:
            self.stdout.write("\n❌ AI Enhancement System Not Available")
            self.stdout.write(
                "   Install dependencies: pip install aiohttp openai python-dotenv"
            )
        except Exception as e:
            self.stdout.write(f"\n❌ Error checking AI status: {e}")

    def handle_ai_enhanced_search(self, options):
        """Perform AI-enhanced search with Q/A pairs and topic classification."""
        try:
            import asyncio

            from ai_enhanced_search import enhance_search_with_ai, get_ai_engine

            ai_engine = get_ai_engine()
            if not ai_engine:
                self.stdout.write(
                    "❌ AI engine not available. Run 'ai-status' to check configuration."
                )
                return

            query = options["query"]
            use_openai = options["use_openai"]
            search_mode = options["mode"]

            self.stdout.write(f"\n🔍 AI-Enhanced Search: '{query}'")
            self.stdout.write(f"   Mode: {search_mode}")
            self.stdout.write(
                f"   AI Provider: {'OpenAI' if use_openai else 'Hugging Face'}"
            )
            self.stdout.write("=" * 60)

            # Get regular search results first
            from video_processor.views import (
                convert_semantic_results_to_display_format,
                get_video_segments_for_search,
                perform_keyword_search,
            )

            search_results = []

            if search_mode == "semantic" and search_engine.is_initialized:
                semantic_results = search_engine.semantic_search(query, top_k=10)
                search_results = convert_semantic_results_to_display_format(
                    semantic_results
                )
            else:
                search_results = perform_keyword_search(query)

            if not search_results:
                self.stdout.write("❌ No results found for your query.")
                return

            # Convert to AI format
            ai_input_results = search_results[:5]  # Limit for processing

            self.stdout.write(
                f"🤖 Applying AI enhancements to {len(ai_input_results)} results..."
            )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            enhanced_results = loop.run_until_complete(
                enhance_search_with_ai(ai_input_results, use_openai)
            )

            # Display enhanced results
            self.stdout.write(f"\n✨ Enhanced Results ({len(enhanced_results)} items):")

            for i, result in enumerate(enhanced_results, 1):
                self.stdout.write(f"\n{i}. 📹 {result.title}")
                self.stdout.write(f"   ⏰ Timestamp: {result.timestamp:.1f}s")
                self.stdout.write(f"   🎯 Relevance: {result.relevance_score:.2f}")
                self.stdout.write(f"   📝 Content: {result.content[:150]}...")

                if result.topic_classification:
                    self.stdout.write(f"   🏷️  Topic: {result.topic_classification}")

                if result.summary:
                    self.stdout.write(f"   📋 Summary: {result.summary}")

                if result.generated_questions:
                    self.stdout.write(f"   ❓ Generated Questions:")
                    for q in result.generated_questions[:2]:
                        self.stdout.write(f"      • {q}")

                if result.confidence_score:
                    self.stdout.write(
                        f"   🎯 AI Confidence: {result.confidence_score:.2f}"
                    )

        except ImportError:
            self.stdout.write(
                "❌ AI features not available. Install dependencies first."
            )
        except Exception as e:
            self.stdout.write(f"❌ AI enhanced search failed: {e}")

    def handle_ai_qa(self, options):
        """Ask a question about videos using AI."""
        try:
            import asyncio

            from ai_enhanced_search import get_ai_engine

            ai_engine = get_ai_engine()
            if not ai_engine:
                self.stdout.write(
                    "❌ AI engine not available. Run 'ai-status' to check configuration."
                )
                return

            question = options["question"]
            context_limit = options["context_limit"]

            self.stdout.write(f"\n❓ Question: {question}")
            self.stdout.write("=" * 60)

            # Search for relevant context
            self.stdout.write("🔍 Searching for relevant context...")

            from video_processor.views import perform_keyword_search

            # Use keyword search to find relevant content
            context_data = []
            videos = VideoJob.objects.filter(
                status=JobStatus.COMPLETED, transcription__isnull=False
            )

            for video in videos:
                if any(
                    word.lower() in video.transcription_text.lower()
                    for word in question.split()
                ):
                    segments = video.search_segments(
                        " ".join(question.split()[:3])
                    )  # Use first 3 words
                    for segment in segments[:2]:  # Limit segments per video
                        context_data.append(
                            {
                                "content": segment["text"],
                                "title": video.video_name,
                                "timestamp": segment["start_time"],
                            }
                        )
                    if len(context_data) >= context_limit:
                        break

            if not context_data:
                self.stdout.write("❌ No relevant context found for your question.")
                return

            self.stdout.write(f"📋 Found {len(context_data)} relevant segments")

            # Get AI answer
            self.stdout.write("🤖 Generating AI answer...")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            answer = loop.run_until_complete(
                ai_engine.answer_question(question, context_data)
            )

            if answer:
                self.stdout.write(f"\n💡 AI Answer:")
                self.stdout.write("=" * 40)
                self.stdout.write(answer)

                self.stdout.write(f"\n📚 Based on context from:")
                for i, context in enumerate(context_data, 1):
                    self.stdout.write(
                        f"   {i}. {context['title']} (at {context['timestamp']:.1f}s)"
                    )
            else:
                self.stdout.write(
                    "❌ Could not generate an answer. Try rephrasing your question."
                )

        except ImportError:
            self.stdout.write(
                "❌ AI features not available. Install dependencies first."
            )
        except Exception as e:
            self.stdout.write(f"❌ AI Q/A failed: {e}")

    def print_help(self):
        """Print help information."""
        self.stdout.write(
            """
🎥 Video Manager Commands:

Basic Commands:
  list [--status STATUS]     List all video jobs
  submit VIDEO_PATH          Submit a video for processing
  process --job-id JOB_ID    Process a specific job
  search QUERY               Search video transcriptions (keyword)
  timestamp JOB_ID TIME      Get content at specific timestamp
  delete JOB_ID              Delete a video job
  stats                      Show processing statistics

🧠 AI Search Commands:
  ai-search QUERY [--mode]   Perform AI-powered semantic search
  rebuild-index              Rebuild semantic search index
  search-stats               Show search engine statistics

🧹 Storage Management:
  cleanup-youtube [--dry-run] Clean up YouTube video files to save storage

Examples:
  python manage.py video_manager list
  python manage.py video_manager submit /path/to/video.mp4
  python manage.py video_manager search "smoking tire"
  python manage.py video_manager ai-search "videos about relationships"
  python manage.py video_manager ai-search "cooking techniques" --mode semantic
  python manage.py video_manager rebuild-index
  python manage.py video_manager cleanup-youtube --dry-run
  python manage.py video_manager timestamp abc123 30.5
  python manage.py video_manager delete abc123
        """
        )
