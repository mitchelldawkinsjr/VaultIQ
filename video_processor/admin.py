from django.contrib import admin
from .models import VideoJob, VideoSearchQuery


@admin.register(VideoJob)
class VideoJobAdmin(admin.ModelAdmin):
    """Admin interface for video processing jobs."""
    
    list_display = [
        'video_name', 'status', 'created_at', 'processing_time', 
        'word_count', 'language', 'duration_display'
    ]
    list_filter = ['status', 'created_at']
    search_fields = ['video_name', 'video_path']
    readonly_fields = [
        'job_id', 'created_at', 'processing_time', 
        'metadata_display', 'transcription_display'
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('job_id', 'video_name', 'video_path', 'file_size_bytes')
        }),
        ('Status & Timing', {
            'fields': ('status', 'created_at', 'started_at', 'completed_at', 'processing_time')
        }),
        ('Processing Results', {
            'fields': ('metadata_display', 'transcription_display'),
            'classes': ('collapse',)
        }),
        ('Error Handling', {
            'fields': ('error_message', 'processing_errors'),
            'classes': ('collapse',)
        }),
    )
    
    def duration_display(self, obj):
        """Display video duration in a readable format."""
        if obj.duration_seconds:
            minutes = int(obj.duration_seconds // 60)
            seconds = int(obj.duration_seconds % 60)
            return f"{minutes}:{seconds:02d}"
        return "-"
    duration_display.short_description = "Duration"
    
    def metadata_display(self, obj):
        """Display metadata in a readable format."""
        if obj.metadata:
            return f"Resolution: {obj.resolution}, Duration: {obj.duration_seconds:.1f}s, Size: {obj.metadata.get('file_size_bytes', 0)/1024/1024:.1f}MB"
        return "No metadata"
    metadata_display.short_description = "Video Metadata"
    
    def transcription_display(self, obj):
        """Display transcription summary."""
        if obj.transcription_text:
            preview = obj.transcription_text[:100] + "..." if len(obj.transcription_text) > 100 else obj.transcription_text
            return f"Words: {obj.word_count}, Language: {obj.language}, Preview: {preview}"
        return "No transcription"
    transcription_display.short_description = "Transcription Summary"


@admin.register(VideoSearchQuery)
class VideoSearchQueryAdmin(admin.ModelAdmin):
    """Admin interface for search query tracking."""
    
    list_display = ['query', 'results_count', 'created_at']
    list_filter = ['created_at', 'results_count']
    search_fields = ['query']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
