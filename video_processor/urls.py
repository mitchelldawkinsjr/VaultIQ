from django.urls import path, include
from . import views

urlpatterns = [
    # Main interfaces
    path('', views.search_interface_view, name='search_interface'),
    path('library/', views.VideoLibraryView.as_view(), name='library'),
    
    # Authentication
    path('accounts/', include('django.contrib.auth.urls')),
    path('register/', views.register_view, name='register'),
    
    # Video upload and processing
    path('upload/', views.upload_video, name='upload_video'),
    
    # Transcript editing
    path('edit-transcript/<str:job_id>/', views.edit_transcript, name='edit_transcript'),
    path('transcript-editor/<str:job_id>/', views.transcript_editor_view, name='transcript_editor'),
    
    # Search functionality
    path('api/search/', views.api_clean_search, name='api_search'),
    path('search/', views.search_videos, name='search_videos'),
    
    # Video serving and info
    path('video/<str:job_id>/', views.serve_video_file, name='serve_video'),
    path('api/video-info/<uuid:video_id>/', views.api_video_info, name='api_video_info'),
    
    # Health check for Docker
    path('health/', views.health_check, name='health_check'),
    
    # Admin API endpoints (existing)
    path('api/jobs/', views.api_jobs, name='api_jobs'),
    path('api/jobs/latest/', views.api_latest_job, name='api_latest_job'),
    path('api/video/<uuid:job_id>/', views.api_video_details, name='api_video_details'),
    path('api/search-status/', views.api_search_status, name='api_search_status'),
    path('api/rebuild-search-index/', views.api_rebuild_search_index, name='api_rebuild_search_index'),
    path('api/cleanup-youtube/', views.api_cleanup_youtube, name='api_cleanup_youtube'),
    path('api/process-job/', views.api_process_job, name='api_process_job'),
    path('api/timestamp-content/<uuid:job_id>/', views.api_timestamp_content, name='api_timestamp_content'),
    path('api/detailed-stats/', views.api_detailed_stats, name='api_detailed_stats'),
    path('api/pending-jobs/', views.api_pending_jobs, name='api_pending_jobs'),
    
    # AI enhancement APIs
    path('api/ai-enhanced-search/', views.api_ai_enhanced_search, name='api_ai_enhanced_search'),
    path('api/ai-question-answer/', views.api_ai_question_answer, name='api_ai_question_answer'),
    path('api/ai-status/', views.api_ai_status, name='api_ai_status'),
] 