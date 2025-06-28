from django.urls import path
from . import views

urlpatterns = [
    # Main video library page (admin interface)
    path('admin/', views.VideoLibraryView.as_view(), name='video_library'),
    
    # Clean search interface (user-facing)
    path('', views.clean_search_interface, name='clean_search'),
    path('api/search/', views.api_clean_search, name='api_clean_search'),
    path('api/video/<uuid:video_id>/', views.api_video_info, name='api_video_info'),
    path('video-file/<uuid:video_id>/', views.serve_video_file, name='serve_video_file'),
    
    # Legacy library interface
    path('library/', views.library_interface, name='legacy_library'),
    
    # Video search (admin)
    path('search/', views.search_videos, name='search_videos'),
    
    # Video upload and processing
    path('upload/', views.upload_video, name='upload_video'),
    
    # Video management
    path('delete/<uuid:job_id>/', views.delete_video, name='delete_video'),
    path('video/<uuid:job_id>/', views.video_player_page, name='video_player'),
    
    # Video serving
    path('video-file/<uuid:job_id>/', views.serve_video, name='serve_video'),
    
    # API endpoints
    path('api/jobs/', views.api_jobs, name='api_jobs'),
    path('api/jobs/latest/', views.api_latest_job, name='api_latest_job'),
    path('api/video/<uuid:job_id>/', views.api_video_details, name='api_video_details'),
    path('api/search-status/', views.api_search_status, name='api_search_status'),
    
    # Admin API endpoints
    path('api/rebuild-search-index/', views.api_rebuild_search_index, name='api_rebuild_search_index'),
    path('api/cleanup-youtube/', views.api_cleanup_youtube, name='api_cleanup_youtube'),
    path('api/process-job/', views.api_process_job, name='api_process_job'),
    path('api/timestamp-content/<uuid:job_id>/', views.api_timestamp_content, name='api_timestamp_content'),
    path('api/detailed-stats/', views.api_detailed_stats, name='api_detailed_stats'),
    path('api/pending-jobs/', views.api_pending_jobs, name='api_pending_jobs'),
    
    # AI Enhancement API endpoints
    path('api/ai/enhanced-search/', views.api_ai_enhanced_search, name='api_ai_enhanced_search'),
    path('api/ai/question-answer/', views.api_ai_question_answer, name='api_ai_question_answer'),
    path('api/ai/status/', views.api_ai_status, name='api_ai_status'),
] 