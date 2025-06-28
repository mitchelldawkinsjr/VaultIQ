from django.urls import path, include
from . import views

urlpatterns = [
    # PUBLIC INTERFACES (No authentication required)
    path('', views.clean_search_interface, name='public_search'),  # Public search as default
    path('public/enhanced/', views.public_enhanced_search_interface, name='public_enhanced_search'),
    
    # PUBLIC USER-SPECIFIC SEARCH (No authentication required, but searches specific user's videos)
    path('search/<str:username>/', views.public_user_search_interface, name='public_user_search'),
    path('search/<str:username>/enhanced/', views.public_user_enhanced_search_interface, name='public_user_enhanced_search'),
    
    # PRIVATE/TENANT INTERFACES (Authentication required)
    path('my-videos/', views.search_interface_view, name='private_search_interface'),
    path('library/', views.VideoLibraryView.as_view(), name='library'),
    path('enhanced-search/', views.enhanced_search_interface, name='enhanced_search'),
    
    # Authentication
    path('accounts/', include('django.contrib.auth.urls')),
    path('register/', views.register_view, name='register'),
    
    # Video upload and processing (Private)
    path('upload/', views.upload_video, name='upload_video'),
    path('delete/<uuid:job_id>/', views.delete_video, name='delete_video'),
    
    # Transcript editing (Private)
    path('edit-transcript/<str:job_id>/', views.edit_transcript, name='edit_transcript'),
    path('transcript-editor/<str:job_id>/', views.transcript_editor_view, name='transcript_editor'),
    
    # PUBLIC SEARCH APIs (No authentication required)
    path('api/search/', views.api_clean_search, name='api_public_search'),
    path('api/public/enhanced-search/', views.api_public_enhanced_search, name='api_public_enhanced_search'),
    path('api/public/rag-qa/', views.api_public_rag_question_answer, name='api_public_rag_qa'),
    
    # PUBLIC USER-SPECIFIC SEARCH APIs (No authentication required, searches specific user's videos)
    path('api/search/<str:username>/', views.api_public_user_search, name='api_public_user_search'),
    path('api/search/<str:username>/enhanced/', views.api_public_user_enhanced_search, name='api_public_user_enhanced_search'),
    path('api/search/<str:username>/rag-qa/', views.api_public_user_rag_qa, name='api_public_user_rag_qa'),
    
    # PRIVATE SEARCH APIs (Authentication required)
    path('search/', views.search_videos, name='search_videos'),  # Keep for backward compatibility
    path('api/enhanced-search/', views.api_enhanced_search, name='api_enhanced_search'),
    path('api/rag-qa/', views.api_rag_question_answer, name='api_rag_qa'),
    path('api/enhanced-status/', views.api_enhanced_transcription_status, name='api_enhanced_status'),
    path('api/rebuild-enhanced-index/', views.api_rebuild_enhanced_index, name='api_rebuild_enhanced_index'),
    
    # Video serving and info (Public access to video content)
    path('video/<str:video_id>/', views.serve_video_file, name='serve_video'),
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
    
    # AI enhancement APIs (existing)
    path('api/ai-enhanced-search/', views.api_ai_enhanced_search, name='api_ai_enhanced_search'),
    path('api/ai-question-answer/', views.api_ai_question_answer, name='api_ai_question_answer'),
    path('api/ai-status/', views.api_ai_status, name='api_ai_status'),
] 