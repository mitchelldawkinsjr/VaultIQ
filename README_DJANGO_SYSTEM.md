# VaultIQ Django System Documentation 🧠🎥

**Intelligent Video Search & Recall System** - Technical documentation for the Django backend implementation.

## Overview

This project has been successfully migrated from a Flask-based system to a robust Django framework, providing enhanced video processing capabilities with a professional web interface, powerful admin panel, and comprehensive API.

## 🏗️ Architecture

### **Three-Layer Django Architecture**

1. **Web Interface** (`video_processor/templates/`) - Modern, responsive UI
2. **Django Models** (`video_processor/models.py`) - Database-backed job management
3. **Core Processor** (`core_video_processor.py`) - Unchanged video processing engine

### **Key Components**

- **Django Models**: Database models for video jobs and search tracking
- **Django Views**: Web interface and API endpoints
- **Django Admin**: Professional admin panel for job management
- **Management Commands**: Command-line interface for video operations
- **Templates**: Modern HTML interface with interactive features

## 🚀 Getting Started

### **1. Start the Django Server**
```bash
python3 manage.py runserver
```

### **2. Access the Application**
- **Main Interface**: http://localhost:8000/
- **Admin Panel**: http://localhost:8000/admin/ (username: admin)

### **3. Command Line Interface**
```bash
# List all videos
python3 manage.py video_manager list

# Search videos
python3 manage.py video_manager search "your query"

# Get processing statistics
python3 manage.py video_manager stats

# Submit new video
python3 manage.py video_manager submit /path/to/video.mp4

# Delete video
python3 manage.py video_manager delete JOB_ID
```

## 📊 Features

### **Web Interface Features**
- ✅ **Drag & Drop Upload**: Modern file upload with progress tracking
- ✅ **Real-time Search**: Search across all video transcriptions
- ✅ **Interactive Timeline**: Click-to-play timestamps
- ✅ **Video Library**: Grid view of all processed videos
- ✅ **Processing Statistics**: Real-time stats dashboard
- ✅ **Responsive Design**: Works on all devices

### **Django Admin Features**
- ✅ **Professional Interface**: Django's powerful admin panel
- ✅ **Job Management**: View, edit, and manage video jobs
- ✅ **Search & Filtering**: Advanced search capabilities
- ✅ **Bulk Operations**: Process multiple jobs at once
- ✅ **Detailed Views**: Comprehensive job information

### **API Features**
- ✅ **RESTful Endpoints**: JSON API for all operations
- ✅ **Job Listing**: `/api/jobs/` - List all video jobs
- ✅ **Video Details**: `/api/video/<job_id>/` - Detailed video information
- ✅ **Search Integration**: API-powered search functionality

### **Command Line Features**
- ✅ **Full CLI Interface**: All operations available via command line
- ✅ **Batch Processing**: Process multiple videos efficiently  
- ✅ **Search & Navigation**: Text search with timestamp results
- ✅ **Statistics**: Comprehensive processing analytics

## 🛠️ Django Models

### **VideoJob Model**
```python
class VideoJob(models.Model):
    job_id = models.UUIDField(primary_key=True)
    video_path = models.CharField(max_length=500)
    video_name = models.CharField(max_length=255)
    file_size_bytes = models.BigIntegerField()
    status = models.CharField(choices=JobStatus.choices)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True)
    processing_time = models.FloatField(null=True)
    metadata = models.JSONField(null=True)  # Video metadata
    transcription = models.JSONField(null=True)  # Transcription results
    processing_errors = models.JSONField(default=list)
```

### **Key Properties**
- `transcription_text` - Full transcription text
- `text_segments` - Timestamped segments
- `word_count` - Number of transcribed words
- `language` - Detected language
- `duration_seconds` - Video duration
- `resolution` - Video resolution

## 🔧 Management Commands

### **Available Commands**

```bash
# List videos (with optional status filter)
python3 manage.py video_manager list [--status completed]

# Submit video for processing
python3 manage.py video_manager submit VIDEO_PATH

# Process specific job
python3 manage.py video_manager process --job-id JOB_ID

# Search transcriptions
python3 manage.py video_manager search "search query"

# Get content at timestamp
python3 manage.py video_manager timestamp JOB_ID 30.5

# Delete video job
python3 manage.py video_manager delete JOB_ID

# Show statistics
python3 manage.py video_manager stats
```

### **Example Usage**
```bash
# Submit and process a video
python3 manage.py video_manager submit test_data/test.mp4

# Search for specific content
python3 manage.py video_manager search "smoking tire"

# Get statistics
python3 manage.py video_manager stats
```

## 🌐 API Endpoints

### **Main Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main video library interface |
| `/search/` | POST | Search video transcriptions |
| `/upload/` | POST | Upload new video |
| `/delete/<job_id>/` | POST | Delete video job |
| `/api/jobs/` | GET | List all jobs (JSON) |
| `/api/video/<job_id>/` | GET | Video details (JSON) |
| `/admin/` | GET | Django admin panel |

### **API Response Examples**

**Job List (`/api/jobs/`)**:
```json
[
  {
    "job_id": "abc123-def456",
    "video_name": "example.mp4",
    "status": "completed",
    "created_at": "2025-06-27T19:03:45+00:00",
    "processing_time": 3.64,
    "has_transcript": true,
    "word_count": 139,
    "language": "en"
  }
]
```

**Video Details (`/api/video/<job_id>/`)**:
```json
{
  "job_id": "abc123-def456",
  "video_name": "example.mp4",
  "status": "completed",
  "metadata": {
    "duration_seconds": 47.5,
    "width_pixels": 1280,
    "height_pixels": 720,
    "file_size_bytes": 13183260
  },
  "transcription": {
    "text": "Full transcription text...",
    "language": "en",
    "word_count": 139
  },
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 3.4,
      "text": "Segment text...",
      "confidence": 0.95
    }
  ]
}
```

## 💾 Database Management

### **Database Operations**
```bash
# Create migrations for model changes
python3 manage.py makemigrations

# Apply migrations
python3 manage.py migrate

# Create admin user
python3 manage.py createsuperuser

# Django shell for advanced operations
python3 manage.py shell
```

### **Data Migration**
The system automatically migrated existing JSON-based jobs to Django models:
- ✅ All existing video jobs preserved
- ✅ Transcriptions and metadata maintained
- ✅ Processing statistics retained
- ✅ Search functionality enhanced

## 🔍 Search Capabilities

### **Web Interface Search**
- Real-time search across all video transcriptions
- Highlighted search results with context
- Interactive timestamp navigation
- Click-to-play functionality

### **Command Line Search**
```bash
# Basic search
python3 manage.py video_manager search "smoking tire"

# Results show timestamps and context
🎯 Search Results for 'smoking tire' (1 videos)
📹 test.mp4
   ⏰ 0.0s - 3.4s: Last year the smoking tire went on...
```

### **Timestamp Navigation**
```bash
# Get content at specific time
python3 manage.py video_manager timestamp JOB_ID 30.5

⏰ Content at 30.5s in video.mp4
Segment: 28.2s - 32.1s
Text: "Specific content at that timestamp..."
```

## 📈 Processing Statistics

### **Real-time Statistics**
- Total videos processed
- Total words transcribed
- Total processing time
- Average processing metrics
- Job status distribution

### **Example Output**
```
📊 Video Processing Statistics
===================================
Total Jobs: 5
  ✅ Completed: 4
  ⏳ Pending: 1
  ❌ Failed: 0

Processing Stats:
  📝 Total Words: 2,847
  ⏱️ Total Processing Time: 18.2s
  📈 Avg Processing Time: 4.55s
  📈 Avg Words per Video: 712
```

## ⚙️ Configuration

### **Django Settings** (`video_recall_project/settings.py`)
```python
# Video Processing Settings
VIDEO_STORAGE_ROOT = BASE_DIR / 'video_data'
UPLOAD_ROOT = BASE_DIR / 'uploads'
MAX_VIDEO_SIZE_MB = 500
WHISPER_MODEL = 'base'
```

### **Core Video Processor**
- Unchanged from previous system
- Zero external dependencies required
- Graceful fallbacks for missing components
- Comprehensive error handling

## 🔧 Development

### **Project Structure**
```
video-processor-mvp/
├── video_recall_project/      # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── video_processor/           # Django app
│   ├── models.py             # Database models
│   ├── views.py              # Web views and API
│   ├── admin.py              # Admin interface
│   ├── urls.py               # URL patterns
│   ├── templates/            # HTML templates
│   └── management/           # Management commands
├── core_video_processor.py   # Core processing engine
├── static/                   # Static files
├── media/                    # Uploaded media
├── db.sqlite3               # SQLite database
└── manage.py                # Django management script
```

### **Running Tests**
```bash
# Run Django tests
python3 manage.py test

# Test specific components
python3 manage.py test video_processor
```

## 🚀 Deployment Considerations

### **Production Setup**
1. **Database**: Migrate to PostgreSQL for production
2. **Static Files**: Configure proper static file serving
3. **Media Files**: Set up proper media file storage
4. **Environment Variables**: Use environment-based configuration
5. **Security**: Update SECRET_KEY and security settings

### **Scaling Options**
- **Background Processing**: Add Celery for video processing
- **File Storage**: Use cloud storage (S3, GCS)
- **Database**: PostgreSQL with connection pooling
- **Caching**: Redis for search result caching
- **Load Balancing**: Multiple Django instances

## 🆕 Migration Benefits

### **Improvements Over Flask System**
- ✅ **Professional Admin Interface**: Django admin for job management
- ✅ **Database-Backed**: Proper relational database instead of JSON files
- ✅ **Better Architecture**: Model-View-Template pattern
- ✅ **Enhanced API**: RESTful API with better structure
- ✅ **Command Line Tools**: Comprehensive management commands
- ✅ **Scalability**: Built-in Django scalability features
- ✅ **Security**: Django's built-in security features
- ✅ **Testing**: Django's testing framework
- ✅ **Documentation**: Auto-generated API documentation

### **Maintained Features**
- ✅ **Core Processing**: Same reliable video processing engine
- ✅ **Search Functionality**: Enhanced search with better performance
- ✅ **Timestamp Navigation**: Improved timestamp features
- ✅ **Error Handling**: Comprehensive error management
- ✅ **Data Preservation**: All existing data migrated successfully

## 🎯 Next Steps

### **Immediate Use**
1. Start the Django server: `python3 manage.py runserver`
2. Access the web interface: http://localhost:8000/
3. Upload videos through the web interface
4. Use command line tools for bulk operations
5. Manage jobs through Django admin

### **Future Enhancements**
- Video player integration
- Advanced search filters
- Batch upload capabilities
- Export functionality
- API authentication
- Mobile application

The Django-based system provides a robust, scalable foundation for video processing with professional-grade features and excellent developer experience. 