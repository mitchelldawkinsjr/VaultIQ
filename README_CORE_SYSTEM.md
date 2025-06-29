# AskMyVideo Core Processing System 🧠⚙️

**Intelligent Video Processing Engine** - Core components for video transcription and AI search capabilities.

This is a **simple, robust video processing system** that replaces the complex, fragile architecture with something reliable and easy to maintain.

## 🎯 What Changed

### Before (Problems):
- ❌ Complex multi-service architecture 
- ❌ Fragile integration between services
- ❌ Scattered video processing logic across multiple files
- ❌ Complex React/Next.js admin interface
- ❌ Database dependencies and migrations
- ❌ Inconsistent error handling

### After (Solutions):
- ✅ **Single core processor** - robust, standalone
- ✅ **Simple file-based management** - no database required
- ✅ **Basic web interface** - single HTML page
- ✅ **Minimal dependencies** - only essential libraries
- ✅ **Self-contained** - runs anywhere Python runs
- ✅ **Comprehensive error handling** and logging

## 🚀 Quick Start

### 1. Test Core Functionality
```bash
python test_core_processor.py
```

### 2. Add and Process Videos
```bash
# Add a video
python simple_video_manager.py add --file test_data/test.mp4

# Process it
python test_manager.py
```

### 3. Use Web Interface
```bash
python video_web_interface.py
# Open browser to: http://localhost:5000
```

## 📁 Core Files

- **`core_video_processor.py`** - Standalone video processor
- **`simple_video_manager.py`** - File-based job management  
- **`video_web_interface.py`** - Single-file web interface
- **`test_core_processor.py`** - Core functionality tests
- **`test_manager.py`** - Management system tests

## ✅ Working Features

The core system is **functional right now** with:

- ✅ Video validation and metadata extraction
- ✅ Audio extraction using FFmpeg
- ✅ File-based job queue and management
- ✅ Background processing workers
- ✅ Web interface for uploads and monitoring
- ✅ Command-line tools for all operations
- ✅ Comprehensive error handling

## 🔧 Installation

```bash
# Install Flask for web interface
pip install flask

# Optional: For full video processing
pip install opencv-python openai-whisper torch

# Install FFmpeg (system package)
# Mac: brew install ffmpeg
# Ubuntu: apt install ffmpeg
```

## 📊 Architecture

```
Web Interface (Flask) 
    ↓
Simple Video Manager (File-based queue)
    ↓  
Core Video Processor (Standalone)
```

**Key Benefits:**
- No database required
- No complex service dependencies  
- Single machine deployment
- Easy to debug and maintain
- Graceful degradation when optional tools are missing

## 🎯 Next Steps

Now that you have a **working, stable core**, you can:

1. **Install optional dependencies** for full functionality
2. **Scale processing** by adding more workers
3. **Enhance the web interface** incrementally
4. **Add integrations** using the core processor
5. **Deploy** - it's just Python files and a data directory

The key is that **the video processing core is now reliable** and doesn't depend on complex infrastructure. You can build management features on top without breaking the core functionality. 