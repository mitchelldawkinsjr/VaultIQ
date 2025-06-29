# AskMyVideo 🧠🎥

**Intelligent Video Search & Recall System** - Upload videos, search content with AI, and jump to exact moments.

## ✨ What AskMyVideo Does

- 🎥 **Smart Video Processing** - Automatic transcription with OpenAI Whisper
- 🔍 **AI-Powered Search** - Find content using natural language queries
- ⚡ **Instant Recall** - Jump to exact timestamps in videos
- 🌐 **Modern Web Interface** - Clean, intuitive search experience
- 📺 **YouTube Integration** - Process YouTube videos with automatic cleanup
- 🧮 **Semantic Search** - FAISS vector similarity with sentence transformers

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r core_requirements.txt

# Start AskMyVideo
python3 manage.py runserver

# Visit: http://localhost:8000
```

## 💡 How It Works

1. **Upload** videos or paste YouTube URLs
2. **AI processes** content and generates searchable transcripts  
3. **Search** using natural language like "relationship advice" or "personal growth"
4. **Jump** to exact moments with timestamp-based video seeking

## 🔍 Search Modes

- **🧠 Smart Search** - Hybrid AI + keyword matching
- **🤖 AI Search** - Pure semantic similarity search
- **📝 Keyword Search** - Traditional text matching

## 🏗️ Architecture

```
Clean Search UI → Django Backend → AI Search Engine → Video Database
                                      ↓
              Whisper AI ← Video Processor ← YouTube/Upload
```

## 📁 Key Components

- **`core_video_processor.py`** - Video processing & transcription engine
- **`semantic_search.py`** - AI-powered search with FAISS vectors  
- **`ai_enhanced_search.py`** - Advanced AI features (OpenAI/HuggingFace)
- **`video_processor/`** - Django app with web interface
- **`video_recall_project/`** - Django project settings

## 🎯 Features

### Core Intelligence
- 🎯 **Semantic Search** - Understand meaning, not just keywords
- 📊 **Relevance Scoring** - AI-ranked search results
- 🎪 **Multiple AI Models** - Sentence transformers, OpenAI, HuggingFace
- 💾 **Smart Storage** - Automatic YouTube cleanup saves 89% disk space

### User Experience  
- 🎨 **Modern UI** - Glassmorphism design with smooth animations
- 📱 **Mobile Responsive** - Works perfectly on any device
- ⚙️ **Admin Panel** - Advanced management tools
- 🔗 **Direct Links** - Share exact video moments with timestamps

### Technical Excellence
- ⚡ **Fast Search** - FAISS vector database for millisecond queries
- 🔄 **Auto-Reload** - Real-time updates and progress tracking
- 🛡️ **Error Handling** - Graceful fallbacks for all AI services
- 🐳 **Production Ready** - Scalable Django architecture

## 🌟 Example Searches

Try searching for:
- "How to overcome challenges"
- "Relationship advice" 
- "Personal growth and development"
- "Dealing with anxiety"
- "Building confidence"

AskMyVideo understands context and meaning, not just exact word matches!

## 🎉 Why AskMyVideo?

- ✅ **Actually Intelligent** - Real AI understanding, not just keyword matching
- ✅ **Instant Results** - Jump to exact video moments in seconds
- ✅ **Beautiful Interface** - Modern, intuitive design
- ✅ **Production Ready** - Robust Django backend with error handling
- ✅ **Cost Effective** - Free tier AI with smart fallbacks
- ✅ **Privacy Focused** - All processing can run locally

Transform how you search and recall video content with **AskMyVideo** - where intelligence meets simplicity. 🚀 