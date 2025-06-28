# VaultIQ AI Enhancement Setup 🧠🚀

**Advanced AI Integration Guide** - Configure OpenAI and Hugging Face APIs for enhanced intelligent search capabilities.

## Overview
The AI enhancement system adds advanced capabilities to the video search functionality, including:
- **Question/Answer Generation**: Auto-generate relevant questions from video content
- **Topic Classification**: Automatically categorize video content by topic
- **Content Summarization**: Generate concise summaries of video segments
- **Sentiment Analysis**: Analyze emotional tone of content
- **Enhanced Search**: Combine AI insights with semantic search

**New in this version**: The system now **gracefully falls back to regular search** when AI tokens are not available, ensuring continuous functionality without requiring API keys.

## Fallback Behavior

### When No AI Tokens Are Available
- ✅ **System continues to work normally**
- ✅ **Regular semantic and keyword search remains fully functional**
- ✅ **No error messages or failed requests**
- ✅ **API responses include fallback indicators**
- ✅ **Search results returned in consistent format**

### API Response Differences
With AI tokens:
```json
{
  "ai_enhanced": true,
  "generated_questions": ["What are the key points?", "How does this apply?"],
  "topic_classification": "personal development",
  "summary": "This segment discusses...",
  "ai_provider": "openai"
}
```

Without AI tokens (fallback):
```json
{
  "ai_enhanced": false,
  "generated_questions": [],
  "topic_classification": null,
  "summary": null,
  "fallback_reason": "No AI tokens available",
  "ai_provider": null
}
```

## Setup Options

### Option 1: No Setup Required (Default Fallback)
- **Cost**: Free
- **Setup Time**: 0 minutes
- **Functionality**: Full search capabilities without AI enhancements
- **Perfect for**: Testing, development, or users who don't need AI features

### Option 2: OpenAI API (Recommended)
- **Cost**: ~$0.01-0.05 per search (pay-as-you-go)
- **Setup Time**: 5 minutes
- **Best Features**: Question generation, summarization, topic classification

### Option 3: Hugging Face API (Free Tier Available)
- **Cost**: Free for 1000 requests/month, then paid
- **Setup Time**: 3 minutes
- **Best Features**: Question answering, sentiment analysis, topic classification

### Option 4: Both APIs (Maximum Features)
- **Cost**: Combined costs, automatic failover
- **Setup Time**: 8 minutes
- **Best Features**: All AI capabilities with redundancy

## Installation

### 1. Install Dependencies

```bash
pip install aiohttp openai python-dotenv
```

Or add to your requirements.txt:
```
aiohttp>=3.8.0
openai>=1.0.0
python-dotenv>=1.0.0
```

### 2. Environment Configuration

Create a `.env` file in your project root:

```bash
# Option 1: No setup - system will work without these
# (Leave blank or don't create .env file)

# Option 2: OpenAI only
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=150

# Option 3: Hugging Face only
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Option 4: Both services (recommended)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Optional: Rate limiting
AI_MAX_REQUESTS_PER_MINUTE=60
AI_REQUEST_DELAY=1.0
```

## Getting API Keys

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login and go to API section
3. Create a new API key
4. Add billing information (required for API access)
5. **Cost**: ~$0.01-0.05 per enhanced search

### Hugging Face API Token  
1. Go to [Hugging Face](https://huggingface.co/)
2. Sign up/login and go to Settings → Access Tokens
3. Create a new token with "Inference API" permission
4. **Free tier**: 1000 requests/month

## Verification

### Check System Status
```bash
# Test the fallback functionality
python test_ai_fallback.py

# Check via web API
curl http://localhost:8000/api/ai/status/
```

### Example Status Responses

**No tokens (fallback mode)**:
```json
{
  "available": true,
  "system_installed": true,
  "tokens_available": false,
  "fallback_mode": true,
  "status": "No AI tokens - fallback to regular search"
}
```

**With tokens**:
```json
{
  "available": true,
  "system_installed": true,
  "tokens_available": true,
  "openai_token_available": true,
  "huggingface_token_available": false,
  "fallback_mode": false,
  "status": "AI tokens available - full functionality"
}
```

### Test Searches

```bash
# AI-enhanced search (will fallback if no tokens)
curl -X POST http://localhost:8000/api/ai/enhanced-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "relationship advice", "mode": "hybrid"}'

# Q&A (will return context only if no tokens)
curl -X POST http://localhost:8000/api/ai/question-answer/ \
  -H "Content-Type: application/json" \
  -d '{"question": "How to improve communication?"}'
```

## Usage

### Enhanced Search API
```python
# Will work with or without AI tokens
response = requests.post('/api/ai/enhanced-search/', {
    'query': 'personal growth',
    'mode': 'hybrid',
    'use_openai': True
})

# Check if AI was actually used
if response.json()['ai_enhanced']:
    # AI features available
    questions = response.json()['results'][0]['generated_questions']
else:
    # Fallback mode - regular search results
    fallback_reason = response.json().get('fallback_reason')
```

### Question & Answer API
```python
# Will return context even without AI tokens
response = requests.post('/api/ai/question-answer/', {
    'question': 'What is this video about?',
    'context_limit': 3
})

if response.json()['ai_available'] and response.json()['answer']:
    # AI-generated answer available
    answer = response.json()['answer']
else:
    # No AI - use context manually
    context = response.json()['context_results']
    suggestion = response.json().get('suggestion', '')
```

## Cost Management

### OpenAI Costs
- **gpt-3.5-turbo**: ~$0.001-0.002 per request
- **Monthly estimate**: $3-10 for moderate usage (100-500 searches)
- **Rate limiting**: Automatically managed to prevent overage

### Hugging Face Costs
- **Free tier**: 1000 requests/month (≈33 searches/day)
- **Paid tier**: $0.001-0.01 per request after free tier
- **No billing required** for free tier

### Cost Optimization Tips
1. **Start without tokens** - test fallback functionality first
2. **Use free tier limits** - Hugging Face provides good free allowance
3. **Monitor usage** - Check API usage dashboards regularly
4. **Set rate limits** - Prevent unexpected costs with rate limiting

## Troubleshooting

### Common Issues

**"No AI tokens available"**
- ✅ **This is normal and expected** if you haven't set up API keys
- ✅ **System continues working** with regular search
- ✅ **No action required** unless you want AI features

**"AI enhancement failed"**
- Check API key validity
- Verify internet connection
- Check rate limits
- Review server logs for detailed errors

**High costs**
- Adjust rate limiting in .env file
- Monitor API usage dashboards
- Consider switching to free tier models

### Support
- **System works without any setup** - fallback mode ensures reliability
- **Gradual enhancement** - add AI features when ready
- **No breaking changes** - existing functionality preserved
- **Flexible configuration** - use only the APIs you need 