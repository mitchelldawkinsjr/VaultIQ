# AskMyVideo Docker Configuration
# Multi-stage build for optimized production container

# Stage 1: Base Python setup
FROM python:3.9-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as deps
COPY core_requirements.txt .
RUN pip install --no-cache-dir -r core_requirements.txt

# Stage 3: Application
FROM deps as app
COPY . .

# Create directories for media and cache
RUN mkdir -p /app/media /app/uploads /app/search_cache /app/static /app/staticfiles
RUN mkdir -p /app/video_data

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user for security
RUN adduser --disabled-password --gecos '' askmyvideo
RUN chown -R askmyvideo:askmyvideo /app
USER askmyvideo

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ || exit 1

EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "video_recall_project.wsgi:application"] 