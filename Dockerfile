# VaultIQ Docker Configuration
# Multi-stage build for optimized production container

# Stage 1: Base Python setup
FROM python:3.9-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
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
RUN mkdir -p /app/media /app/uploads /app/search_cache /app/static

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user for security
RUN adduser --disabled-password --gecos '' vaultiq
RUN chown -R vaultiq:vaultiq /app
USER vaultiq

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "video_recall_project.wsgi:application"] 