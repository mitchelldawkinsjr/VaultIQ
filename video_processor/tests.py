import json
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse

from .models import VideoJob


class VaultIQPhase1Tests(TestCase):
    """Test suite for VaultIQ Phase 1: Foundation features."""

    def setUp(self):
        """Set up test data for each test."""
        self.client = Client()

        # Create test users
        self.user1 = User.objects.create_user(
            username="testuser1", password="testpass123", email="test1@example.com"
        )
        self.user2 = User.objects.create_user(
            username="testuser2", password="testpass123", email="test2@example.com"
        )

        # Create test video jobs
        self.job1 = VideoJob.objects.create(
            user=self.user1,
            video_path="/test/video1.mp4",
            video_name="Test Video 1.mp4",  # This is what gets displayed
            title="Test Video 1",
            status="completed",
            transcript="This is a test transcript for video 1.",
        )

        self.job2 = VideoJob.objects.create(
            user=self.user2,
            video_path="/test/video2.mp4",
            video_name="Test Video 2.mp4",  # This is what gets displayed
            title="Test Video 2",
            status="completed",
            transcript="This is a test transcript for video 2.",
        )

    def test_user_registration(self):
        """Test user registration functionality."""
        response = self.client.get(reverse("register"))
        self.assertEqual(response.status_code, 200)

        # Test successful registration
        response = self.client.post(
            reverse("register"),
            {
                "username": "newuser",
                "password1": "complexpass123",
                "password2": "complexpass123",
            },
        )
        self.assertEqual(response.status_code, 302)  # Redirect after success
        self.assertTrue(User.objects.filter(username="newuser").exists())

    def test_user_login(self):
        """Test user login functionality."""
        response = self.client.get(reverse("login"))
        self.assertEqual(response.status_code, 200)

        # Test successful login
        response = self.client.post(
            reverse("login"), {"username": "testuser1", "password": "testpass123"}
        )
        self.assertEqual(response.status_code, 302)  # Redirect after success

    def test_multi_tenant_video_access(self):
        """Test that users can only access their own videos."""
        # Login as user1
        self.client.login(username="testuser1", password="testpass123")

        # Access library - should only see user1's videos
        response = self.client.get(reverse("library"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Video 1.mp4")
        self.assertNotContains(response, "Test Video 2.mp4")

        # Try to access user2's video directly - should fail
        response = self.client.get(f"/edit-transcript/{self.job2.job_id}/")
        self.assertEqual(response.status_code, 404)

    def test_authentication_required(self):
        """Test that authentication is required for protected views."""
        # Try to access library without login
        response = self.client.get(reverse("library"))
        self.assertEqual(response.status_code, 302)  # Redirect to login

        # Try to access transcript editor without login
        response = self.client.get(f"/transcript-editor/{self.job1.job_id}/")
        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_transcript_editing_get(self):
        """Test getting transcript for editing."""
        self.client.login(username="testuser1", password="testpass123")

        response = self.client.get(f"/edit-transcript/{self.job1.job_id}/")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["transcript"], "This is a test transcript for video 1.")
        self.assertEqual(data["job_id"], str(self.job1.job_id))

    def test_transcript_editing_post(self):
        """Test saving edited transcript."""
        self.client.login(username="testuser1", password="testpass123")

        new_transcript = "This is an updated test transcript for video 1."
        response = self.client.post(
            f"/edit-transcript/{self.job1.job_id}/",
            json.dumps({"transcript": new_transcript}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertEqual(data["status"], "success")

        # Verify transcript was updated in database
        self.job1.refresh_from_db()
        self.assertEqual(self.job1.transcript, new_transcript)

    def test_transcript_editor_view(self):
        """Test transcript editor page renders correctly."""
        self.client.login(username="testuser1", password="testpass123")

        response = self.client.get(f"/transcript-editor/{self.job1.job_id}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Transcript Editor")
        self.assertContains(response, str(self.job1.job_id))

    def test_health_check(self):
        """Test health check endpoint for Docker monitoring."""
        response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        self.assertEqual(data["database"], "connected")

    @patch("video_processor.views.search_engine")
    def test_health_check_with_search_engine(self, mock_search_engine):
        """Test health check includes search engine status."""
        mock_search_engine.is_available = True

        response = self.client.get("/health/")
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.content)
        self.assertEqual(data["search_engine"], "available")

    def test_video_ownership_isolation(self):
        """Test that video operations respect user ownership."""
        # Create a job for user1
        job = VideoJob.objects.create(
            user=self.user1,
            video_path="/test/owned_video.mp4",
            title="Owned Video",
            status="completed",
        )

        # Login as user2 and try to access user1's video
        self.client.login(username="testuser2", password="testpass123")

        response = self.client.get(f"/edit-transcript/{job.job_id}/")
        self.assertEqual(response.status_code, 404)  # Should not find

        # Login as user1 and should be able to access
        self.client.login(username="testuser1", password="testpass123")

        response = self.client.get(f"/edit-transcript/{job.job_id}/")
        self.assertEqual(response.status_code, 200)

    def test_user_specific_video_counts(self):
        """Test that video statistics are user-specific."""
        # Create additional videos for user1
        VideoJob.objects.create(
            user=self.user1,
            video_path="/test/video3.mp4",
            title="User1 Video 2",
            status="processing",
        )
        VideoJob.objects.create(
            user=self.user1,
            video_path="/test/video4.mp4",
            title="User1 Video 3",
            status="failed",
        )

        self.client.login(username="testuser1", password="testpass123")
        response = self.client.get(reverse("library"))

        # Should see user1's 3 videos, not user2's videos
        self.assertEqual(response.context["video_count"], 3)
        self.assertEqual(response.context["completed_count"], 1)
        self.assertEqual(response.context["processing_count"], 1)
        self.assertEqual(response.context["failed_count"], 1)


class DockerConfigTests(TestCase):
    """Test Docker configuration and setup."""

    def test_dockerfiles_exist(self):
        """Test that Docker configuration files exist."""
        import os

        # Check if Dockerfile exists
        self.assertTrue(os.path.exists("Dockerfile"))

        # Check if docker-compose.yml exists
        self.assertTrue(os.path.exists("docker-compose.yml"))

        # Check if .dockerignore exists
        self.assertTrue(os.path.exists(".dockerignore"))

    def test_requirements_include_production_deps(self):
        """Test that requirements include production dependencies."""
        with open("core_requirements.txt", "r") as f:
            requirements = f.read()

        # Check for key production dependencies
        self.assertIn("gunicorn", requirements)
        self.assertIn("celery", requirements)
        self.assertIn("redis", requirements)
        self.assertIn("Django", requirements)


class CIConfigTests(TestCase):
    """Test CI/CD configuration."""

    def test_github_actions_config_exists(self):
        """Test that GitHub Actions CI configuration exists."""
        import os

        self.assertTrue(os.path.exists(".github/workflows/ci.yml"))

    def test_ci_config_structure(self):
        """Test that CI configuration has required jobs."""
        with open(".github/workflows/ci.yml", "r") as f:
            ci_config = f.read()

        # Check for required CI jobs
        self.assertIn("test:", ci_config)
        self.assertIn("lint:", ci_config)
        self.assertIn("docker:", ci_config)
        self.assertIn("security:", ci_config)

        # Check for key testing steps
        self.assertIn("health/", ci_config)  # Health check test
        self.assertIn("python manage.py test", ci_config)  # Django tests
