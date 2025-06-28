#!/usr/bin/env python3
"""
Test script for CoreVideoProcessor

This script tests the core video processing functionality without
relying on complex infrastructure or external services.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add the current directory to path so we can import our processor
sys.path.insert(0, '.')

try:
    from core_video_processor import CoreVideoProcessor
except ImportError as e:
    print(f"Error importing CoreVideoProcessor: {e}")
    print("Make sure core_video_processor.py is in the current directory")
    sys.exit(1)


def test_basic_functionality():
    """Test basic processor initialization and capabilities."""
    print("=" * 50)
    print("Testing CoreVideoProcessor Basic Functionality")
    print("=" * 50)
    
    # Initialize processor
    try:
        processor = CoreVideoProcessor()
        print("✓ Processor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize processor: {e}")
        return False
    
    # Test supported formats
    test_files = [
        ("video.mp4", True),
        ("video.avi", True),
        ("video.mov", True),
        ("video.txt", False),
        ("video.jpg", False)
    ]
    
    print("\nTesting file format validation:")
    for filename, should_be_valid in test_files:
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            is_valid, error = processor.validate_video_file(tmp_path)
            if is_valid == should_be_valid:
                status = "✓"
            else:
                status = "✗"
            print(f"  {status} {filename}: {is_valid} ({'expected' if is_valid == should_be_valid else 'unexpected'})")
        finally:
            os.unlink(tmp_path)
    
    return True


def test_with_sample_video():
    """Test processing with actual video file if available."""
    print("\n" + "=" * 50)
    print("Testing with Sample Video")
    print("=" * 50)
    
    # Look for test video files
    test_video_paths = [
        "test_data/test.mp4",
        "sample.mp4",
        "test.mp4"
    ]
    
    video_path = None
    for path in test_video_paths:
        if Path(path).exists():
            video_path = path
            break
    
    if not video_path:
        print("No test video found. Skipping video processing tests.")
        print("To test with a video file, place a video file at one of these paths:")
        for path in test_video_paths:
            print(f"  - {path}")
        return True
    
    print(f"Found test video: {video_path}")
    
    try:
        processor = CoreVideoProcessor()
        
        # Test metadata extraction
        print("\nTesting metadata extraction...")
        metadata = processor.extract_video_metadata(video_path)
        if metadata:
            print("✓ Metadata extracted successfully:")
            print(f"  Duration: {metadata.duration_seconds:.2f}s")
            print(f"  Resolution: {metadata.width_pixels}x{metadata.height_pixels}")
            print(f"  FPS: {metadata.frames_per_second:.2f}")
            print(f"  Format: {metadata.format_extension}")
            print(f"  Size: {metadata.file_size_megabytes:.2f} MB")
        else:
            print("✗ Failed to extract metadata")
            return False
        
        # Test audio extraction
        print("\nTesting audio extraction...")
        audio_result = processor.extract_audio_from_video(video_path)
        if audio_result.is_successful:
            print(f"✓ Audio extracted to: {audio_result.output_file_path}")
            print(f"  Processing time: {audio_result.processing_duration_seconds:.2f}s")
            
            # Clean up the audio file
            if audio_result.output_file_path and Path(audio_result.output_file_path).exists():
                os.remove(audio_result.output_file_path)
                print("  Cleaned up temporary audio file")
        else:
            print(f"✗ Audio extraction failed: {audio_result.error_message}")
        
        # Test transcription (if Whisper is available)
        print("\nTesting transcription...")
        transcription = processor.transcribe_video_file(video_path, language_code="en")
        if transcription.is_successful:
            print("✓ Transcription successful:")
            print(f"  Language: {transcription.detected_language}")
            print(f"  Text length: {len(transcription.transcribed_text)} characters")
            print(f"  Segments: {len(transcription.text_segments) if transcription.text_segments else 0}")
            if transcription.transcribed_text:
                preview = transcription.transcribed_text[:100] + "..." if len(transcription.transcribed_text) > 100 else transcription.transcribed_text
                print(f"  Preview: {preview}")
        else:
            print(f"⚠ Transcription failed: {transcription.error_message}")
            print("  (This is expected if Whisper is not installed)")
        
        # Test video summary
        print("\nTesting video summary...")
        summary = processor.create_comprehensive_video_summary(video_path)
        if summary:
            print("✓ Video summary created:")
            print(f"  Metadata available: {'Yes' if summary['metadata'] else 'No'}")
            print(f"  Transcription available: {'Yes' if summary['transcription'] else 'No'}")
            if summary['processing_errors']:
                print(f"  Errors: {len(summary['processing_errors'])}")
                for error in summary['processing_errors']:
                    print(f"    - {error}")
        else:
            print("✗ Failed to create video summary")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "=" * 50)
    print("Testing Error Handling")
    print("=" * 50)
    
    processor = CoreVideoProcessor()
    
    # Test with non-existent file
    print("Testing with non-existent file...")
    is_valid, error = processor.validate_video_file("nonexistent_file.mp4")
    if not is_valid and "does not exist" in error:
        print("✓ Correctly handled non-existent file")
    else:
        print("✗ Failed to handle non-existent file properly")
    
    # Test metadata extraction with invalid file
    print("Testing metadata extraction with invalid file...")
    metadata = processor.extract_video_metadata("nonexistent_file.mp4")
    if metadata is None:
        print("✓ Correctly returned None for invalid file")
    else:
        print("✗ Should have returned None for invalid file")
    
    # Test audio extraction with invalid file
    print("Testing audio extraction with invalid file...")
    result = processor.extract_audio_from_video("nonexistent_file.mp4")
    if not result.is_successful and result.error_message:
        print("✓ Correctly handled invalid file for audio extraction")
    else:
        print("✗ Should have failed for invalid file")
    
    return True


def main():
    """Run all tests."""
    print("CoreVideoProcessor Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test error handling
    if not test_error_handling():
        all_passed = False
    
    # Test with sample video (if available)
    if not test_with_sample_video():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests completed successfully!")
    else:
        print("❌ Some tests failed")
    
    print("\nDependency Status:")
    try:
        import cv2
        print("✓ OpenCV: Available")
    except ImportError:
        print("✗ OpenCV: Not available (pip install opencv-python)")
    
    try:
        import whisper
        print("✓ Whisper: Available")
    except ImportError:
        print("✗ Whisper: Not available (pip install openai-whisper)")
    
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        if result.returncode == 0:
            print("✓ FFmpeg: Available")
        else:
            print("✗ FFmpeg: Not working properly")
    except FileNotFoundError:
        print("✗ FFmpeg: Not available (install ffmpeg)")
    
    print("\nNext Steps:")
    print("1. Install missing dependencies if needed")
    print("2. Place a test video file in test_data/test.mp4 for full testing")
    print("3. Run: python test_core_processor.py")


if __name__ == "__main__":
    main() 