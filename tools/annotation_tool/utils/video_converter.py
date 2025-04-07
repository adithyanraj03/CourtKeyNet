import os
import cv2
import threading
import time
import queue

class VideoToFramesConverter:
    """
    Utility class for converting video files to image frames
    for annotation purposes. Supports multi-threading, progress
    tracking, and can be stopped mid-conversion.
    """
    def __init__(self, status_callback=None, progress_callback=None):
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.stop_requested = False
        self.conversion_queue = queue.Queue()
        self.is_converting = False
    
    def convert_video(self, video_path, output_dir, frame_rate=None, max_frames=None, 
                       resize=None, prefix="frame"):
        """
        Start video conversion in a separate thread.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_rate: Frames per second to extract (None = extract all frames)
            max_frames: Maximum number of frames to extract (None = no limit)
            resize: Tuple (width, height) to resize frames (None = original size)
            prefix: Prefix for the output frame filenames
        """
        # Check if already converting
        if self.is_converting:
            return False
        
        # Reset stop flag
        self.stop_requested = False
        
        # Create conversion thread
        self.conversion_thread = threading.Thread(
            target=self._run_conversion,
            args=(video_path, output_dir, frame_rate, max_frames, resize, prefix)
        )
        
        # Start conversion
        self.is_converting = True
        self.conversion_thread.start()
        
        return True
    
    def stop_conversion(self):
        """Request to stop the current conversion process"""
        if self.is_converting:
            self.stop_requested = True
            return True
        return False
    
    def is_conversion_active(self):
        """Check if conversion thread is still active"""
        return self.is_converting
    
    def _run_conversion(self, video_path, output_dir, frame_rate, max_frames, 
                         resize, prefix):
        """
        Run conversion process in a separate thread.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_rate: Frames per second to extract (None = extract all frames)
            max_frames: Maximum number of frames to extract (None = no limit)
            resize: Tuple (width, height) to resize frames (None = original size)
            prefix: Prefix for the output frame filenames
        """
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                if self.status_callback:
                    self.status_callback(f"Error: Could not open video file: {video_path}")
                self.is_converting = False
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame extraction parameters
            if frame_rate is None:
                frame_interval = 1
            else:
                frame_interval = int(original_fps / frame_rate)
                frame_interval = max(1, frame_interval)  # At least 1
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Process frames
            frame_count = 0
            saved_count = 0
            
            if self.status_callback:
                self.status_callback(f"Converting video: {os.path.basename(video_path)}")
            
            while True:
                # Check if stop requested
                if self.stop_requested:
                    if self.status_callback:
                        self.status_callback("Conversion stopped by user")
                    break
                
                # Read frame
                ret, frame = cap.read()
                
                # Break if end of video
                if not ret:
                    break
                
                # Process frames at specified interval
                if frame_count % frame_interval == 0:
                    # Resize frame if requested
                    if resize is not None:
                        frame = cv2.resize(frame, resize)
                    
                    # Save frame
                    frame_path = os.path.join(
                        output_dir,
                        f"{prefix}_{saved_count:06d}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    
                    # Update counter
                    saved_count += 1
                    
                    # Update progress
                    if self.progress_callback and total_frames > 0:
                        progress = min(100, int((frame_count / total_frames) * 100))
                        self.progress_callback(progress)
                    
                    # Check maximum frames limit
                    if max_frames is not None and saved_count >= max_frames:
                        if self.status_callback:
                            self.status_callback(f"Reached maximum frames limit ({max_frames})")
                        break
                
                # Increment frame counter
                frame_count += 1
                
                # Yield to other threads briefly
                time.sleep(0.001)
            
            # Release video capture
            cap.release()
            
            # Final update
            if self.progress_callback:
                self.progress_callback(100)
            
            if self.status_callback:
                self.status_callback(f"Conversion completed: {saved_count} frames extracted")
        
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error during conversion: {str(e)}")
        
        finally:
            # Reset conversion state
            self.is_converting = False