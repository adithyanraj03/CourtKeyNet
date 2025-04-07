import os
import sys
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import threading
import queue

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from courtkeynet.utils.annotation import CourtAnnotationDecoder


class UndoRedoManager:
    """Manages undo/redo operations for annotations"""
    
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.undo_stack = []
        self.redo_stack = []
    
    def add_state(self, state):
        """Add a new state to the history"""
        # Make a deep copy of the state
        state_copy = state.copy()
        
        # Add to undo stack
        self.undo_stack.append(state_copy)
        
        # Clear redo stack after a new action
        self.redo_stack.clear()
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
    
    def undo(self):
        """Undo the last action"""
        if not self.undo_stack:
            return None
        
        # Get current state (before undo)
        current_state = self.undo_stack.pop()
        
        # Add to redo stack
        self.redo_stack.append(current_state)
        
        # Return the previous state (or None if stack is empty)
        return self.undo_stack[-1].copy() if self.undo_stack else None
    
    def redo(self):
        """Redo the last undone action"""
        if not self.redo_stack:
            return None
        
        # Get next state
        next_state = self.redo_stack.pop()
        
        # Add to undo stack
        self.undo_stack.append(next_state)
        
        return next_state.copy()
    
    def can_undo(self):
        """Check if undo is possible"""
        return len(self.undo_stack) > 0
    
    def can_redo(self):
        """Check if redo is possible"""
        return len(self.redo_stack) > 0


class VideoToFramesConverter:
    """Converts video files to individual frames"""
    
    def __init__(self, status_callback=None, progress_callback=None):
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.stop_flag = False
    
    def update_status(self, message):
        """Update status message"""
        if self.status_callback:
            self.status_callback(message)
    
    def update_progress(self, current, total):
        """Update progress bar"""
        if self.progress_callback:
            progress = int(100 * current / total) if total > 0 else 0
            self.progress_callback(progress)
    
    def stop_conversion(self):
        """Stop the conversion process"""
        self.stop_flag = True
    
    def convert(self, video_path, output_dir, frame_rate=None, max_frames=None):
        """
        Convert video to frames
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_rate: If specified, extract frames at this rate (e.g., 1 frame per second)
            max_frames: Maximum number of frames to extract
        """
        self.stop_flag = False
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.update_status(f"Error: Could not open video file: {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        self.update_status(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        # Calculate frame extraction interval
        if frame_rate is not None:
            # Extract at specific rate (e.g., 1 frame per second)
            interval = int(fps / frame_rate)
        else:
            # Extract all frames
            interval = 1
        
        # Limit max frames if specified
        extract_total = min(total_frames // interval, max_frames) if max_frames else total_frames // interval
        
        self.update_status(f"Extracting approximately {extract_total} frames...")
        
        # Extract frames
        count = 0
        saved_count = 0
        
        while count < total_frames:
            if self.stop_flag:
                self.update_status("Conversion stopped by user")
                break
                
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            # Update progress every 10 frames
            if saved_count % 10 == 0:
                self.update_progress(count, total_frames)
                self.update_status(f"Extracted {saved_count} frames...")
            
            # Move to next frame
            count += interval
        
        # Release resources
        cap.release()
        
        self.update_progress(100, 100)
        self.update_status(f"Finished extracting {saved_count} frames")
        
        return True


class CourtAnnotationTool:
    """
    A professional GUI tool for annotating badminton courts with keypoints
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("CourtKeyNet Annotation Tool")
        self.root.geometry("1280x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize variables
        self.current_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.annotations = {}  # {image_path: [pt1, pt2, pt3, pt4]}
        self.zoom_level = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        
        # Current annotation state
        self.current_points = []
        self.dragging_point_index = -1
        self.court_class = 0
        
        # Undo/Redo manager
        self.history_manager = UndoRedoManager()
        
        # Video conversion
        self.converter = VideoToFramesConverter(
            status_callback=self.update_status,
            progress_callback=self.update_progress_bar
        )
        self.conversion_thread = None
        
        # Create the GUI
        self.create_gui()
        
        # Keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<Control-o>", lambda e: self.open_directory())
        self.root.bind("<Control-v>", lambda e: self.open_video())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Delete>", lambda e: self.clear_annotation())
        self.root.bind("1", lambda e: self.set_court_class(0))
        self.root.bind("2", lambda e: self.set_court_class(1))
        self.root.bind("3", lambda e: self.set_court_class(2))
        self.root.bind("<Escape>", lambda e: self.cancel_annotation())
        
        # Initial status update
        self.update_status("Ready")
    
    def create_gui(self):
        """Create the GUI components"""
        # Main layout: left panel, main canvas, right panel
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel (file browser)
        self.left_frame = ttk.Frame(self.paned_window, width=200)
        self.paned_window.add(self.left_frame, weight=1)
        
        # File browser
        self.create_file_browser()
        
        # Main frame (canvas and controls)
        self.main_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.main_frame, weight=4)
        
        # Canvas for image display and annotation
        self.create_canvas()
        
        # Controls under canvas
        self.create_controls()
        
        # Right panel (annotation details)
        self.right_frame = ttk.Frame(self.paned_window, width=250)
        self.paned_window.add(self.right_frame, weight=1)
        
        # Annotation details
        self.create_annotation_panel()
        
        # Status bar at the bottom
        self.create_status_bar()

    # [Rest of the implementation continues with methods for the GUI functionality]


if __name__ == "__main__":
    root = tk.Tk()
    app = CourtAnnotationTool(root)
    root.mainloop()