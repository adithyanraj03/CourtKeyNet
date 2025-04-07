import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import json
import shutil
from datetime import datetime
import threading
import queue


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
        
        # Set application icon
        try:
            # You would need an icon file in practice
            # self.root.iconbitmap("icon.ico")
            pass
        except:
            pass
        
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
        
        # File change detection
        self.last_modification_times = {}
        
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
        
        # Style configuration
        self.configure_styles()
    
    def configure_styles(self):
        """Configure styles for the GUI elements"""
        # Create a custom style
        style = ttk.Style()
        
        # Configure Treeview (file browser)
        style.configure("Treeview", font=('Arial', 10))
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        
        # Configure Buttons
        style.configure("TButton", font=('Arial', 10))
        
        # Configure Labels
        style.configure("TLabel", font=('Arial', 10))
        
        # Configure Frames
        style.configure("TFrame", background="#f0f0f0")
        
        # Configure Notebook (tabs)
        style.configure("TNotebook", background="#f0f0f0")
        style.configure("TNotebook.Tab", font=('Arial', 10))
    
    def create_file_browser(self):
        """Create the file browser panel"""
        # Directory controls
        dir_frame = ttk.Frame(self.left_frame)
        dir_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(dir_frame, text="Open Directory", command=self.open_directory).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(dir_frame, text="Refresh", command=self.refresh_directory).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Video controls
        video_frame = ttk.Frame(self.left_frame)
        video_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(video_frame, text="Open Video", command=self.open_video).pack(fill=tk.X)
        
        # File browser
        browser_frame = ttk.Frame(self.left_frame)
        browser_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File list with scrollbar
        self.file_list = ttk.Treeview(browser_frame, columns=("status"), show="tree headings")
        self.file_list.heading("#0", text="Files")
        self.file_list.heading("status", text="Status")
        self.file_list.column("#0", width=150)
        self.file_list.column("status", width=50, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(browser_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.file_list.bind("<<TreeviewSelect>>", self.on_file_select)
    
    def create_canvas(self):
        """Create the canvas for image display and annotation"""
        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg="gray90", cursor="crosshair")
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky='nsew')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self.on_mouse_wheel(e, delta=1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.on_mouse_wheel(e, delta=-1))  # Linux
        self.canvas.bind("<Control-Button-1>", self.start_pan)
        self.canvas.bind("<Control-B1-Motion>", self.pan)
        self.canvas.bind("<Control-ButtonRelease-1>", self.end_pan)
    
    def create_controls(self):
        """Create controls under the canvas"""
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Label(nav_frame, text="Image:").pack(side=tk.LEFT, padx=5)
        self.image_counter_label = ttk.Label(nav_frame, text="0/0")
        self.image_counter_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        ttk.Button(nav_frame, text="Zoom In", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Zoom Out", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        ttk.Button(nav_frame, text="Skip Unannotated", command=self.skip_to_unannotated).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Skip Annotated", command=self.skip_to_annotated).pack(side=tk.LEFT, padx=5)
        
        # Bottom controls (save, undo, etc.)
        bottom_frame = ttk.Frame(controls_frame)
        bottom_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(bottom_frame, text="Clear", command=self.clear_annotation).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Redo", command=self.redo).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(bottom_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        ttk.Button(bottom_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Export All", command=self.export_annotations).pack(side=tk.LEFT, padx=5)
    
    def create_annotation_panel(self):
        """Create the annotation details panel"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Settings tab
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="Settings")
        
        # Class selection
        class_frame = ttk.LabelFrame(settings_frame, text="Court Class")
        class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.class_var = tk.IntVar(value=0)
        ttk.Radiobutton(class_frame, text="Standard Court (0)", variable=self.class_var, value=0, command=lambda: self.set_court_class(0)).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(class_frame, text="Doubles Court (1)", variable=self.class_var, value=1, command=lambda: self.set_court_class(1)).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(class_frame, text="Alternative (2)", variable=self.class_var, value=2, command=lambda: self.set_court_class(2)).pack(anchor=tk.W, padx=5, pady=2)
        
        # Display settings
        display_frame = ttk.LabelFrame(settings_frame, text="Display Settings")
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.show_annotated_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Annotated", variable=self.show_annotated_var, command=self.refresh_file_list).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_unannotated_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Unannotated", variable=self.show_unannotated_var, command=self.refresh_file_list).pack(anchor=tk.W, padx=5, pady=2)
        
        # Point display settings
        point_frame = ttk.LabelFrame(settings_frame, text="Point Settings")
        point_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.point_size_var = tk.IntVar(value=5)
        ttk.Label(point_frame, text="Point Size:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(point_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                 variable=self.point_size_var, command=lambda e: self.display_image()).pack(fill=tk.X, padx=5, pady=2)
        
        self.line_width_var = tk.IntVar(value=2)
        ttk.Label(point_frame, text="Line Width:").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Scale(point_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                 variable=self.line_width_var, command=lambda e: self.display_image()).pack(fill=tk.X, padx=5, pady=2)
        
        # Annotation stats
        stats_frame = ttk.LabelFrame(settings_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.total_files_label = ttk.Label(stats_frame, text="Total Files: 0")
        self.total_files_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.annotated_files_label = ttk.Label(stats_frame, text="Annotated: 0")
        self.annotated_files_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Help tab
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="Help")
        
        # Scrollable text widget for help
        help_text = tk.Text(help_frame, wrap=tk.WORD, width=30, height=20)
        help_scroll = ttk.Scrollbar(help_frame, orient=tk.VERTICAL, command=help_text.yview)
        help_text.configure(yscrollcommand=help_scroll.set)
        
        help_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        help_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Add help content
        help_content = """
CourtKeyNet Annotation Tool

Keyboard Shortcuts:
- Ctrl+O: Open directory
- Ctrl+V: Open video
- Ctrl+S: Save annotations
- Ctrl+Z: Undo
- Ctrl+Y: Redo
- Right Arrow: Next image
- Left Arrow: Previous image
- Delete: Clear annotation
- 1/2/3: Set court class
- Esc: Cancel annotation

Mouse Controls:
- Left Click: Add/select point
- Drag: Move selected point
- Right Click: Context menu
- Ctrl+Drag: Pan image
- Mouse Wheel: Zoom in/out

Annotation Tips:
- Add 4 points for the court corners
- Points should be added in clockwise order
- Zoom in for precise placement
- Save regularly

For more help, visit:
https://github.com/yourusername/courtkeynet
        """
        
        help_text.insert(tk.END, help_content)
        help_text.configure(state=tk.DISABLED)  # Make read-only
        
        # Current annotation tab
        annotation_frame = ttk.Frame(self.notebook)
        self.notebook.add(annotation_frame, text="Current Annotation")
        
        # Current annotation details
        self.annotation_tree = ttk.Treeview(annotation_frame, columns=("x", "y"), show="headings")
        self.annotation_tree.heading("x", text="X")
        self.annotation_tree.heading("y", text="Y")
        self.annotation_tree.column("x", width=60)
        self.annotation_tree.column("y", width=60)
        
        anno_scrollbar = ttk.Scrollbar(annotation_frame, orient=tk.VERTICAL, command=self.annotation_tree.yview)
        self.annotation_tree.configure(yscrollcommand=anno_scrollbar.set)
        
        self.annotation_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        anno_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Bind selection event for annotation tree
        self.annotation_tree.bind("<<TreeviewSelect>>", self.on_annotation_select)
    
    def create_status_bar(self):
        """Create the status bar at the bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status message
        self.status_label = ttk.Label(status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=3)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, mode='determinate', length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=3)
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_progress_bar(self, value):
        """Update the progress bar value (0-100)"""
        self.progress_bar['value'] = value
        self.root.update_idletasks()
    
    def open_directory(self):
        """Open a directory containing images"""
        directory = filedialog.askdirectory(title="Select Directory with Images")
        if not directory:
            return
        
        self.current_dir = directory
        self.update_status(f"Loading images from {directory}...")
        self.load_images_from_directory(directory)
        self.load_annotations()
        self.update_statistics()
    
    def load_images_from_directory(self, directory):
        """Load all images from the directory"""
        # Clear existing files
        self.image_files = []
        
        # Get all image files
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    self.image_files.append((rel_path, full_path))
        
        # Sort files
        self.image_files.sort()
        
        # Update file list display
        self.refresh_file_list()
        
        # Reset current image
        self.current_image_index = -1
        if self.image_files:
            self.load_image(0)
        
        self.update_status(f"Loaded {len(self.image_files)} images")
    
    def refresh_file_list(self):
        """Refresh the file list display"""
        # Clear the tree
        for item in self.file_list.get_children():
            self.file_list.delete(item)
        
        # Add files to the tree
        for rel_path, full_path in self.image_files:
            # Check if it should be displayed based on annotation status
            has_annotation = full_path in self.annotations
            
            if (has_annotation and not self.show_annotated_var.get()) or \
               (not has_annotation and not self.show_unannotated_var.get()):
                continue
            
            # Add parent directories if needed
            parts = rel_path.split(os.sep)
            if len(parts) > 1:
                parent = ""
                for i, part in enumerate(parts[:-1]):
                    current_level = os.sep.join(parts[:i+1])
                    parent_level = os.sep.join(parts[:i]) if i > 0 else ""
                    
                    # Check if this level already exists
                    if not self.file_list.exists(current_level):
                        self.file_list.insert(parent_level, 'end', current_level, text=part, values=(""))
                    
                    parent = current_level
            else:
                parent = ""
            
            # Add the file with its status
            status = "✓" if full_path in self.annotations else ""
            self.file_list.insert(os.path.dirname(rel_path), 'end', rel_path, text=os.path.basename(rel_path), values=(status,))
    
    def refresh_directory(self):
        """Refresh the current directory"""
        if not self.current_dir:
            self.update_status("No directory loaded")
            return
        
        self.load_images_from_directory(self.current_dir)
        self.update_status("Directory refreshed")
    
    def on_file_select(self, event):
        """Handle file selection in the file browser"""
        selected_items = self.file_list.selection()
        if not selected_items:
            return
        
        # Find the selected file in our list
        selected_path = selected_items[0]
        for i, (rel_path, full_path) in enumerate(self.image_files):
            if rel_path == selected_path:
                self.load_image(i)
                break
    
    def load_image(self, index):
        """Load an image at the specified index"""
        if not self.image_files or index < 0 or index >= len(self.image_files):
            return
        
        self.current_image_index = index
        _, full_path = self.image_files[index]
        self.current_image_path = full_path
        
        try:
            # Open image
            image = Image.open(full_path)
            self.current_image = image
            
            # Reset view
            self.zoom_level = 1.0
            self.pan_start_x = 0
            self.pan_start_y = 0
            
            # Display image with annotations
            self.display_image()
            
            # Update image counter
            self.update_image_counter()
            
            # Load existing annotation if available
            self.current_points = self.annotations.get(full_path, []).copy()
            
            # Update annotation details
            self.update_annotation_details()
            
            self.update_status(f"Loaded image: {os.path.basename(full_path)}")
            
        except Exception as e:
            self.update_status(f"Error loading image: {e}")
    
    def update_image_counter(self):
        """Update the image counter label"""
        if self.image_files:
            self.image_counter_label.config(text=f"{self.current_image_index + 1}/{len(self.image_files)}")
        else:
            self.image_counter_label.config(text="0/0")
    
    def display_image(self):
        """Display the current image with annotations"""
        if self.current_image is None:
            return
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas size is tiny, wait for it to be properly sized
        if canvas_width < 50 or canvas_height < 50:
            self.root.after(100, self.display_image)
            return
        
        # Resize image to fit canvas with zoom
        img_width, img_height = self.current_image.size
        
        # Apply zoom
        display_width = int(img_width * self.zoom_level)
        display_height = int(img_height * self.zoom_level)
        
        # Resize image
        display_img = self.current_image.resize((display_width, display_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(display_img)
        
        # Add image to canvas
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo_image, tags="image")
        
        # Configure canvas scrollregion
        self.canvas.config(scrollregion=(
            canvas_width//2 - display_width//2, 
            canvas_height//2 - display_height//2,
            canvas_width//2 + display_width//2, 
            canvas_height//2 + display_height//2
        ))
        
        # Draw annotations if available
        self.draw_annotations()
    
    def draw_annotations(self):
        """Draw the current annotation points and lines"""
        if not self.current_points:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # Get point size and line width
        point_size = self.point_size_var.get()
        line_width = self.line_width_var.get()
        
        # Draw points and lines
        for i, point in enumerate(self.current_points):
            x_norm, y_norm = point
            
            # Convert normalized coordinates to canvas coordinates
            x = canvas_width//2 - (img_width * self.zoom_level)//2 + int(x_norm * img_width * self.zoom_level)
            y = canvas_height//2 - (img_height * self.zoom_level)//2 + int(y_norm * img_height * self.zoom_level)
            
            # Draw point
            self.canvas.create_oval(
                x - point_size, y - point_size, 
                x + point_size, y + point_size, 
                fill="red", outline="white", width=1, 
                tags=f"point_{i}"
            )
            
            # Add label
            self.canvas.create_text(
                x, y - point_size - 5, 
                text=str(i+1), 
                fill="white", font=("Arial", 8, "bold"),
                tags=f"label_{i}"
            )
            
            # Draw lines between points
            if len(self.current_points) > 1:
                next_point = self.current_points[(i + 1) % len(self.current_points)]
                next_x_norm, next_y_norm = next_point
                
                next_x = canvas_width//2 - (img_width * self.zoom_level)//2 + int(next_x_norm * img_width * self.zoom_level)
                next_y = canvas_height//2 - (img_height * self.zoom_level)//2 + int(next_y_norm * img_height * self.zoom_level)
                
                self.canvas.create_line(
                    x, y, next_x, next_y, 
                    fill="yellow", width=line_width, 
                    tags=f"line_{i}"
                )
    
    def on_canvas_click(self, event):
        """Handle canvas click events"""
        if self.current_image is None:
            return
        
        # Check if clicking on an existing point
        clicked_on_point = False
        closest_point = self.find_closest_point(event.x, event.y)
        
        if closest_point >= 0:
            # Start dragging the point
            self.dragging_point_index = closest_point
            clicked_on_point = True
        
        # If not clicking on a point and less than 4 points, add a new one
        if not clicked_on_point and len(self.current_points) < 4:
            # Convert canvas coordinates to normalized image coordinates
            norm_x, norm_y = self.canvas_to_image_coords(event.x, event.y)
            
            # Add point
            self.current_points.append((norm_x, norm_y))
            
            # Save current state to history
            self.history_manager.add_state(self.current_points)
            
            # Redraw
            self.display_image()
            self.update_annotation_details()
        
        # If we have 4 points, finalize the annotation
        if len(self.current_points) == 4 and not clicked_on_point:
            self.finalize_annotation()
    
    def on_canvas_drag(self, event):
        """Handle dragging points"""
        if self.dragging_point_index >= 0 and self.current_image is not None:
            # Convert canvas coordinates to normalized image coordinates
            norm_x, norm_y = self.canvas_to_image_coords(event.x, event.y)
            
            # Update point
            self.current_points[self.dragging_point_index] = (norm_x, norm_y)
            
            # Redraw
            self.display_image()
            self.update_annotation_details()
    
    def on_canvas_release(self, event):
        """Handle mouse release"""
        if self.dragging_point_index >= 0:
            # Save current state to history
            self.history_manager.add_state(self.current_points)
            
            # Reset dragging state
            self.dragging_point_index = -1
    
    def on_right_click(self, event):
        """Handle right-click for context menu"""
        if self.current_image is None:
            return
            
        # Create context menu
        context_menu = tk.Menu(self.root, tearoff=0)
        
        # Check if clicking near a point
        closest_point = self.find_closest_point(event.x, event.y)
        
        if closest_point >= 0:
            context_menu.add_command(label=f"Delete Point {closest_point+1}", 
                                    command=lambda: self.delete_point(closest_point))
        
        context_menu.add_command(label="Clear All Points", command=self.clear_annotation)
        context_menu.add_separator()
        context_menu.add_command(label="Save Annotation", command=self.save_current_annotation)
        
        # Display context menu
        context_menu.tk_popup(event.x_root, event.y_root)
    
    def delete_point(self, index):
        """Delete a point"""
        if 0 <= index < len(self.current_points):
            # Save current state to history
            self.history_manager.add_state(self.current_points.copy())
            
            # Remove point
            self.current_points.pop(index)
            
            # Redraw
            self.display_image()
            self.update_annotation_details()
    
    def find_closest_point(self, canvas_x, canvas_y):
        """Find the closest point to the given canvas coordinates"""
        if not self.current_points or self.current_image is None:
            return -1
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # Point size for hit testing
        hit_radius = max(10, self.point_size_var.get() * 2)
        
        # Check each point
        for i, point in enumerate(self.current_points):
            x_norm, y_norm = point
            
            # Convert normalized coordinates to canvas coordinates
            x = canvas_width//2 - (img_width * self.zoom_level)//2 + int(x_norm * img_width * self.zoom_level)
            y = canvas_height//2 - (img_height * self.zoom_level)//2 + int(y_norm * img_height * self.zoom_level)
            
            # Check distance
            dist = ((canvas_x - x) ** 2 + (canvas_y - y) ** 2) ** 0.5
            if dist <= hit_radius:
                return i
                
        return -1
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to normalized image coordinates"""
        if self.current_image is None:
            return 0, 0
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # Calculate normalized coordinates
        x_offset = canvas_width//2 - (img_width * self.zoom_level)//2
        y_offset = canvas_height//2 - (img_height * self.zoom_level)//2
        
        x_norm = (canvas_x - x_offset) / (img_width * self.zoom_level)
        y_norm = (canvas_y - y_offset) / (img_height * self.zoom_level)
        
        # Clamp to [0, 1]
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))
        
        return x_norm, y_norm
    
    def update_annotation_details(self):
        """Update the annotation details view"""
        # Clear existing items
        for item in self.annotation_tree.get_children():
            self.annotation_tree.delete(item)
            
        # Add current points
        for i, (x, y) in enumerate(self.current_points):
            self.annotation_tree.insert("", "end", text=str(i+1), values=(f"{x:.6f}", f"{y:.6f}"))
    
    def on_annotation_select(self, event):
        """Handle selection in the annotation tree"""
        selection = self.annotation_tree.selection()
        if not selection:
            return
            
        index = self.annotation_tree.index(selection[0])
        if 0 <= index < len(self.current_points):
            # Highlight the selected point on the canvas
            self.canvas.itemconfig(f"point_{index}", width=3, outline="blue")
    
    def finalize_annotation(self):
        """Finalize the current annotation"""
        if len(self.current_points) != 4:
            return
            
        # Save annotation
        self.save_current_annotation()
        
        # Update file status in the list
        self.update_file_status()
    
    def save_current_annotation(self):
        """Save the current annotation"""
        if not self.current_image_path or not self.current_points:
            return
            
        # Store annotation
        self.annotations[self.current_image_path] = self.current_points.copy()
        
        # Update status
        self.update_status(f"Annotation saved for {os.path.basename(self.current_image_path)}")
        
        # Update statistics
        self.update_statistics()
        
        # Update file status in the list
        self.update_file_status()
    
    def update_file_status(self):
        """Update the status indicator for the current file in the list"""
        if not self.current_image_path:
            return
            
        # Find the item in the tree
        for rel_path, full_path in self.image_files:
            if full_path == self.current_image_path:
                # Update status
                self.file_list.item(rel_path, values=("✓" if full_path in self.annotations else "",))
                break
    
    def clear_annotation(self):
        """Clear the current annotation"""
        if not self.current_points:
            return
            
        # Save current state to history
        self.history_manager.add_state(self.current_points.copy())
        
        # Clear points
        self.current_points = []
        
        # Redraw
        self.display_image()
        self.update_annotation_details()
        
        # If this image has a saved annotation, remove it
        if self.current_image_path in self.annotations:
            del self.annotations[self.current_image_path]
            self.update_file_status()
            self.update_statistics()
    
    def cancel_annotation(self):
        """Cancel the current annotation (clear without saving to history)"""
        self.current_points = []
        self.display_image()
        self.update_annotation_details()
    
    def undo(self):
        """Undo the last annotation action"""
        if not self.history_manager.can_undo():
            self.update_status("Nothing to undo")
            return
            
        # Get previous state
        previous_state = self.history_manager.undo()
        
        if previous_state is not None:
            # Restore state
            self.current_points = previous_state
        else:
            # If no previous state, clear points
            self.current_points = []
            
        # Redraw
        self.display_image()
        self.update_annotation_details()
        self.update_status("Undo successful")
    
    def redo(self):
        """Redo the last undone action"""
        if not self.history_manager.can_redo():
            self.update_status("Nothing to redo")
            return
            
        # Get next state
        next_state = self.history_manager.redo()
        
        if next_state is not None:
            # Restore state
            self.current_points = next_state
            
            # Redraw
            self.display_image()
            self.update_annotation_details()
            self.update_status("Redo successful")
    
    def next_image(self):
        """Go to the next image"""
        if not self.image_files:
            return
            
        next_index = (self.current_image_index + 1) % len(self.image_files)
        self.load_image(next_index)
    
    def prev_image(self):
        """Go to the previous image"""
        if not self.image_files:
            return
            
        prev_index = (self.current_image_index - 1) % len(self.image_files)
        self.load_image(prev_index)
    
    def skip_to_unannotated(self):
        """Skip to the next unannotated image"""
        if not self.image_files:
            return
            
        start_index = (self.current_image_index + 1) % len(self.image_files)
        
        # Search for an unannotated image
        for i in range(len(self.image_files)):
            index = (start_index + i) % len(self.image_files)
            _, full_path = self.image_files[index]
            
            if full_path not in self.annotations:
                self.load_image(index)
                return
                
        self.update_status("No unannotated images found")
    
    def skip_to_annotated(self):
        """Skip to the next annotated image"""
        if not self.image_files:
            return
            
        start_index = (self.current_image_index + 1) % len(self.image_files)
        
        # Search for an annotated image
        for i in range(len(self.image_files)):
            index = (start_index + i) % len(self.image_files)
            _, full_path = self.image_files[index]
            
            if full_path in self.annotations:
                self.load_image(index)
                return
                
        self.update_status("No annotated images found")
    
    def on_mouse_wheel(self, event, delta=None):
        """Handle mouse wheel for zooming"""
        if self.current_image is None:
            return
            
        # Get scroll direction
        if delta is not None:
            # Linux scroll event
            direction = delta
        else:
            # Windows scroll event
            direction = 1 if event.delta > 0 else -1
            
        # Zoom factor
        factor = 1.1 if direction > 0 else 0.9
        
        # Apply zoom
        self.zoom(factor, event.x, event.y)
    
    def zoom(self, factor, center_x=None, center_y=None):
        """Zoom the image"""
        if self.current_image is None:
            return
            
        # If no center specified, use canvas center
        if center_x is None or center_y is None:
            center_x = self.canvas.winfo_width() // 2
            center_y = self.canvas.winfo_height() // 2
            
        # Get current scroll position
        current_scroll_x = self.canvas.canvasx(center_x)
        current_scroll_y = self.canvas.canvasy(center_y)
        
        # Apply zoom
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        
        # Limit zoom range
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
        
        # Redraw
        self.display_image()
        
        # Adjust scroll position to keep the center point
        new_scroll_x = current_scroll_x * (self.zoom_level / old_zoom)
        new_scroll_y = current_scroll_y * (self.zoom_level / old_zoom)
        
        # Set new scroll position
        self.canvas.xview_moveto(new_scroll_x / self.canvas.winfo_width())
        self.canvas.yview_moveto(new_scroll_y / self.canvas.winfo_height())
    
    def reset_view(self):
        """Reset the view (zoom and pan)"""
        self.zoom_level = 1.0
        self.display_image()
    
    def start_pan(self, event):
        """Start panning the image"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        # Change cursor
        self.canvas.config(cursor="fleur")
    
    def pan(self, event):
        """Pan the image"""
        if not self.is_panning:
            return
            
        # Calculate movement
        dx = self.pan_start_x - event.x
        dy = self.pan_start_y - event.y
        
        # Move canvas
        self.canvas.xview_scroll(int(dx), "units")
        self.canvas.yview_scroll(int(dy), "units")
        
        # Update start position
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def end_pan(self, event):
        """End panning"""
        self.is_panning = False
        
        # Reset cursor
        self.canvas.config(cursor="crosshair")
    
    def set_court_class(self, class_id):
        """Set the court class"""
        self.court_class = class_id
        self.class_var.set(class_id)
        self.update_status(f"Court class set to {class_id}")
    
    def save_annotations(self):
        """Save all annotations to disk"""
        if not self.current_dir or not self.annotations:
            self.update_status("No annotations to save")
            return
            
        # Create annotations directory
        annotations_dir = os.path.join(self.current_dir, "labels")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Save each annotation
        count = 0
        for image_path, points in self.annotations.items():
            if len(points) != 4:
                continue
                
            # Get the relative path and base name
            rel_path = os.path.relpath(image_path, self.current_dir)
            base_name = os.path.splitext(os.path.basename(rel_path))[0]
            
            # Create annotation file path
            label_path = os.path.join(annotations_dir, f"{base_name}.txt")
            
            # Create annotation text
            # Format: class_id center_x center_y width height kpt1_x kpt1_y kpt1_vis kpt2_x kpt2_y kpt2_vis ...
            
            # Calculate bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Create annotation line
            parts = [str(self.court_class)]  # Class ID
            parts.extend([f"{val:.8f}" for val in [center_x, center_y, width, height]])  # Bounding box
            
            # Add keypoints
            for x, y in points:
                parts.extend([f"{x:.8f}", f"{y:.8f}", "2"])  # 2 = visible
                
            annotation_line = " ".join(parts)
            
            # Write to file
            with open(label_path, "w") as f:
                f.write(annotation_line)
                
            count += 1
            
        self.update_status(f"Saved {count} annotations to {annotations_dir}")
    
    def export_annotations(self):
        """Export annotations to a custom format"""
        if not self.current_dir or not self.annotations:
            self.update_status("No annotations to export")
            return
            
        # Ask for export location
        export_path = filedialog.asksaveasfilename(
            title="Export Annotations",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not export_path:
            return
            
        # Convert annotations to export format
        export_data = {}
        
        for image_path, points in self.annotations.items():
            # Get relative path
            rel_path = os.path.relpath(image_path, self.current_dir)
            
            # Store points
            export_data[rel_path] = {
                "points": points,
                "class": self.court_class
            }
            
        # Save to file
        try:
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)
                
            self.update_status(f"Exported {len(export_data)} annotations to {export_path}")
        except Exception as e:
            self.update_status(f"Error exporting annotations: {e}")
    
    def load_annotations(self):
        """Load annotations from disk"""
        if not self.current_dir:
            return
            
        # Look for labels directory
        labels_dir = os.path.join(self.current_dir, "labels")
        if not os.path.exists(labels_dir) or not os.path.isdir(labels_dir):
            self.update_status("No labels directory found")
            return
            
        # Clear existing annotations
        self.annotations = {}
        
        # Load annotations
        count = 0
        for root, _, files in os.walk(labels_dir):
            for file in files:
                if not file.endswith(".txt"):
                    continue
                    
                # Get base name
                base_name = os.path.splitext(file)[0]
                
                # Find corresponding image
                image_path = None
                for rel_path, full_path in self.image_files:
                    if os.path.splitext(os.path.basename(rel_path))[0] == base_name:
                        image_path = full_path
                        break
                        
                if not image_path:
                    continue
                    
                # Read annotation
                label_path = os.path.join(root, file)
                
                try:
                    with open(label_path, "r") as f:
                        line = f.read().strip()
                        
                    # Parse annotation line
                    parts = line.split()
                    if len(parts) < 5 + 4*3:  # class_id + bbox(4) + keypoints(4*3)
                        continue
                        
                    # Extract keypoints
                    keypoints = []
                    for i in range(5, len(parts), 3):
                        if i + 2 < len(parts):
                            x = float(parts[i])
                            y = float(parts[i+1])
                            keypoints.append((x, y))
                            
                    # Store annotation
                    if len(keypoints) == 4:
                        self.annotations[image_path] = keypoints
                        count += 1
                        
                except Exception as e:
                    print(f"Error loading annotation {label_path}: {e}")
                    
        self.update_status(f"Loaded {count} annotations")
        self.update_file_status()
        self.update_statistics()
    
    def update_statistics(self):
        """Update annotation statistics"""
        total_files = len(self.image_files)
        annotated_files = len(self.annotations)
        
        self.total_files_label.config(text=f"Total Files: {total_files}")
        self.annotated_files_label.config(text=f"Annotated: {annotated_files}")
    
    def open_video(self):
        """Open a video file for frame extraction"""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if not video_path:
            return
            
        # Ask for output directory
        output_dir = filedialog.askdirectory(
            title="Select Output Directory for Frames"
        )
        
        if not output_dir:
            return
            
        # Ask for extraction settings
        settings_dialog = VideoExtractionDialog(self.root)
        if not settings_dialog.result:
            return
            
        # Extract frames
        self.extract_frames(video_path, output_dir, settings_dialog.result)
    
    def extract_frames(self, video_path, output_dir, settings):
        """Extract frames from a video file"""
        # Parse settings
        frame_rate = settings.get("frame_rate", None)
        if frame_rate is not None:
            try:
                frame_rate = float(frame_rate)
            except ValueError:
                frame_rate = None
                
        max_frames = settings.get("max_frames", None)
        if max_frames is not None:
            try:
                max_frames = int(max_frames)
            except ValueError:
                max_frames = None
        
        # Start extraction in a separate thread
        if self.conversion_thread and self.conversion_thread.is_alive():
            self.converter.stop_conversion()
            self.conversion_thread.join()
            
        self.conversion_thread = threading.Thread(
            target=self._run_extraction,
            args=(video_path, output_dir, frame_rate, max_frames)
        )
        self.conversion_thread.daemon = True
        self.conversion_thread.start()
    
    def _run_extraction(self, video_path, output_dir, frame_rate, max_frames):
        """Run frame extraction in a thread"""
        try:
            # Extract frames
            self.converter.convert(video_path, output_dir, frame_rate, max_frames)
            
            # Load the directory when finished
            self.root.after(0, lambda: self._load_extracted_frames(output_dir))
        except Exception as e:
            self.update_status(f"Error during extraction: {e}")
    
    def _load_extracted_frames(self, directory):
        """Load extracted frames"""
        self.current_dir = directory
        self.load_images_from_directory(directory)
        self.update_status(f"Loaded extracted frames from {directory}")
    
    def on_close(self):
        """Handle window close event"""
        # Check for unsaved annotations
        if self.annotations:
            save = messagebox.askyesnocancel(
                "Save Annotations",
                "Do you want to save annotations before exiting?"
            )
            
            if save is None:  # Cancel
                return
                
            if save:  # Yes
                self.save_annotations()
                
        # Stop any running conversions
        if self.conversion_thread and self.conversion_thread.is_alive():
            self.converter.stop_conversion()
            
        # Close the window
        self.root.destroy()


class VideoExtractionDialog:
    """Dialog for video extraction settings"""
    
    def __init__(self, parent):
        self.result = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Frame Extraction Settings")
        self.dialog.geometry("400x250")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        dialog_width = 400
        dialog_height = 250
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Create widgets
        ttk.Label(self.dialog, text="Frame Extraction Settings", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Frame rate
        frame_rate_frame = ttk.Frame(self.dialog)
        frame_rate_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(frame_rate_frame, text="Frame Rate (FPS):").pack(side=tk.LEFT, padx=5)
        self.frame_rate_var = tk.StringVar(value="1")
        ttk.Entry(frame_rate_frame, textvariable=self.frame_rate_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_rate_frame, text="(blank for all frames)").pack(side=tk.LEFT, padx=5)
        
        # Max frames
        max_frames_frame = ttk.Frame(self.dialog)
        max_frames_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(max_frames_frame, text="Max Frames:").pack(side=tk.LEFT, padx=5)
        self.max_frames_var = tk.StringVar(value="100")
        ttk.Entry(max_frames_frame, textvariable=self.max_frames_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(max_frames_frame, text="(blank for all frames)").pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Extract", command=self.on_extract).pack(side=tk.RIGHT, padx=5)
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def on_extract(self):
        """Handle extract button click"""
        # Get values
        frame_rate = self.frame_rate_var.get().strip()
        max_frames = self.max_frames_var.get().strip()
        
        # Validate
        if frame_rate and not frame_rate.replace(".", "", 1).isdigit():
            messagebox.showerror("Invalid Input", "Frame rate must be a number")
            return
            
        if max_frames and not max_frames.isdigit():
            messagebox.showerror("Invalid Input", "Max frames must be a number")
            return
            
        # Store results
        self.result = {
            "frame_rate": frame_rate if frame_rate else None,
            "max_frames": max_frames if max_frames else None
        }
        
        # Close dialog
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle cancel button click"""
        self.dialog.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CourtAnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
