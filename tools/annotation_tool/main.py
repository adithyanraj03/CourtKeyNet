import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from .utils.undo_redo import UndoRedoManager
from .utils.video_converter import VideoToFramesConverter
from .gui.main_window import create_main_window
from .gui.file_browser import create_file_browser
from .gui.annotation_canvas import create_annotation_canvas
from .gui.control_panel import create_control_panel

def main():
    """Main entry point for the annotation tool"""
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()

class AnnotationTool:
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
        create_main_window(self)
        create_file_browser(self)
        create_annotation_canvas(self)
        create_control_panel(self)
        
        # Set up keyboard shortcuts
        self.setup_shortcuts()
        
        # Initial status update
        self.update_status("Ready")
    
    def setup_shortcuts(self):
        """Set up keyboard shortcuts"""
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
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def update_progress_bar(self, value):
        """Update the progress bar value (0-100)"""
        self.progress_bar['value'] = value
        self.root.update_idletasks()

# Implementation details for other methods would go here