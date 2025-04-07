import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

def create_annotation_canvas(app):
    """
    Create the annotation canvas for the annotation tool.
    
    Args:
        app: AnnotationTool instance
    """
    # Create canvas frame
    canvas_frame = ttk.Frame(app.right_frame)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create scrollbars
    scrollbar_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    
    scrollbar_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Create canvas
    app.canvas = tk.Canvas(
        canvas_frame,
        yscrollcommand=scrollbar_y.set,
        xscrollcommand=scrollbar_x.set,
        background="#333333",
        cursor="crosshair"
    )
    app.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configure scrollbars
    scrollbar_y.config(command=app.canvas.yview)
    scrollbar_x.config(command=app.canvas.xview)
    
    # Bind mouse events
    app.canvas.bind("<Button-1>", app.on_canvas_click)
    app.canvas.bind("<B1-Motion>", app.on_canvas_drag)
    app.canvas.bind("<ButtonRelease-1>", app.on_canvas_release)
    
    # Bind zooming events
    app.canvas.bind("<MouseWheel>", app.on_canvas_zoom)  # Windows
    app.canvas.bind("<Button-4>", app.on_canvas_zoom)  # Linux scroll up
    app.canvas.bind("<Button-5>", app.on_canvas_zoom)  # Linux scroll down
    
    # Bind panning events (middle mouse button)
    app.canvas.bind("<Button-2>", app.on_pan_start)
    app.canvas.bind("<B2-Motion>", app.on_pan_move)
    app.canvas.bind("<ButtonRelease-2>", app.on_pan_end)
    
    # Set up image display
    app.canvas_image_id = None
    app.canvas_item_ids = []
    
    # Create zoom control
    zoom_frame = ttk.Frame(app.right_frame)
    zoom_frame.pack(fill=tk.X, pady=5)
    
    zoom_label = ttk.Label(zoom_frame, text="Zoom:")
    zoom_label.pack(side=tk.LEFT, padx=5)
    
    app.zoom_scale = ttk.Scale(
        zoom_frame,
        from_=0.1,
        to=5.0,
        orient=tk.HORIZONTAL,
        value=1.0,
        command=app.on_zoom_change
    )
    app.zoom_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    app.zoom_value_label = ttk.Label(zoom_frame, text="100%")
    app.zoom_value_label.pack(side=tk.LEFT, padx=5)
    
    zoom_fit_btn = ttk.Button(zoom_frame, text="Fit to Window", command=app.fit_to_window)
    zoom_fit_btn.pack(side=tk.RIGHT, padx=5)