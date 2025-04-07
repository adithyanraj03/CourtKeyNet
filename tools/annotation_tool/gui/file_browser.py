import os
import tkinter as tk
from tkinter import ttk, filedialog

def create_file_browser(app):
    """
    Create the file browser panel for the annotation tool.
    
    Args:
        app: AnnotationTool instance
    """
    # Create file browser controls
    control_frame = ttk.Frame(app.left_frame)
    control_frame.pack(fill=tk.X, pady=5)
    
    # Open directory button
    btn_open_dir = ttk.Button(control_frame, text="Open Directory", command=app.open_directory)
    btn_open_dir.pack(side=tk.LEFT, padx=2)
    
    # Open video button
    btn_open_video = ttk.Button(control_frame, text="Open Video", command=app.open_video)
    btn_open_video.pack(side=tk.RIGHT, padx=2)
    
    # Create file list
    list_frame = ttk.Frame(app.left_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create scrollbars
    scrollbar_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    
    scrollbar_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Create listbox
    app.file_listbox = tk.Listbox(
        list_frame,
        yscrollcommand=scrollbar_y.set,
        xscrollcommand=scrollbar_x.set,
        selectmode=tk.SINGLE,
        background="white",
        foreground="black",
        font=("Arial", 10)
    )
    app.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Configure scrollbars
    scrollbar_y.config(command=app.file_listbox.yview)
    scrollbar_x.config(command=app.file_listbox.xview)
    
    # Bind selection event
    app.file_listbox.bind('<<ListboxSelect>>', app.on_file_select)
    
    # Create file navigation buttons
    nav_frame = ttk.Frame(app.left_frame)
    nav_frame.pack(fill=tk.X, pady=5)
    
    btn_prev = ttk.Button(nav_frame, text="Previous", command=app.prev_image)
    btn_prev.pack(side=tk.LEFT, padx=2)
    
    btn_next = ttk.Button(nav_frame, text="Next", command=app.next_image)
    btn_next.pack(side=tk.RIGHT, padx=2)
    
    # Create label to show annotation status
    app.status_counter = ttk.Label(app.left_frame, text="0/0 annotated")
    app.status_counter.pack(side=tk.BOTTOM, pady=5)