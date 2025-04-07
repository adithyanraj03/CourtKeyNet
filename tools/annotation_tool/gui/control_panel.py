import tkinter as tk
from tkinter import ttk, filedialog

def create_control_panel(app):
    """
    Create the control panel for the annotation tool.
    
    Args:
        app: AnnotationTool instance
    """
    # Create annotation controls
    control_frame = ttk.Frame(app.bottom_frame)
    control_frame.pack(fill=tk.X, pady=5)
    
    # Class selection
    class_frame = ttk.Frame(control_frame)
    class_frame.pack(side=tk.LEFT, padx=10)
    
    class_label = ttk.Label(class_frame, text="Court Class:")
    class_label.pack(side=tk.LEFT, padx=5)
    
    app.class_var = tk.IntVar(value=0)
    
    rb_class0 = ttk.Radiobutton(class_frame, text="Standard", variable=app.class_var, value=0)
    rb_class0.pack(side=tk.LEFT)
    
    rb_class1 = ttk.Radiobutton(class_frame, text="Doubles", variable=app.class_var, value=1)
    rb_class1.pack(side=tk.LEFT)
    
    rb_class2 = ttk.Radiobutton(class_frame, text="Alternative", variable=app.class_var, value=2)
    rb_class2.pack(side=tk.LEFT)
    
    # Actions
    action_frame = ttk.Frame(control_frame)
    action_frame.pack(side=tk.RIGHT, padx=10)
    
    btn_clear = ttk.Button(action_frame, text="Clear", command=app.clear_annotation)
    btn_clear.pack(side=tk.LEFT, padx=5)
    
    btn_undo = ttk.Button(action_frame, text="Undo", command=app.undo)
    btn_undo.pack(side=tk.LEFT, padx=5)
    
    btn_redo = ttk.Button(action_frame, text="Redo", command=app.redo)
    btn_redo.pack(side=tk.LEFT, padx=5)
    
    btn_save = ttk.Button(action_frame, text="Save", command=app.save_annotations)
    btn_save.pack(side=tk.LEFT, padx=5)
    
    # Video conversion controls
    video_frame = ttk.LabelFrame(app.bottom_frame, text="Video Conversion Settings")
    video_frame.pack(fill=tk.X, pady=5, padx=10)
    
    settings_frame = ttk.Frame(video_frame)
    settings_frame.pack(fill=tk.X, pady=5)
    
    # Frame rate
    frame_rate_label = ttk.Label(settings_frame, text="Frame Rate:")
    frame_rate_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
    
    app.frame_rate_var = tk.DoubleVar(value=1.0)
    frame_rate_entry = ttk.Entry(settings_frame, textvariable=app.frame_rate_var, width=8)
    frame_rate_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
    
    frame_rate_help = ttk.Label(settings_frame, text="fps (0 = extract all frames)")
    frame_rate_help.grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
    
    # Max frames
    max_frames_label = ttk.Label(settings_frame, text="Max Frames:")
    max_frames_label.grid(row=0, column=3, padx=5, pady=2, sticky=tk.W)
    
    app.max_frames_var = tk.IntVar(value=100)
    max_frames_entry = ttk.Entry(settings_frame, textvariable=app.max_frames_var, width=8)
    max_frames_entry.grid(row=0, column=4, padx=5, pady=2, sticky=tk.W)
    
    max_frames_help = ttk.Label(settings_frame, text="(0 = no limit)")
    max_frames_help.grid(row=0, column=5, padx=5, pady=2, sticky=tk.W)
    
    # Resize
    resize_label = ttk.Label(settings_frame, text="Resize:")
    resize_label.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
    
    resize_frame = ttk.Frame(settings_frame)
    resize_frame.grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
    
    app.resize_width_var = tk.IntVar(value=0)
    resize_width_entry = ttk.Entry(resize_frame, textvariable=app.resize_width_var, width=6)
    resize_width_entry.pack(side=tk.LEFT)
    
    resize_x_label = ttk.Label(resize_frame, text="x")
    resize_x_label.pack(side=tk.LEFT, padx=2)
    
    app.resize_height_var = tk.IntVar(value=0)
    resize_height_entry = ttk.Entry(resize_frame, textvariable=app.resize_height_var, width=6)
    resize_height_entry.pack(side=tk.LEFT)
    
    resize_help = ttk.Label(settings_frame, text="(0,0 = original size)")
    resize_help.grid(row=1, column=3, columnspan=3, padx=5, pady=2, sticky=tk.W)