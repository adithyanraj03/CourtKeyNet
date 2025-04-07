import tkinter as tk
from tkinter import ttk

def create_main_window(app):
    """
    Create the main window structure for the annotation tool.
    
    Args:
        app: AnnotationTool instance
    """
    # Configure root window
    app.root.title("CourtKeyNet Annotation Tool")
    app.root.configure(bg="#f0f0f0")
    
    # Create main frames
    app.top_frame = ttk.Frame(app.root)
    app.top_frame.pack(fill=tk.X, padx=10, pady=5)
    
    app.main_frame = ttk.Frame(app.root)
    app.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    app.bottom_frame = ttk.Frame(app.root)
    app.bottom_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Split main frame into left (file browser) and right (canvas) parts
    app.left_frame = ttk.Frame(app.main_frame, width=250)
    app.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    app.left_frame.pack_propagate(False)
    
    app.right_frame = ttk.Frame(app.main_frame)
    app.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Create status bar
    app.status_frame = ttk.Frame(app.root)
    app.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    app.status_label = ttk.Label(app.status_frame, text="Ready", anchor=tk.W)
    app.status_label.pack(side=tk.LEFT, padx=5)
    
    app.progress_bar = ttk.Progressbar(app.status_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
    app.progress_bar.pack(side=tk.RIGHT, padx=5, pady=5)
    
    # Create style
    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat", background="#ccc")
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabel", background="#f0f0f0")