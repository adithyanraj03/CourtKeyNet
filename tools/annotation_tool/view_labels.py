import tkinter as tk
from tkinter import filedialog, Listbox, Scrollbar
from PIL import Image, ImageTk, ImageDraw
import os

class ViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Badminton Court Annotation Viewer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2d2d2d")

        # Directories
        self.img_dir = ""
        self.lbl_dir = ""
        self.image_files = []

        self.create_widgets()

    def create_widgets(self):
        # --- Top Control Panel ---
        control_frame = tk.Frame(self.root, height=60, bg="#333333", pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"bg": "#4CAF50", "fg": "white", "font": ("Arial", 10, "bold"), "padx": 15, "pady": 5, "bd": 0}
        
        btn_img = tk.Button(control_frame, text="📂 Select Images", command=self.select_img_dir, **btn_style)
        btn_img.pack(side=tk.LEFT, padx=15)

        btn_lbl = tk.Button(control_frame, text="📂 Select Labels", command=self.select_lbl_dir, **btn_style)
        btn_lbl.pack(side=tk.LEFT, padx=5)
        
        self.status_lbl = tk.Label(control_frame, text="Please select folders...", bg="#333333", fg="#cccccc", font=("Arial", 10))
        self.status_lbl.pack(side=tk.LEFT, padx=20)

        # --- Main Content Area ---
        content_frame = tk.Frame(self.root, bg="#2d2d2d")
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: File List
        list_container = tk.Frame(content_frame, width=300, bg="#2d2d2d")
        list_container.pack(side=tk.LEFT, fill=tk.Y)
        
        scrollbar = Scrollbar(list_container, bg="#333333")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = Listbox(list_container, yscrollcommand=scrollbar.set, width=35, 
                             bg="#404040", fg="white", selectbackground="#4CAF50", font=("Consolas", 10))
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind("<<ListboxSelect>>", self.on_select_image)

        # Right: Image Display
        self.canvas_frame = tk.Frame(content_frame, bg="black", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Handle resize
        self.canvas.bind("<Configure>", self.on_resize)
        self.current_image_path = None

    def select_img_dir(self):
        path = filedialog.askdirectory(title="Select Images Directory")
        if path:
            self.img_dir = path
            self.load_file_list()

    def select_lbl_dir(self):
        path = filedialog.askdirectory(title="Select Labels Directory")
        if path:
            self.lbl_dir = path
            self.update_status()
            # Refresh current image if selected
            if self.listbox.curselection():
                self.on_select_image(None)

    def load_file_list(self):
        if not self.img_dir: return
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = [f for f in os.listdir(self.img_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        self.image_files.sort()
        
        self.listbox.delete(0, tk.END)
        for f in self.image_files:
            self.listbox.insert(tk.END, f)
        self.update_status()

    def update_status(self):
        img_txt = f"{len(self.image_files)} Images" if self.img_dir else "No Images"
        lbl_txt = "Labels Set" if self.lbl_dir else "No Labels"
        self.status_lbl.config(text=f"{img_txt}  |  {lbl_txt}")

    def on_select_image(self, event):
        selection = self.listbox.curselection()
        if not selection: return
        filename = self.listbox.get(selection[0])
        self.current_image_path = os.path.join(self.img_dir, filename)
        self.show_image()

    def on_resize(self, event):
        if self.current_image_path:
            self.show_image()

    def show_image(self):
        if not self.current_image_path: return
        
        try:
            pil_image = Image.open(self.current_image_path)
        except: return

        # Draw Annotations
        draw = ImageDraw.Draw(pil_image)
        w, h = pil_image.size
        
        filename = os.path.basename(self.current_image_path)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.lbl_dir, label_name) if self.lbl_dir else None

        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    
                    # --- KEYPOINT DRAWING ---
                    # YOLO Pose format: class x y w h px1 py1 v1 px2 py2 v2 ...
                    if len(parts) > 5:
                        keypoints = parts[5:]
                        num_kpts = len(keypoints) // 3
                        
                        # Draw connections (for 4 court corners: 0->1->3->2->0)
                        # Assumes order: TL, TR, BR, BL
                        corners = []
                        
                        for i in range(num_kpts):
                            kx, ky, kv = keypoints[i*3], keypoints[i*3+1], keypoints[i*3+2]
                            px, py = kx * w, ky * h
                            corners.append((px, py))
                            
                            # Draw Point
                            r = 8
                            # Color code corners: TL=Red, TR=Green, BR=Blue, BL=Yellow
                            colors = ["#FF0000", "#00FF00", "#0088FF", "#FFFF00"]
                            c = colors[i % 4]
                            
                            draw.ellipse((px-r, py-r, px+r, py+r), fill=c, outline="white", width=2)
                            draw.text((px+10, py-10), f"P{i}", fill="white")

                        # Draw Lines between corners
                        if len(corners) == 4:
                            # Draw quadrilateral 0-1-2-3-0 (adjust based on your index order)
                            # Standard: 0:TL, 1:TR, 2:BR, 3:BL -> 0-1-2-3-0
                            # But often keypoints are defined differently. Just drawing a cycle here.
                            draw.line([corners[0], corners[1]], fill="#00FF00", width=3)
                            draw.line([corners[1], corners[2]], fill="#00FF00", width=3)
                            draw.line([corners[2], corners[3]], fill="#00FF00", width=3)
                            draw.line([corners[3], corners[0]], fill="#00FF00", width=3)

                    # --- BOUNDING BOX DRAWING (Fallback) ---
                    elif len(parts) == 5:
                        cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]
                        x1, y1 = (cx - bw/2) * w, (cy - bh/2) * h
                        x2, y2 = (cx + bw/2) * w, (cy + bh/2) * h
                        draw.rectangle([x1, y1, x2, y2], outline="#FF0000", width=3)

        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w > 10 and canvas_h > 10:
            ratio = min(canvas_w/w, canvas_h/h)
            new_w, new_h = int(w*ratio), int(h*ratio)
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor=tk.CENTER, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ViewerApp(root)
    root.mainloop()

