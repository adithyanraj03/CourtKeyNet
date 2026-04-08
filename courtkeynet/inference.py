"""
CourtKeyNet Inference GUI v3 - With Confidence Scoring
Filters detections based on multiple confidence metrics:
1. Heatmap Peak Confidence (how strong is the detection)
2. Geometric Validity (is the quad convex and reasonable)
3. Combined Confidence Score
"""
import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import yaml

# Internal safetensors support (do not modify)
try:
    from utils._safetensors import load_weights as _load_weights
except ImportError:
    import sys as _sys
    _sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils._safetensors import load_weights as _load_weights

# Import CourtKeyNet
try:
    from models.courtkeynet import CourtKeyNet
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.courtkeynet import CourtKeyNet


class ConfidenceCalculator:
    """
    Computes multiple confidence metrics for court detection
    """
    
    @staticmethod
    def heatmap_confidence(heatmaps, kpts):
        """
        Compute confidence from heatmap peak values at predicted keypoint locations
        
        Args:
            heatmaps: (B, 4, H, W) heatmap tensor
            kpts: (B, 4, 2) predicted keypoints (normalized 0-1)
        
        Returns:
            confidence: (B,) tensor with mean confidence per sample
            per_keypoint: (B, 4) confidence per keypoint
        """
        B, K, H, W = heatmaps.shape
        
        # Apply softmax to get probabilities
        hm_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(hm_flat, dim=-1)  # (B, K, H*W)
        
        # Get the max probability (peak strength)
        peak_probs, _ = probs.max(dim=-1)  # (B, K)
        
        # Alternative: sample probability at predicted location
        # This measures if the prediction matches the peak
        grid = kpts.view(B, K, 1, 2) * 2 - 1  # -> [-1, 1]
        sampled = F.grid_sample(
            F.softmax(heatmaps.view(B, K, H, W).view(B*K, 1, H, W) / 0.1, dim=-1).view(B*K, 1, H, W),
            grid.view(B*K, 1, 1, 2),
            align_corners=False
        ).view(B, K)
        
        # Use peak probability as confidence
        # Based on testing:
        # - Random noise: ~0.03 peak
        # - Real courts: ~0.4-0.6 peak
        # Normalize so 0.1 = 0 and 0.5 = 1
        normalized_conf = (peak_probs - 0.05) / (0.4 - 0.05)
        normalized_conf = normalized_conf.clamp(0, 1)
        
        mean_conf = normalized_conf.mean(dim=1)  # (B,)
        
        return mean_conf, normalized_conf
    
    @staticmethod
    def geometric_validity(kpts):
        """
        Check if the predicted quadrilateral is geometrically valid
        
        Args:
            kpts: (B, 4, 2) keypoints in order TL, TR, BR, BL
            
        Returns:
            validity: (B,) tensor with validity score 0-1
            details: dict with individual checks
        """
        B = kpts.shape[0]
        device = kpts.device
        
        validity_scores = torch.ones(B, device=device)
        
        # 1. Check convexity using cross products
        # For a convex quad, all cross products should have the same sign
        def cross_2d(v1, v2):
            return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        
        # Edges: TL->TR, TR->BR, BR->BL, BL->TL
        edges = torch.stack([
            kpts[:, 1] - kpts[:, 0],  # TL->TR
            kpts[:, 2] - kpts[:, 1],  # TR->BR
            kpts[:, 3] - kpts[:, 2],  # BR->BL
            kpts[:, 0] - kpts[:, 3],  # BL->TL
        ], dim=1)  # (B, 4, 2)
        
        # Cross products of consecutive edges
        crosses = torch.stack([
            cross_2d(edges[:, 0], edges[:, 1]),
            cross_2d(edges[:, 1], edges[:, 2]),
            cross_2d(edges[:, 2], edges[:, 3]),
            cross_2d(edges[:, 3], edges[:, 0]),
        ], dim=1)  # (B, 4)
        
        # All same sign = convex
        is_convex = ((crosses > 0).all(dim=1) | (crosses < 0).all(dim=1)).float()
        
        # 2. Check area is reasonable (not too small, not too large)
        # Shoelace formula for area
        x = kpts[:, :, 0]
        y = kpts[:, :, 1]
        area = 0.5 * torch.abs(
            x[:, 0] * (y[:, 1] - y[:, 3]) +
            x[:, 1] * (y[:, 2] - y[:, 0]) +
            x[:, 2] * (y[:, 3] - y[:, 1]) +
            x[:, 3] * (y[:, 0] - y[:, 2])
        )
        
        # Area should be between 1% and 95% of image (normalized coords)
        area_valid = ((area > 0.01) & (area < 0.95)).float()
        
        # 3. Check aspect ratio is reasonable for a court
        # Width (avg of top and bottom edges)
        width_top = torch.norm(kpts[:, 1] - kpts[:, 0], dim=1)
        width_bot = torch.norm(kpts[:, 2] - kpts[:, 3], dim=1)
        avg_width = (width_top + width_bot) / 2
        
        # Height (avg of left and right edges)
        height_left = torch.norm(kpts[:, 3] - kpts[:, 0], dim=1)
        height_right = torch.norm(kpts[:, 2] - kpts[:, 1], dim=1)
        avg_height = (height_left + height_right) / 2
        
        aspect_ratio = avg_width / (avg_height + 1e-6)
        
        # Badminton court has roughly 2:1 aspect ratio (13.4m x 6.1m)
        # But camera angles can distort this significantly
        # Allow aspect ratios from 0.3 to 5.0
        aspect_valid = ((aspect_ratio > 0.3) & (aspect_ratio < 5.0)).float()
        
        # 4. Check that keypoints are in correct relative positions
        # TL should be top-left of center, TR top-right, etc.
        center = kpts.mean(dim=1)  # (B, 2)
        
        # TL should have x < center_x and y < center_y (in normalized coords, y increases downward)
        tl_valid = ((kpts[:, 0, 0] < center[:, 0]) & (kpts[:, 0, 1] < center[:, 1])).float()
        tr_valid = ((kpts[:, 1, 0] > center[:, 0]) & (kpts[:, 1, 1] < center[:, 1])).float()
        br_valid = ((kpts[:, 2, 0] > center[:, 0]) & (kpts[:, 2, 1] > center[:, 1])).float()
        bl_valid = ((kpts[:, 3, 0] < center[:, 0]) & (kpts[:, 3, 1] > center[:, 1])).float()
        
        position_valid = (tl_valid + tr_valid + br_valid + bl_valid) / 4.0
        
        # Combine all validity checks
        validity_scores = (
            0.3 * is_convex +
            0.2 * area_valid +
            0.2 * aspect_valid +
            0.3 * position_valid
        )
        
        return validity_scores, {
            'convex': is_convex,
            'area': area,
            'area_valid': area_valid,
            'aspect_ratio': aspect_ratio,
            'aspect_valid': aspect_valid,
            'position_valid': position_valid
        }
    
    @staticmethod
    def heatmap_entropy(heatmaps):
        """
        Compute entropy of heatmaps - lower entropy = more confident
        
        Args:
            heatmaps: (B, 4, H, W)
            
        Returns:
            confidence: (B,) - inverse normalized entropy
        """
        B, K, H, W = heatmaps.shape
        
        hm_flat = heatmaps.view(B, K, -1)
        probs = F.softmax(hm_flat, dim=-1)
        
        # Entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, K)
        
        # Max entropy for uniform distribution = log(H*W)
        max_entropy = np.log(H * W)
        
        # Normalize and invert (low entropy = high confidence)
        normalized_entropy = entropy / max_entropy
        confidence = 1.0 - normalized_entropy.mean(dim=1)  # (B,)
        
        return confidence


class CourtKeyNetTesterV3:
    def __init__(self, root):
        self.root = root
        self.root.title("CourtKeyNet Inference Studio")
        self.root.geometry("1100x800")
        
        # Colors
        self.colors = {
            "bg": "#1e1e1e",
            "fg": "#ffffff",
            "panel": "#2d2d2d",
            "accent": "#007acc",
            "success": "#4caf50",
            "warning": "#ff9800",
            "error": "#f44336",
            "overlay_high": (160, 90, 145),  # Professional grey-purple for high confidence
            "overlay_medium": (0, 165, 255), # Orange for medium
            "overlay_low": (0, 0, 255),      # Red for low
        }
        
        self.root.configure(bg=self.colors["bg"])
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure Styles
        self.style.configure("Dark.TFrame", background=self.colors["bg"])
        self.style.configure("Panel.TFrame", background=self.colors["panel"], relief="flat")
        self.style.configure("Header.TLabel", background=self.colors["panel"], foreground=self.colors["fg"], font=("Segoe UI", 11, "bold"))
        self.style.configure("Normal.TLabel", background=self.colors["panel"], foreground=self.colors["fg"], font=("Segoe UI", 10))
        self.style.configure("Status.TLabel", background=self.colors["panel"], foreground="gray", font=("Consolas", 9))
        
        # State
        self.model_path = tk.StringVar()
        self.source_path = tk.StringVar()
        self.is_running = False
        self.source_type = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Confidence settings
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.use_heatmap_conf = tk.BooleanVar(value=True)
        self.use_geometric_conf = tk.BooleanVar(value=True)
        self.use_entropy_conf = tk.BooleanVar(value=True)
        self.opacity = 0.4  # Increased opacity
        self.hide_low_conf = tk.BooleanVar(value=False)  # Hide detections below threshold
        
        self.confidence_calc = ConfidenceCalculator()
        
        self.create_ui()

    def create_ui(self):
        # Header
        header_frame = ttk.Frame(self.root, style="Panel.TFrame")
        header_frame.pack(fill="x", padx=0, pady=0)
        
        title = tk.Label(header_frame, text="🏸 CourtKeyNet Inference Studio", 
                         bg=self.colors["accent"], fg="white", 
                         font=("Segoe UI", 14, "bold"), pady=12)
        title.pack(fill="x")

        # Main Layout
        main_container = ttk.Frame(self.root, style="Dark.TFrame")
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Left Panel
        controls_panel = ttk.Frame(main_container, style="Panel.TFrame")
        controls_panel.pack(side="left", fill="y", padx=(0, 15), ipadx=15, ipady=15)
        
        # 1. Model Selection
        ttk.Label(controls_panel, text="1. Model Selection", style="Header.TLabel").pack(anchor="w", pady=(0, 8))
        
        self.btn_load_model = tk.Button(controls_panel, text="📂 Load Weights (.pt / .safetensors)", 
                                      bg=self.colors["panel"], fg="white", relief="flat",
                                      command=self.select_model)
        self.btn_load_model.pack(fill="x", pady=3)
        
        self.lbl_model_status = ttk.Label(controls_panel, text="No model loaded", style="Status.TLabel")
        self.lbl_model_status.pack(anchor="w")

        # 2. Source Selection
        ttk.Label(controls_panel, text="2. Input Source", style="Header.TLabel").pack(anchor="w", pady=(15, 8))
        
        self.btn_load_file = tk.Button(controls_panel, text="🎬 Select File (Video/Image)", 
                                     bg=self.colors["panel"], fg="white", relief="flat",
                                     command=self.select_file)
        self.btn_load_file.pack(fill="x", pady=2)

        self.btn_load_dir = tk.Button(controls_panel, text="📁 Select Image Directory", 
                                    bg=self.colors["panel"], fg="white", relief="flat",
                                    command=self.select_directory)
        self.btn_load_dir.pack(fill="x", pady=2)
        
        self.lbl_source_status = ttk.Label(controls_panel, text="No source selected", style="Status.TLabel")
        self.lbl_source_status.pack(anchor="w")

        # 3. Confidence Settings
        ttk.Label(controls_panel, text="3. Confidence Settings", style="Header.TLabel").pack(anchor="w", pady=(15, 8))
        
        # Threshold slider
        ttk.Label(controls_panel, text="Confidence Threshold:", style="Normal.TLabel").pack(anchor="w")
        self.slider_conf = tk.Scale(controls_panel, from_=0.0, to=1.0, resolution=0.05,
                                   orient="horizontal", bg=self.colors["panel"], fg="white",
                                   highlightthickness=0, variable=self.conf_threshold, length=220)
        self.slider_conf.pack(fill="x")
        
        # Confidence method checkboxes
        conf_frame = ttk.Frame(controls_panel, style="Panel.TFrame")
        conf_frame.pack(fill="x", pady=5)
        
        tk.Checkbutton(conf_frame, text="Heatmap Peak", variable=self.use_heatmap_conf,
                      bg=self.colors["panel"], fg="white", selectcolor=self.colors["accent"],
                      activebackground=self.colors["panel"]).pack(anchor="w")
        tk.Checkbutton(conf_frame, text="Geometric Validity", variable=self.use_geometric_conf,
                      bg=self.colors["panel"], fg="white", selectcolor=self.colors["accent"],
                      activebackground=self.colors["panel"]).pack(anchor="w")
        tk.Checkbutton(conf_frame, text="Entropy (experimental)", variable=self.use_entropy_conf,
                      bg=self.colors["panel"], fg="white", selectcolor=self.colors["accent"],
                      activebackground=self.colors["panel"]).pack(anchor="w")
        
        # Hide low confidence option
        tk.Checkbutton(conf_frame, text="Hide detections below threshold", variable=self.hide_low_conf,
                      bg=self.colors["panel"], fg="white", selectcolor=self.colors["accent"],
                      activebackground=self.colors["panel"]).pack(anchor="w", pady=(5,0))

        # 4. Visualization Settings
        ttk.Label(controls_panel, text="4. Visualization", style="Header.TLabel").pack(anchor="w", pady=(15, 8))
        
        ttk.Label(controls_panel, text="Overlay Opacity:", style="Normal.TLabel").pack(anchor="w")
        self.slider_opacity = tk.Scale(controls_panel, from_=0.0, to=1.0, resolution=0.1, 
                                     orient="horizontal", bg=self.colors["panel"], fg="white",
                                     highlightthickness=0, command=self.update_opacity, length=220)
        self.slider_opacity.set(0.4)  # Increased opacity
        self.slider_opacity.pack(fill="x")
        
        # 5. Action Button
        self.btn_start = tk.Button(controls_panel, text="▶ START INFERENCE", 
                                 bg=self.colors["success"], fg="white", font=("Segoe UI", 11, "bold"),
                                 relief="flat", state="disabled", command=self.start_inference)
        self.btn_start.pack(fill="x", side="bottom", pady=10)
        
        # Right Panel - Info
        info_frame = ttk.Frame(main_container, style="Panel.TFrame")
        info_frame.pack(side="right", fill="both", expand=True)
        
        info_text = (
            "🎯 CONFIDENCE-AWARE DETECTION\n\n"
            "This version includes confidence scoring to filter\n"
            "out unreliable detections on:\n"
            "  • Partial courts\n"
            "  • Non-court images\n"
            "  • Occluded/unclear views\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "CONFIDENCE METRICS:\n\n"
            "📊 Heatmap Peak\n"
            "   How strong the detection peaks are.\n"
            "   Weak peaks = model is uncertain.\n\n"
            "📐 Geometric Validity\n"
            "   Checks if quad is convex, reasonable\n"
            "   size, and corners are positioned correctly.\n\n"
            "📉 Entropy (experimental)\n"
            "   Low entropy = confident detection.\n"
            "   High entropy = spread-out uncertainty.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "COLOR CODING:\n"
            "  🟢 Green  = High confidence (> threshold)\n"
            "  🟠 Orange = Medium confidence\n"
            "  🔴 Red    = Low confidence (hidden by default)\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "CONTROLS:\n"
            "  Q = Quit\n"
            "  SPACE = Pause/Resume\n"
            "  N/→ = Next Image\n"
            "  P/← = Previous Image"
        )
        
        lbl_info = tk.Label(info_frame, text=info_text, bg="black", fg="#cccccc", 
                          font=("Consolas", 9), justify="left", padx=15, pady=15, anchor="nw")
        lbl_info.pack(fill="both", expand=True)

    def select_model(self):
        # Default to the finetuned model path
        initial_dir = "runs/courtkeynet_finetune"
        if not os.path.exists(initial_dir):
            initial_dir = "runs/courtkeynet"
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()
            
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                ("Model Weights", "*.pt *.safetensors"),
                ("SafeTensors", "*.safetensors"),
                ("PyTorch", "*.pt"),
            ]
        )
        if path:
            self.model_path.set(path)
            fmt = "safetensors" if path.endswith(".safetensors") else "pt"
            self.lbl_model_status.config(
                text=f"[{fmt}] {os.path.basename(path)}",
                foreground=self.colors["success"]
            )
            try:
                self.load_model(path)
                self.check_ready()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")
                self.model = None

    def load_model(self, path):
        print(f"Loading weights from {path}...")
        ckpt = _load_weights(path, device=self.device)
        
        if 'config' in ckpt:
            config = ckpt['config']
        else:
            with open('configs/courtkeynet.yaml', 'r') as f:
                config = yaml.safe_load(f)
        
        self.model = CourtKeyNet(config).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        
        epoch = ckpt.get('epoch', '?')
        val_loss = ckpt.get('best_val_loss', '?')
        print(f"Model loaded: epoch={epoch}, val_loss={val_loss}")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Media Files", "*.mp4 *.avi *.mov *.mkv *.jpg *.png *.jpeg")
        ])
        if path:
            self.source_path.set(path)
            ext = os.path.splitext(path)[1].lower()
            self.source_type = 'image' if ext in ['.jpg', '.png', '.jpeg'] else 'video'
            self.lbl_source_status.config(
                text=f"[{self.source_type.upper()}] {os.path.basename(path)}", 
                foreground=self.colors["success"]
            )
            self.check_ready()

    def select_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.source_path.set(path)
            self.source_type = 'directory'
            self.lbl_source_status.config(
                text=f"[DIR] {os.path.basename(path)}", 
                foreground=self.colors["success"]
            )
            self.check_ready()

    def update_opacity(self, val):
        self.opacity = float(val)

    def check_ready(self):
        if self.model_path.get() and self.source_path.get() and self.model is not None:
            self.btn_start.config(state="normal", bg=self.colors["success"])

    def compute_confidence(self, heatmaps, kpts):
        """Compute combined confidence score"""
        scores = []
        weights = []
        
        if self.use_heatmap_conf.get():
            hm_conf, _ = self.confidence_calc.heatmap_confidence(heatmaps, kpts)
            scores.append(hm_conf)
            weights.append(0.4)
        
        if self.use_geometric_conf.get():
            geo_conf, _ = self.confidence_calc.geometric_validity(kpts)
            scores.append(geo_conf)
            weights.append(0.4)
        
        if self.use_entropy_conf.get():
            ent_conf = self.confidence_calc.heatmap_entropy(heatmaps)
            scores.append(ent_conf)
            weights.append(0.2)
        
        if not scores:
            return torch.ones(kpts.shape[0], device=kpts.device)
        
        # Weighted average
        total_weight = sum(weights)
        combined = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        return combined

    def get_overlay_color(self, confidence):
        """Get color based on confidence level"""
        threshold = self.conf_threshold.get()
        
        if confidence >= threshold:
            return self.colors["overlay_high"]  # Green
        elif confidence >= threshold * 0.6:
            return self.colors["overlay_medium"]  # Orange
        else:
            return self.colors["overlay_low"]  # Red

    def draw_court_overlay(self, img, kpts, confidence):
        """Draw court with confidence-based coloring"""
        threshold = self.conf_threshold.get()
        
        # Skip if below threshold and hide is selected
        if confidence < threshold and self.hide_low_conf.get():
            # Draw a small indicator that detection was skipped
            cv2.putText(img, f"No confident detection ({confidence:.2f})", 
                       (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (100, 100, 100), 2)
            return img
        
        overlay = img.copy()
        
        # Scale keypoints to image size
        h, w = img.shape[:2]
        points = kpts.copy()
        points[:, 0] *= w
        points[:, 1] *= h
        points = points.astype(np.int32)
        
        # Get color based on confidence
        color = self.get_overlay_color(confidence)
        
        # Fill polygon
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, self.opacity, img, 1 - self.opacity, 0, img)
        
        # Draw borders
        border_color = (255, 255, 255) if confidence >= threshold else (128, 128, 128)
        cv2.polylines(img, [points], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)
        
        # Draw corners
        corner_colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
        labels = ["TL", "TR", "BR", "BL"]
        
        for i, pt in enumerate(points):
            if i >= 4: break
            pt_tuple = tuple(pt)
            cv2.circle(img, pt_tuple, 6, (0, 0, 0), -1)
            cv2.circle(img, pt_tuple, 4, corner_colors[i], -1)
            cv2.putText(img, labels[i], (pt_tuple[0]+10, pt_tuple[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Confidence indicator
        conf_text = f"Confidence: {confidence:.2f}"
        conf_color = color
        cv2.putText(img, conf_text, (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        return img

    def process_frame(self, frame):
        """Process a single frame with confidence scoring"""
        try:
            # Preprocess
            img_in = cv2.resize(frame, (640, 640))
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_in).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                kpts = outputs['kpts_refined']  # (1, 4, 2)
                heatmaps = outputs['heatmaps']  # (1, 4, H, W)
                
                # Compute confidence
                confidence = self.compute_confidence(heatmaps, kpts)
                confidence = confidence[0].item()
                
                kpts_np = kpts[0].cpu().numpy()
            
            # Visualization with confidence
            annotated = self.draw_court_overlay(frame, kpts_np, confidence)
            return annotated
            
        except Exception as e:
            print(f"Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return frame

    def start_inference(self):
        if self.is_running: return
        self.is_running = True
        self.btn_start.config(text="Running...", state="disabled")
        
        thread = threading.Thread(target=self.run_inference_loop)
        thread.daemon = True
        thread.start()

    def run_inference_loop(self):
        source = self.source_path.get()
        cv2.namedWindow("CourtKeyNet Inference Studio", cv2.WINDOW_NORMAL)
        
        if self.source_type == 'video':
            cap = cv2.VideoCapture(source)
            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret: break
                
                processed = self.process_frame(frame)
                cv2.imshow("CourtKeyNet Inference Studio", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
            
        elif self.source_type == 'image':
            frame = cv2.imread(source)
            if frame is not None:
                processed = self.process_frame(frame)
                cv2.imshow("CourtKeyNet Inference Studio", processed)
                while self.is_running:
                    if cv2.waitKey(100) & 0xFF == ord('q'): break
                    
        elif self.source_type == 'directory':
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            images = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(valid_exts)]
            images.sort()
            
            if not images:
                print("No images found")
                self.stop_inference()
                return

            idx = 0
            paused = False
            
            while self.is_running and idx < len(images):
                img_path = images[idx]
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    if frame.shape[1] > 1920:
                        scale = 1920 / frame.shape[1]
                        frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
                        
                    processed = self.process_frame(frame)
                    
                    cv2.putText(processed, f"{os.path.basename(img_path)} ({idx+1}/{len(images)})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("CourtKeyNet Inference Studio", processed)
                    
                    key = cv2.waitKey(0 if paused else 2000) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = not paused
                    elif key == ord('n') or key == 83:
                        idx += 1
                        paused = True
                    elif key == ord('p') or key == 81:
                        idx = max(0, idx - 1)
                        paused = True
                    elif not paused:
                        idx += 1
                else:
                    idx += 1

        cv2.destroyAllWindows()
        self.stop_inference()

    def stop_inference(self, error_msg=None):
        self.is_running = False
        self.root.after(0, lambda: self.reset_ui(error_msg))

    def reset_ui(self, error_msg=None):
        self.btn_start.config(text="▶ START INFERENCE", state="normal")
        if error_msg:
            messagebox.showerror("Error", error_msg)


if __name__ == "__main__":
    root = tk.Tk()
    app = CourtKeyNetTesterV3(root)
    root.mainloop()
