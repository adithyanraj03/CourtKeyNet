"""
Geometric Consistency Loss (Numerically Stable Implementation)
Manuscript: Section 3.6, Equations (44)-(47)

Key improvements over naive implementation:
1. Log-ratio for edge loss (scale invariance)
2. Cosine similarity for angle loss (avoids acos gradient issues)
3. SmoothL1 loss (robust to outliers)
4. Warmup strategy for geometric losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricConsistencyLoss(nn.Module):
    """
    Enforces geometric constraints on predicted quadrilateral
    Uses stable formulations to prevent gradient explosion
    """
    def __init__(self):
        super().__init__()

    def compute_edges_diagonals(self, kpts):
        """
        kpts: (B, 4, 2) - normalized coordinates [0,1]
        Returns: edges (B,4), diagonals (B,2)
        """
        # Edge lengths with stable sqrt (add eps inside to prevent gradient explosion at 0)
        edges = []
        for i in range(4):
            p1 = kpts[:, i]
            p2 = kpts[:, (i+1) % 4]
            dist_sq = ((p1 - p2) ** 2).sum(dim=1)
            dist = torch.sqrt(dist_sq + 1e-8)  # Stable sqrt
            edges.append(dist)
        edges = torch.stack(edges, dim=1)  # (B, 4)
        
        # Diagonals
        d1_sq = ((kpts[:,0] - kpts[:,2]) ** 2).sum(dim=1)
        d2_sq = ((kpts[:,1] - kpts[:,3]) ** 2).sum(dim=1)
        d1 = torch.sqrt(d1_sq + 1e-8)
        d2 = torch.sqrt(d2_sq + 1e-8)
        diagonals = torch.stack([d1, d2], dim=1)  # (B, 2)
        
        return edges, diagonals

    def edge_loss(self, pred_kpts, gt_kpts):
        """
        Manuscript: Equation (44)
        Uses log-ratio formulation for scale invariance
        L_edge = Σ SmoothL1(log(ê_i), log(e_i))
        """
        pred_edges, _ = self.compute_edges_diagonals(pred_kpts)
        gt_edges, _ = self.compute_edges_diagonals(gt_kpts)
        
        # Log-ratio is more stable than direct ratio
        pred_log = torch.log(pred_edges + 1e-8)
        gt_log = torch.log(gt_edges + 1e-8)
        
        # Smooth L1 is more robust to outliers than L1
        return F.smooth_l1_loss(pred_log, gt_log, beta=0.1)

    def diagonal_loss(self, pred_kpts, gt_kpts):
        """
        Manuscript: Equation (45)
        Enforces diagonal ratio consistency
        """
        _, pred_diag = self.compute_edges_diagonals(pred_kpts)
        _, gt_diag = self.compute_edges_diagonals(gt_kpts)
        
        # Diagonal ratio (d1 / (d1 + d2)) - bounded [0, 1]
        pred_ratio = pred_diag[:,0] / (pred_diag.sum(dim=1) + 1e-8)
        gt_ratio = gt_diag[:,0] / (gt_diag.sum(dim=1) + 1e-8)
        
        return F.smooth_l1_loss(pred_ratio, gt_ratio, beta=0.1)

    def angle_loss(self, pred_kpts, gt_kpts):
        """
        Manuscript: Equation (46)
        Uses cosine similarity instead of acos for numerical stability
        L_angle = Σ SmoothL1(cos(θ̂_i), cos(θ_i))
        
        This avoids gradient singularities at cos(θ) = ±1
        """
        def compute_cos_angles(kpts):
            """Returns cosine of internal angles (more stable than angles)"""
            cos_angles = []
            for i in range(4):
                prev = kpts[:, (i-1) % 4]
                curr = kpts[:, i]
                next_pt = kpts[:, (i+1) % 4]
                
                v1 = prev - curr
                v2 = next_pt - curr
                
                # Normalize vectors
                v1_norm = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
                v2_norm = v2 / (torch.norm(v2, dim=1, keepdim=True) + 1e-8)
                
                # Cosine of angle (dot product of unit vectors)
                cos_angle = (v1_norm * v2_norm).sum(dim=1)
                cos_angles.append(cos_angle)
            
            return torch.stack(cos_angles, dim=1)  # (B, 4)
        
        pred_cos = compute_cos_angles(pred_kpts)
        gt_cos = compute_cos_angles(gt_kpts)
        
        return F.smooth_l1_loss(pred_cos, gt_cos, beta=0.1)

    def forward(self, pred_kpts, gt_kpts):
        """
        Returns: L_edge, L_diag, L_angle
        """
        l_edge = self.edge_loss(pred_kpts, gt_kpts)
        l_diag = self.diagonal_loss(pred_kpts, gt_kpts)
        l_angle = self.angle_loss(pred_kpts, gt_kpts)
        
        return l_edge, l_diag, l_angle


class TotalLoss(nn.Module):
    """
    Manuscript: Equation (47) with Warmup Strategy (Equation 48)
    
    L_total = λ_kpt·L_kpt + λ_hm·L_hm + λ_geo(t)·(L_edge + L_diag + L_angle)
    
    where λ_geo(t) = min(1, t/N_warmup) · λ_geo for gradual introduction
    """
    def __init__(self, config):
        super().__init__()
        self.weights = config['train']['loss_weights']
        self.geo_loss = GeometricConsistencyLoss()
        self.heatmap_sigma = config['model']['heatmap_sigma']
        
        # Warmup configuration
        self.geo_warmup_epochs = config['train'].get('geo_warmup_epochs', 10)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Call this at the start of each epoch for warmup calculation"""
        self.current_epoch = epoch

    def generate_heatmap(self, kpts, H, W):
        """
        Generate Gaussian heatmaps from keypoints
        kpts: (B, K, 2) normalized [0,1]
        """
        B, K = kpts.shape[:2]
        device = kpts.device
        
        # Create coordinate grid
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid_x = grid_x.view(1, 1, H, W)
        grid_y = grid_y.view(1, 1, H, W)
        
        # Compute Gaussian heatmaps
        kpts_x = kpts[:, :, 0].view(B, K, 1, 1)
        kpts_y = kpts[:, :, 1].view(B, K, 1, 1)
        
        sigma = self.heatmap_sigma / max(H, W)  # Normalize sigma
        
        dx = (grid_x - kpts_x) ** 2
        dy = (grid_y - kpts_y) ** 2
        heatmap = torch.exp(-(dx + dy) / (2 * sigma ** 2))
        
        return heatmap

    def forward(self, outputs, targets):
        """
        outputs: dict from model forward()
        targets: dict with 'kpts' (B,4,2)
        """
        pred_hm = outputs['heatmaps']
        pred_kpts = outputs['kpts_refined']
        gt_kpts = targets['kpts']
        
        # 1. Keypoint coordinate loss (use smooth L1 for robustness)
        l_kpt = F.smooth_l1_loss(pred_kpts, gt_kpts, beta=0.01)
        
        # 2. Heatmap loss
        gt_hm = self.generate_heatmap(gt_kpts, pred_hm.shape[2], pred_hm.shape[3])
        l_hm = F.mse_loss(torch.sigmoid(pred_hm), gt_hm)
        
        # 3. Geometric losses
        l_edge, l_diag, l_angle = self.geo_loss(pred_kpts, gt_kpts)
        
        # Warmup factor: linearly increase geometric loss weight
        # λ_geo(t) = min(1, t/N_warmup) per Manuscript Eq. 48
        if self.geo_warmup_epochs > 0:
            geo_scale = min(1.0, self.current_epoch / self.geo_warmup_epochs)
        else:
            geo_scale = 1.0
        
        # Weighted sum (Manuscript Eq. 47)
        total = (
            self.weights['keypoint'] * l_kpt +
            self.weights['heatmap'] * l_hm +
            geo_scale * self.weights['edge'] * l_edge +
            geo_scale * self.weights['diagonal'] * l_diag +
            geo_scale * self.weights['angle'] * l_angle
        )
        
        return {
            'total': total,
            'l_kpt': l_kpt,
            'l_hm': l_hm,
            'l_edge': l_edge,
            'l_diag': l_diag,
            'l_angle': l_angle,
            'geo_scale': torch.tensor(geo_scale)  # For logging
        }
