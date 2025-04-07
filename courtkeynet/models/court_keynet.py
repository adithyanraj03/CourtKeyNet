import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..cpp import courtkeynet_cpp

# Import our custom modules
from .modules.octave_feature_extractor import OctaveFeatureExtractor
from .modules.keypoint_localization import KeypointLocalizationModule
from .modules.court_detector import CourtDetector
from .modules.quadrilateral_constraint import QuadrilateralConstraintModule

class CourtKeyNet(nn.Module):
    """
    Complete novel architecture for court keypoint detection
    """
    def __init__(self, num_keypoints=4):
        super(CourtKeyNet, self).__init__()
        
        # Feature extraction
        self.feature_extractor = OctaveFeatureExtractor()
        
        # Court detection
        self.court_detector = CourtDetector(feature_dim=256)
        
        # Keypoint localization
        self.keypoint_localizer = KeypointLocalizationModule(
            feature_dim=256, 
            num_keypoints=num_keypoints
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Detect court
        court_map = self.court_detector(features)
        
        # Localize keypoints
        keypoint_outputs = self.keypoint_localizer(features)
        
        return {
            'court_heatmap': court_map,
            'keypoints': keypoint_outputs['keypoints'],
            'keypoint_heatmaps': keypoint_outputs['heatmaps']
        }


def build_courtkeynet(pretrained=False, weights_path=None):
    """
    Build the CourtKeyNet model
    
    Args:
        pretrained: Whether to load pretrained weights
        weights_path: Path to weights file (if pretrained=True)
    
    Returns:
        Initialized CourtKeyNet model
    """
    model = CourtKeyNet(num_keypoints=4)
    
    if pretrained:
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
        else:
            raise ValueError("weights_path must be provided when pretrained=True")
        
    return model