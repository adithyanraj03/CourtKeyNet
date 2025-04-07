import torch

class CourtAnnotationDecoder:
    """
    Utility class to convert between our custom annotation format and model inputs/outputs
    """
    @staticmethod
    def decode_keypoint_format(label_line, img_width, img_height):
        """
        Decode our custom keypoint format to model inputs
        Format: class x_center y_center width height kpt1_x kpt1_y kpt1_visible kpt2_x kpt2_y kpt2_visible ...
        """
        parts = label_line.strip().split()
        class_id = int(parts[0])
        
        # Bounding box - already normalized
        x_center, y_center = float(parts[1]), float(parts[2])
        width, height = float(parts[3]), float(parts[4])
        
        # Keypoints
        keypoints = []
        for i in range(5, len(parts), 3):
            if i + 2 < len(parts):
                kx = float(parts[i])
                ky = float(parts[i + 1])
                kv = int(float(parts[i + 2]))  # visibility
                keypoints.append([kx, ky, kv])
            
        return {
            'class_id': class_id,
            'box': torch.tensor([x_center, y_center, width, height]),
            'keypoints': torch.tensor(keypoints)
        }
    
    @staticmethod
    def encode_to_annotation_format(class_id, box, keypoints):
        """
        Encode model outputs to our custom annotation format
        """
        parts = [str(class_id)]
        
        # Add bounding box
        parts.extend([f"{val:.6f}" for val in box])
        
        # Add keypoints
        for kp in keypoints:
            parts.extend([f"{kp[0]:.6f}", f"{kp[1]:.6f}", "2"])  # assuming all predicted keypoints are visible
            
        return " ".join(parts)