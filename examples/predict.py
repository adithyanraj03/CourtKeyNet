import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms

# Import from our package
from courtkeynet import build_courtkeynet
from courtkeynet.transforms import get_transform
from courtkeynet.utils.visualization import draw_court

def parse_args():
    parser = argparse.ArgumentParser(description="CourtKeyNet Prediction")
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Path to input image, video, or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to output file or directory')
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video', 'batch'],
                       help='Prediction mode: image, video, or batch')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--save-frames', action='store_true', help='Save video frames with predictions')
    parser.add_argument('--frames-dir', type=str, default='frames', help='Directory to save frames')
    parser.add_argument('--save-labels', action='store_true', help='Save keypoints as label files')
    parser.add_argument('--labels-dir', type=str, default='labels', help='Directory to save label files')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (cuda or cpu)')
    
    return parser.parse_args()

def predict_image(model, img_path, output_path, img_size, device, save_labels=False, labels_dir=None):
    """Run prediction on a single image"""
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # original shape
    
    # Get transform
    transform = get_transform(img_size, is_train=False)
    
    # Apply transform
    transformed = transform(image=img)
    img_tensor = transformed["image"].unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get keypoints
    keypoints = output['keypoints'][0].cpu().numpy()
    
    # Draw court on image
    result_img = draw_court(img.copy(), keypoints, (w0, h0))
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"Prediction saved to {output_path}")
    else:
        cv2.imshow('Prediction', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save labels if requested
    if save_labels and labels_dir:
        os.makedirs(labels_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        save_keypoints_to_label(keypoints, label_path)
    
    return keypoints, result_img

def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model = build_courtkeynet()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    
    # Run prediction based on mode
    if args.mode == 'image':
        predict_image(
            model=model,
            img_path=args.input,
            output_path=args.output,
            img_size=args.img_size,
            device=device,
            save_labels=args.save_labels,
            labels_dir=args.labels_dir
        )
    elif args.mode == 'video':
        predict_video(
            model=model,
            video_path=args.input,
            output_path=args.output,
            img_size=args.img_size,
            device=device,
            save_frames=args.save_frames,
            frames_dir=args.frames_dir
        )
    elif args.mode == 'batch':
        predict_batch(
            model=model,
            img_dir=args.input,
            output_dir=args.output,
            img_size=args.img_size,
            device=device,
            save_labels=args.save_labels,
            labels_dir=args.labels_dir
        )

if __name__ == "__main__":
    main()