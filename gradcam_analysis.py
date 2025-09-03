import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {'benign': 0, 'malignant': 1}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id = self.df.iloc[idx, 0]
        image_path = os.path.join(self.data_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        raw_label = self.df.iloc[idx, 1]
        if isinstance(raw_label, str):
            label = self.label_map[raw_label.lower()]
        else:
            label = int(raw_label)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, image_id

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Clear gradients
        self.model.zero_grad()
        
        # Backward pass
        target = output[0][class_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def cleanup(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def load_segmentation_mask(image_id, mask_dir):
    """Load and preprocess segmentation mask"""
    mask_path = os.path.join(mask_dir, f'{image_id}_Segmentation.png')
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        return (mask > 127).astype(np.float32)  # Binary mask
    return None

def calculate_attention_leakage(gradcam_heatmap, segmentation_mask):
    """Calculate attention leakage metrics - FIXED VERSION"""
    if segmentation_mask is None:
        return None, None, None
    
    # Ensure binary mask
    lesion_mask = (segmentation_mask > 0.5).astype(np.float32)
    background_mask = 1 - lesion_mask
    
    # Calculate total attention
    total_attention = gradcam_heatmap.sum()
    if total_attention == 0:
        return 0, 0, 0
    
    # Calculate attention scores
    lesion_pixels = lesion_mask.sum()
    background_pixels = background_mask.sum()
    
    if lesion_pixels > 0:
        inside_score = (gradcam_heatmap * lesion_mask).sum() / lesion_pixels
    else:
        inside_score = 0
        
    if background_pixels > 0:
        outside_score = (gradcam_heatmap * background_mask).sum() / background_pixels
    else:
        outside_score = 0
    
    # Leakage ratio
    leakage = (gradcam_heatmap * background_mask).sum() / total_attention
    
    return inside_score, outside_score, leakage

def visualize_gradcam_with_leakage(original_img, gradcam, mask, pred_class, true_class, leakage_score, save_path=None):
    """Visualize results with leakage analysis"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original\nTrue: {["Benign", "Malignant"][true_class]}')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    axes[1].imshow(gradcam, cmap='jet')
    axes[1].set_title(f'Grad-CAM\nPred: {["Benign", "Malignant"][pred_class]}')
    axes[1].axis('off')
    
    # Segmentation mask
    if mask is not None:
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Lesion Mask')
    else:
        axes[2].text(0.5, 0.5, 'No Mask', ha='center', va='center')
        axes[2].set_title('No Mask')
    axes[2].axis('off')
    
    # Overlay with leakage info
    if mask is not None:
        # Create colored overlay
        overlay = np.zeros((224, 224, 3))
        overlay[:, :, 0] = gradcam * (1 - mask)  # Red for leaked attention
        overlay[:, :, 1] = gradcam * mask        # Green for correct attention
        overlay = np.clip(overlay, 0, 1)
        
        axes[3].imshow(overlay)
        axes[3].set_title(f'Attention Analysis\nLeakage: {leakage_score:.3f}')
    else:
        axes[3].imshow(gradcam, cmap='jet')
        axes[3].set_title('No Analysis')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # SAVE PLOT
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    plt.show()

# Main analysis function
def run_gradcam_analysis():
    # Update these paths
    test_csv = r"ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"
    test_dir = r"ISBI2016_ISIC_Part3B_Test_Data"
    mask_dir = test_dir 
    model_path = "efficientnet_best.pth" 
    
    # CREATE RESULTS DIRECTORIES
    os.makedirs('results_visualization/gradcam_plots', exist_ok=True)
    os.makedirs('results_visualization/summary_plots', exist_ok=True)
    
    # Load model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    

    target_layer = model.features[-1][0]  # Last conv layer for EfficientNet
    gradcam = GradCAM(model, target_layer)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SimpleDataset(test_dir, test_csv, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = []
    class_names = ['Benign', 'Malignant']
    
    print("Running Grad-CAM analysis with attention leakage...")
    
    for i, (image, label, image_id) in enumerate(tqdm(loader)):
        if i >= 20:  # Analyze first 20 samples
            break
            
        image = image.to(device)
        
        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(dim=1).item()
        
        cam = gradcam.generate_cam(image, pred_class)
        mask = load_segmentation_mask(image_id[0], mask_dir)
        inside_score, outside_score, leakage = calculate_attention_leakage(cam, mask)
        
        if inside_score is not None:
            # STORE CAM AND MASK IN RESULTS
            results.append({
                'image_id': image_id[0],
                'true_label': label.item(),
                'pred_label': pred_class,
                'correct': label.item() == pred_class,
                'inside_score': inside_score,
                'outside_score': outside_score,
                'leakage': leakage,
                'gradcam': cam,
                'mask': mask
            })
        
        # Save individual visualizations for first 3 samples
        if i < 3 and mask is not None:
            # Denormalize image for visualization
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img_denorm = image.cpu().squeeze() * std[:, None, None] + mean[:, None, None]
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_np = img_denorm.permute(1, 2, 0).numpy()
            
            save_path = f'results_visualization/gradcam_plots/{image_id[0]}_detailed.png'
            visualize_gradcam_with_leakage(
                img_np, cam, mask, pred_class, label.item(), leakage, save_path
            )
    
    # Cleanup
    gradcam.cleanup()
    
    # Analyze results
    if results:
        df = pd.DataFrame(results)
        print(f"\nAttention Leakage Analysis Results:")
        print(f"Total samples analyzed: {len(df)}")
        print(f"Overall mean leakage: {df['leakage'].mean():.4f}")
        print(f"Std leakage: {df['leakage'].std():.4f}")
        
        # Per-class analysis
        for class_idx, class_name in enumerate(class_names):
            class_data = df[df['true_label'] == class_idx]
            if len(class_data) > 0:
                print(f"{class_name} lesions:")
                print(f"  Mean leakage: {class_data['leakage'].mean():.4f}")
                print(f"  Mean inside attention: {class_data['inside_score'].mean():.4f}")
                print(f"  Mean outside attention: {class_data['outside_score'].mean():.4f}")
        
        # CREATE AND SAVE SUMMARY PLOTS
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Attention Leakage by Class (Box Plot)
        benign_leakage = df[df['true_label'] == 0]['leakage']
        malignant_leakage = df[df['true_label'] == 1]['leakage']
        
        ax1.boxplot([benign_leakage, malignant_leakage], labels=['Benign', 'Malignant'])
        ax1.set_title('Attention Leakage by Class')
        ax1.set_ylabel('Leakage Score')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Leakage Distribution
        ax2.hist(df['leakage'], bins=15, alpha=0.7, edgecolor='black', color='skyblue')
        ax2.axvline(df['leakage'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["leakage"].mean():.3f}')
        ax2.set_xlabel('Leakage Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Attention Leakage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Inside vs Outside Attention Scores
        ax3.scatter(df['inside_score'], df['outside_score'], 
                   c=df['true_label'], cmap='RdYlBu', alpha=0.7)
        ax3.set_xlabel('Inside Attention Score')
        ax3.set_ylabel('Outside Attention Score')
        ax3.set_title('Inside vs Outside Attention')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal attention line')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mean Leakage by Class (Bar Plot)
        class_leakage = [benign_leakage.mean(), malignant_leakage.mean()]
        bars = ax4.bar(class_names, class_leakage, color=['lightblue', 'lightcoral'])
        ax4.set_title('Mean Attention Leakage by Class')
        ax4.set_ylabel('Mean Leakage Score')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, class_leakage)):
            ax4.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # SAVE SUMMARY PLOT
        plt.savefig('results_visualization/summary_plots/attention_leakage_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # CREATE ACCURACY VS LEAKAGE PLOT
        plt.figure(figsize=(10, 6))
        correct_cases = df[df['correct'] == True]['leakage']
        incorrect_cases = df[df['correct'] == False]['leakage']
        
        plt.boxplot([correct_cases, incorrect_cases], 
                   labels=['Correct Predictions', 'Incorrect Predictions'])
        plt.title('Attention Leakage vs Prediction Accuracy')
        plt.ylabel('Leakage Score')
        plt.grid(True, alpha=0.3)
        
        # Add statistical info
        plt.text(0.02, 0.98, f'Correct mean: {correct_cases.mean():.3f}\nIncorrect mean: {incorrect_cases.mean():.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.savefig('results_visualization/summary_plots/leakage_vs_accuracy.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results CSV
        df_save = df.drop(['gradcam', 'mask'], axis=1)  # Remove arrays for CSV
        df_save.to_csv('results_visualization/gradcam_leakage_results.csv', index=False)
        print("Results saved to 'results_visualization/gradcam_leakage_results.csv'")
        
        print(f"\nPlots saved to:")
        print(f"  - results_visualization/summary_plots/attention_leakage_analysis.png")
        print(f"  - results_visualization/summary_plots/leakage_vs_accuracy.png")
        print(f"  - results_visualization/gradcam_plots/ (individual visualizations)")
        
    else:
        print("No valid results found. Check if segmentation masks exist.")

if __name__ == "__main__":
    run_gradcam_analysis()
