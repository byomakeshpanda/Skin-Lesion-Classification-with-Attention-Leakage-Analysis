import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results_visualization', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class SkinLesionDataset(Dataset):
    def __init__(self, data_dir, dataframe, transform=None):
        self.data_dir = data_dir
        self.df = dataframe.reset_index(drop=True)
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
            
        return image, label

train_csv = "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
test_csv = "ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"
train_dir = "ISBI2016_ISIC_Part3B_Training_Data"
test_dir = "ISBI2016_ISIC_Part3B_Test_Data"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Train class distribution:")
print(train_df.iloc[:, 1].value_counts())

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Stratified Split to handle Class Imbalance
train_subset, val_subset = train_test_split(
    train_df, test_size=0.2, random_state=42, 
    stratify=train_df.iloc[:, 1]
)

print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
train_dataset = SkinLesionDataset(train_dir, train_subset, train_transform)
val_dataset = SkinLesionDataset(train_dir, val_subset, val_transform)
test_dataset = SkinLesionDataset(test_dir, test_df, val_transform)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def get_model(model_name, num_classes=2):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

model_names = ['resnet18', 'efficientnet', 'mobilenet']
for name in model_names:
    model = get_model(name)
    print(f"{name} loaded successfully")
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        epoch_loss = running_loss / len(train_loader)
        
        train_losses.append(epoch_loss)
        val_accs.append(val_acc)
        
        print(f'Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        scheduler.step(epoch_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model.pth')
    
    return model, train_losses, val_accs, best_val_acc


def evaluate_model(model, test_loader, class_names=['Benign', 'Malignant'],save_path=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        plt.savefig(f'results_visualization/{save_path}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
    plt.show()
    plt.show()
    
    return accuracy, precision, recall, f1


models_to_train = ['resnet18', 'efficientnet', 'mobilenet']
trained_models = {}
results = {}

for model_name in models_to_train:
    print(f"\nTraining {model_name}...")
    model = get_model(model_name)
    trained_model, _, _, best_val_acc = train_model(model, train_loader, val_loader, num_epochs=10)
    torch.save(trained_model.state_dict(), f'{model_name}_best.pth')
    trained_models[model_name] = trained_model
    print(f"Evaluating {model_name} on test set...")
    acc, precision, recall, f1 = evaluate_model(trained_model, test_loader,save_path=model_name)
    results[model_name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[k]['accuracy'] for k in results.keys()],
    'Benign_Precision': [results[k]['precision'][0] for k in results.keys()],
    'Malignant_Precision': [results[k]['precision'][1] for k in results.keys()],
    'Benign_Recall': [results[k]['recall'][0] for k in results.keys()],
    'Malignant_Recall': [results[k]['recall'][1] for k in results.keys()],
    'Benign_F1': [results[k]['f1'][0] for k in results.keys()],
    'Malignant_F1': [results[k]['f1'][1] for k in results.keys()]
})
print("\nComparison of all models:")
print(results_df.round(4))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


models = results_df['Model']
accuracies = results_df['Accuracy']
ax1.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
ax1.set_title('Model Comparison - Test Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.8, 0.9)  # Adjust based on your results
for i, v in enumerate(accuracies):
    ax1.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
x = range(len(models))
width = 0.35
ax2.bar([i - width/2 for i in x], results_df['Benign_F1'], 
        width, label='Benign', color='lightblue')
ax2.bar([i + width/2 for i in x], results_df['Malignant_F1'], 
        width, label='Malignant', color='lightcoral')
ax2.set_title('Per-Class F1 Scores')
ax2.set_ylabel('F1 Score')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()

plt.tight_layout()
plt.savefig('results_visualization/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()



best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = trained_models[best_model_name]
print(f"\nBest model: {best_model_name}")