import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import time
from tqdm import tqdm
import json
import os

class ModelTrainer:
    def __init__(self, model_name='resnet34', num_classes=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data transforms with stronger augmentation
        self.transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        # Load CIFAR-100
        print("Loading CIFAR-100 dataset...")
        self.train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=self.transform_train
        )
        self.test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=self.transform_test
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=128, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Load model
        print(f"Loading {model_name} model...")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model = self.model.to(self.device)
        self.model_name = model_name
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=0.05
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=10
        )
        
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}', 
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return running_loss/len(self.train_loader), 100.*correct/total
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100.*correct/total
    
    def train(self, epochs=10):
        results = {
            'model': self.model_name,
            'epochs': [], 
            'train_loss': [], 
            'train_acc': [], 
            'test_acc': [], 
            'training_time': 0,
            'best_accuracy': 0
        }
        
        start_time = time.time()
        best_acc = 0
        
        for epoch in range(epochs):
            print(f'\n{"="*60}')
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'{"="*60}')
            
            train_loss, train_acc = self.train_epoch()
            test_acc = self.evaluate()
            self.scheduler.step()
            
            results['epochs'].append(epoch+1)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), 'vit_cifar100.pth')
                print(f'✅ New best model saved! Accuracy: {test_acc:.2f}%')
            
            print(f'\nResults:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Train Acc:  {train_acc:.2f}%')
            print(f'  Test Acc:   {test_acc:.2f}%')
            print(f'  Best Acc:   {best_acc:.2f}%')
        
        results['training_time'] = time.time() - start_time
        results['best_accuracy'] = best_acc
        
        # Save results
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f'\n{"="*60}')
        print(f'Training Complete!')
        print(f'{"="*60}')
        print(f'Best Accuracy: {best_acc:.2f}%')
        print(f'Training Time: {results["training_time"]/60:.2f} minutes')
        print(f'Model saved as: vit_cifar100.pth')
        print(f'Results saved as: results.json')
        
        return results

def main():
    print("="*60)
    print("CIFAR-100 Model Training")
    print("="*60)
    print()
    
    # Check for existing model
    if os.path.exists('vit_cifar100.pth'):
        print("⚠️  Found existing model file: vit_cifar100.pth")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Choose model
    print("\nAvailable models:")
    print("1. ResNet34 (Recommended - Fast & Accurate)")
    print("2. ResNet50 (More parameters)")
    print("3. ViT-Small (Vision Transformer)")
    
    choice = input("\nChoose model (1-3) [default: 1]: ").strip()
    
    model_map = {
        '1': 'resnet34',
        '2': 'resnet50',
        '3': 'vit_small_patch16_224',
        '': 'resnet34'
    }
    
    model_name = model_map.get(choice, 'resnet34')
    
    # Choose epochs
    epochs_input = input("\nNumber of epochs [default: 10]: ").strip()
    epochs = int(epochs_input) if epochs_input else 10
    
    print(f"\n🚀 Starting training with {model_name} for {epochs} epochs...")
    print()
    
    # Train
    trainer = ModelTrainer(model_name=model_name)
    results = trainer.train(epochs=epochs)
    
    print("\n✅ Training completed successfully!")
    print("\nNext steps:")
    print("  1. Test locally: python -m flask run")
    print("  2. Build Docker: docker build -t vit-classifier .")
    print("  3. Deploy: ./deploy.sh")

if __name__ == '__main__':
    main()