import torch
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import os

"""
Model implementation for animals 
"""
class AnimalNetwork(nn.Module):
    def __init__(self):
        super(AnimalNetwork, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)  # Three classes: bird, cat, dog

    def forward(self, x):
        return self.model(x)

# Definizione del modello per persone
class PeopleNetwork(nn.Module):
    def __init__(self):
        super(PeopleNetwork, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Two classes: male, female

    def forward(self, x):
        return self.model(x)

# Trasformazioni per il dataset
def get_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Funzione per addestrare e salvare un modello
def train_and_save_model(model, data_dir, model_name, num_classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_transforms = get_transforms()
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    
    # Verifica il numero di classi nel dataset
    if len(train_dataset.classes) != num_classes:
        raise ValueError(f"Numero di classi nel dataset ({len(train_dataset.classes)}) non corrisponde al numero di classi del modello ({num_classes}).")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(f"Inizio dell'addestramento del modello {model_name}...")

    for epoch in range(10):  # Numero di epoche
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Salvataggio del modello
    torch.save(model.state_dict(), model_name)
    print(f"Modello salvato come '{model_name}'")

if __name__ == "__main__":
    # Configurazione e addestramento dei modelli
    train_and_save_model(AnimalNetwork(), 'images/train/animal', 'animal_model.pth', 3)
    train_and_save_model(PeopleNetwork(), 'images/train/people', 'people_model.pth', 2)
