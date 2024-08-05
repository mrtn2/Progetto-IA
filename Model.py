import torch
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn

# Definizione del modello
class AnimalNetwork(nn.Module):
    def __init__(self):
        super(AnimalNetwork, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)  # Numero di classi

    def forward(self, x):
        return self.model(x)

# Configurazione
data_dir = 'images/train'
model = AnimalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Trasformazioni per il dataset
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Creazione dei dataset e dataloader
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Addestramento
for epoch in range(10):  # Numero di epoche
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Salvataggio del modello
torch.save(model.state_dict(), 'animal_model.pth')
print("Modello salvato come 'animal_model.pth'")
