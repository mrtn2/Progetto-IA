import torch
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import os
import subprocess  # per automatizzazione di script in sequenza
import matplotlib.pyplot as plt  # per il grafico delle loss
import json
import jsonschema

# Carica lo schema di configurazione
with open('config_schema.json', 'r') as f:
    config_schema = json.load(f)

# Carica il file di configurazione
with open('config.json', 'r') as f:
    config = json.load(f)

# Valida il file di configurazione rispetto allo schema
try:
    jsonschema.validate(instance=config, schema=config_schema)
    print("Il file config.json è valido.")
except jsonschema.exceptions.ValidationError as e:
    print(f"Errore di validazione del file config.json: {e.message}")
    exit(1)


"""
Model implementation for animals 
"""
# Carica la configurazione dal file JSON
with open('config.json', 'r') as f:
    config = json.load(f)

# Definizione del modello per animali
class AnimalNetwork(nn.Module):
    def __init__(self):
        super(AnimalNetwork, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, config['training']['animal']['num_classes'])  # numero di classi dal JSON

    def forward(self, x):
        return self.model(x)

# Definizione del modello per persone
class PeopleNetwork(nn.Module):
    def __init__(self):
        super(PeopleNetwork, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, config['training']['people']['num_classes'])  # numero di classi dal JSON

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
def train_and_save_model(model, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    
    train_transforms = get_transforms()
    train_dataset = datasets.ImageFolder(root=config['data_dir'], transform=train_transforms)
    
    # Verifica il numero di classi nel dataset
    if len(train_dataset.classes) != config['num_classes']:
        raise ValueError(f"Numero di classi nel dataset ({len(train_dataset.classes)}) non corrisponde al numero di classi del modello ({config['num_classes']}).")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    print(f"Inizio dell'addestramento del modello {config['model_name']}...")

    losses = []  # Lista per memorizzare i valori di loss
    best_loss = float('inf')  # Inizializza la migliore loss con un valore molto alto
    early_stop_counter = 0
    patience = config['early_stopping']['patience']

    for epoch in range(config['epochs']):  # Numero di epoche
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

        # Salvataggio del modello migliore
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stop_counter = 0
            best_model_path = f"best_{config['model_name']}"
            torch.save(model.state_dict(), best_model_path)
            print(f"Modello migliorato e salvato in '{best_model_path}' con Loss: {best_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"Early stopping attivato. Addestramento interrotto")
            break


    # Salvataggio del modello finale
    final_model_path = config['model_name']
    torch.save(model.state_dict(), final_model_path)
    print(f"Modello salvato come '{final_model_path}")

    return losses  # Restituisce la lista delle loss per ogni epoca

if __name__ == "__main__":
    # Addestramento e salvataggio dei modelli
    animal_config = config['training']['animal']
    animal_losses = train_and_save_model(AnimalNetwork(), animal_config)

    people_config = config['training']['people']
    people_losses = train_and_save_model(PeopleNetwork(), people_config)

    # Creazione del grafico per le loss
    plt.figure(figsize=(10, 5))
    epochs = range(1, animal_config['epochs'] + 1) # Adattato al numero di epoche
    plt.plot(epochs, animal_losses, label='AnimalNetwork Loss')
    plt.plot(epochs, people_losses, label='PeopleNetwork Loss')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.title('Andamento delle Loss durante le Epoche')
    plt.legend()
    plt.savefig('training_loss_plot.png')  # Salva il grafico come immagine
    plt.show()  # Mostra il grafico

    # Esecuzione automatica dello script animal.py
    print("Esecuzione di animal.py...")
    subprocess.run(['python', 'Animal.py'])

    # Esecuzione automatica dello script gender.py
    print("Esecuzione di gender.py...")
    subprocess.run(['python', 'Gender.py'])
