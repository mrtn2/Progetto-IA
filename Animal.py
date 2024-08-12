import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt
import random
from PIL import ImageOps
import json

# Definizione del modello
class AnimalNetwork(nn.Module):
    def __init__(self):
        super(AnimalNetwork, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)  # Numero di classi

    def forward(self, x):
        return self.model(x)

# Caricamento del modello addestrato
model = AnimalNetwork()

# Assicurati che il percorso del file sia corretto
model_path = 'animal_model.pth'
if not os.path.exists(model_path):
    print(f"File del modello '{model_path}' non trovato. Assicurati di eseguire prima il file Model.py.")
    exit()

try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
except Exception as e:
    print(f"Errore nel caricare il modello: {e}")
    exit()

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carica la configurazione dal file JSON
with open('config.json', 'r') as f:
    config = json.load(f)

# Definizione dei filtri
def add_dog_filter(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)  # Ruota l'immagine sottosopra

def add_cat_filter(image):
    return image.convert('L').convert('RGB')  # Converti in bianco e nero

def add_bird_filter(image):
    return ImageOps.invert(image.convert('RGB'))

# Funzione che gestisce l'eliminazione selettiva delle cartelle specifiche per gli animali
def clear_output_directory(output_dir, categories_to_clear):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in categories_to_clear:
        category_path = os.path.join(output_dir, category)
        if os.path.exists(category_path):
            shutil.rmtree(category_path)  # Rimuove la cartella specificata e il suo contenuto
        os.makedirs(category_path)  # Ricrea la cartella vuota

# Funzione che crea le sottocartelle per ciascuna categoria se non esistono già
def create_category_folders(base_dir, categories):
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

# Funzione per selezionare un campione casuale di immagini
def get_random_images(image_list, num_images):
    return random.sample(image_list, min(num_images, len(image_list)))

# Funzione per processare le immagini
def process_images(input_dir, output_dir, model, num_images=100):
    categories = ['bird', 'cat', 'dog']
    
    clear_output_directory(output_dir, categories)
    create_category_folders(output_dir, categories)

    image_names = os.listdir(input_dir)
    random_images = get_random_images(image_names, num_images)

    data = {"Name": [], "Category": [], "Confidence": []}
    confidences = []

    for image_name in random_images:
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            animal_category = categories[predicted.item()]

            confidences.append(confidence.item())

        # Applicare il filtro appropriato
        if animal_category == 'dog':
            modified_image = add_dog_filter(image)
        elif animal_category == 'cat':
            modified_image = add_cat_filter(image)
        elif animal_category == 'bird':
            modified_image = add_bird_filter(image)
        else:
            modified_image = image  # Nessun filtro applicato

        # Salvare l'immagine modificata
        category_folder = os.path.join(output_dir, animal_category)
        modified_image_path = os.path.join(category_folder, image_name)
        
        modified_image.save(modified_image_path)

        # Rimuovere l'estensione dal nome dell'immagine per il grafico
        image_name_without_ext = os.path.splitext(image_name)[0]

        data["Name"].append(image_name_without_ext)
        data["Category"].append(animal_category)
        data["Confidence"].append(confidence.item())

    df = pd.DataFrame(data)
    print(df)

    # Visualizzazione delle confidenze per ogni immagine
    plt.figure(figsize=(10, 6))  # Ridotto la dimensione della finestra del plot
    plt.bar(df["Name"], df["Confidence"], color='green', edgecolor='black', alpha=0.7)
    plt.xlabel('Nome Immagine', fontsize=10)  # Ridotto la grandezza del testo dell'asse x
    plt.ylabel('Confidenza', fontsize=10)  # Ridotto la grandezza del testo dell'asse y
    plt.title('Confidenza della Predizione per Ogni Immagine', fontsize=12)  # Ridotto la grandezza del titolo
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Ruota le etichette dell'asse x di 45 gradi e riduci la grandezza del testo
    plt.grid(True)
    plt.tight_layout()  # Per assicurare che le etichette non vengano tagliate
    plt.show()

# Esecuzione della funzione di elaborazione
input_dir = config['inference']['animal']['input_dir']
output_dir = config['inference']['animal']['output_dir']

process_images(input_dir, output_dir, model, num_images=config['inference']['animal']['num_images'])
