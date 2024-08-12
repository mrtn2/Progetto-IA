import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import pandas as pd
import torchvision.models as models
import matplotlib.pyplot as plt
import json

# Definizione del modello per persone
class PeopleNetwork(nn.Module):
    def __init__(self):
        super(PeopleNetwork, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Due classi: male, female

    def forward(self, x):
        return self.model(x)

# Caricamento del modello addestrato
model = PeopleNetwork()
model.load_state_dict(torch.load('people_model.pth'))
model.eval()

# Trasformazioni per le immagini
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

transform = get_transforms()

# Carica la configurazione dal file JSON
with open('config.json', 'r') as f:
    config = json.load(f)

# Funzione per applicare il filtro colorato
def apply_color_filter(image, gender):
    color_filters = config['filters']['people']
    if gender == 'male':
        color_filter = Image.new('RGB', image.size, color=color_filters['male'])
    elif gender == 'female':
        color_filter = Image.new('RGB', image.size, color=color_filters['female'])
    else:
        return image

    image = Image.blend(image, color_filter, alpha=0.5)
    return image

# Funzione che gestisce l'eliminazione selettiva delle cartelle
def clear_output_directory(output_dir, categories_to_clear):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in categories_to_clear:
        category_path = os.path.join(output_dir, category)
        if os.path.exists(category_path):
            shutil.rmtree(category_path)  # Rimuove la cartella specificata e il suo contenuto
        os.makedirs(category_path)  # Ricrea la cartella vuota

# Funzione per creare le sottocartelle se non esistono già
def create_category_folders(base_dir, categories):
    for category in categories:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

# Funzione per processare le immagini
def process_images(input_dir, output_dir, model):
    categories = ['female', 'male']
    clear_output_directory(output_dir, categories)

    image_names = os.listdir(input_dir)
    data = {"Name": [], "Gender": [], "Confidence": []}

    confidences = []

    for image_name in image_names:
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            gender_category = categories[predicted.item()]

            confidences.append(confidence.item())

        filtered_image = apply_color_filter(image, gender_category)

        output_path = os.path.join(output_dir, gender_category, f"filtered_{image_name}")
        filtered_image.save(output_path)

        # Rimuovere l'estensione dal nome dell'immagine per il grafico
        image_name_without_ext = os.path.splitext(image_name)[0]

        data["Name"].append(image_name_without_ext)
        data["Gender"].append(gender_category)
        data["Confidence"].append(confidence.item())

    df = pd.DataFrame(data)
    print(df)

    # Visualizzazione della confidenza per ogni immagine
    plt.figure(figsize=(10, 6))  # Ridotto la dimensione della finestra del plot
    plt.bar(df["Name"], df["Confidence"], color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Nome Immagine', fontsize=10)  # Ridotto la grandezza del testo dell'asse x
    plt.ylabel('Confidenza', fontsize=10)  # Ridotto la grandezza del testo dell'asse y
    plt.title('Confidenza della Predizione per Ogni Immagine', fontsize=12)  # Ridotto la grandezza del titolo
    plt.xticks(rotation=45, ha='right', fontsize=8)  # Ruota le etichette dell'asse x di 45 gradi e riduci la grandezza del testo
    plt.grid(True)
    plt.tight_layout()  # Per assicurare che le etichette non vengano tagliate
    plt.show()

# Esecuzione della funzione di elaborazione
input_dir = config['inference']['people']['input_dir']
output_dir = config['inference']['people']['output_dir']

process_images(input_dir, output_dir, model)
