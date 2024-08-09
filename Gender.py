import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import pandas as pd
import torchvision.models as models

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

# Funzione per applicare il filtro colorato
def apply_color_filter(image, gender):
    if gender == 'male':
        # Filtro azzurro
        color_filter = Image.new('RGB', image.size, color='lightblue')
    elif gender == 'female':
        # Filtro rosa chiaro
        color_filter = Image.new('RGB', image.size, color='lightpink')
    else:
        return image

    # Sovrapponi il filtro all'immagine
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
    clear_output_directory(output_dir, categories)  # Passa solo le categorie relative a gender

    image_names = os.listdir(input_dir)
    data = {"Name": [], "Gender": []}
    
    for image_name in image_names:
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            gender_category = categories[predicted.item()]

        # Applica il filtro colorato
        filtered_image = apply_color_filter(image, gender_category)

        # Salva l'immagine modificata nella rispettiva cartella
        output_path = os.path.join(output_dir, gender_category, f"filtered_{image_name}")
        filtered_image.save(output_path)

        data["Name"].append(image_name)
        data["Gender"].append(gender_category)

    df = pd.DataFrame(data)
    print(df)
    # df.to_csv(os.path.join(output_dir, 'results.csv'), index=False) # Salva i risultati in un CSV se necessario

# Esecuzione della funzione di elaborazione
input_dir = 'images/test/people'
output_dir = 'images/valid/people'
process_images(input_dir, output_dir, model)
