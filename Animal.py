from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import torchvision.models as models
import random
from PIL import ImageOps
import shutil

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

# Funzione per processare le immagini
def process_images(input_dir, output_dir, model):
    categories = ['bird', 'cat', 'dog']
    
    # Elimina solo le cartelle specifiche per gli animali
    clear_output_directory(output_dir, categories)
    
    create_category_folders(output_dir, categories)

    image_names = os.listdir(input_dir)
    print("Ordine originale delle immagini:")
    print(image_names)

    random.shuffle(image_names)
    print("Ordine dopo shuffle:")
    print(image_names)

    for image_name in image_names:
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        # Debug: Verifica il numero di canali
        print(f"Numero di canali prima della trasformazione: {len(image.getbands())}")

        image_tensor = transform(image).unsqueeze(0)

        # Classificare l'immagine
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            animal_category = categories[predicted.item()]

        # Stampa per verificare la categoria predetta
        print(f"Immagine: {image_name}, Categoria Predetta: {animal_category}")

        # Applicare il filtro appropriato
        if animal_category == 'dog':
            print("Applico filtro cane.")
            modified_image = add_dog_filter(image)
        elif animal_category == 'cat':
            print("Applico filtro gatto.")
            modified_image = add_cat_filter(image)
        elif animal_category == 'bird':
            print("Applico filtro uccello.")
            modified_image = add_bird_filter(image)
        else:
            print("Nessun filtro applicato.")
            modified_image = image  # Nessun filtro applicato

        # Salvare l'immagine modificata
        category_folder = os.path.join(output_dir, animal_category)
        modified_image_path = os.path.join(category_folder, image_name)
        
        modified_image.save(modified_image_path)

# Esecuzione della funzione di elaborazione
input_dir = 'images/test/animals'
output_dir = 'images/valid/animals'

process_images(input_dir, output_dir, model)
