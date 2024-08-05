from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import torchvision.models as models

# Definizione del modello
class AnimalNetwork(nn.Module):
    def __init__(self):
        super(AnimalNetwork, self).__init__()
        self.model = models.resnet18(weights=None)
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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Definizione dei filtri
def add_dog_filter(image):
    return image.transpose(Image.FLIP_TOP_BOTTOM)  # Ruota l'immagine sottosopra

def add_cat_filter(image):
    return image.convert('L').convert('RGB')  # Converti in bianco e nero

def add_bird_filter(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)  # Riflette orizzontalmente

# Funzione per processare le immagini
def process_images(input_dir, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = ['dog', 'cat', 'bird']

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path)#.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Classificare l'immagine
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            animal_category = categories[predicted.item()]

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
        modified_image_path = os.path.join(output_dir, image_name)
        modified_image.save(modified_image_path)

# Esecuzione della funzione di elaborazione
input_dir = 'images/test'
output_dir = 'images/valid'

process_images(input_dir, output_dir, model)
