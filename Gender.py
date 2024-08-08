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

# Funzione per processare le immagini
def process_images(input_dir, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = ['female', 'male']
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

        print(f"Immagine: {image_name}, Genere Predetto: {gender_category}")
        data["Name"].append(image_name)
        data["Gender"].append(gender_category)

        # Applicare il filtro colorato
        filtered_image = apply_color_filter(image, gender_category)
        filtered_image.save(os.path.join(output_dir, f"filtered_{image_name}"))

    df = pd.DataFrame(data)
    print(df)
    #df.to_csv(os.path.join(output_dir, 'results.csv'), index=False) #per salvare output in un csv

# Esecuzione della funzione di elaborazione
input_dir = 'images/test/people'
output_dir = 'images/valid'
process_images(input_dir, output_dir, model)
