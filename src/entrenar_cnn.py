"""
Fase 1: Entrenamiento de Red Neuronal Convolucional (PyTorch)
Proyecto Final - Orquestador Agrícola

Este script implementa y entrena una CNN (Red Neuronal Convolucional) para clasificar
enfermedades agrícolas, reemplazando el perceptrón multicapa (MLP) hecho en NumPy.

Uso:
1. Colocar las imágenes reales del dataset (ej. PlantVillage) en la carpeta "data/".
2. Ejecutar este script para entrenar la red.
3. Se generará un archivo "modelo_vision.pth" listo para ser consumido por FastAPI.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PREPARACIÓN DE DATOS (MOCK SI NO HAY DATOS REALES)
# -----------------------------------------------------------------------------
DATA_DIR = "data"
CLASSES = ["Planta_Sana", "Tizon_Tardio_Papa", "Oidio_Vid"]
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 10

def generate_mock_dataset():
    """Genera imágenes falsas (ruido de colores) si no existe el dataset real.
    Esto permite que el script se ejecute y compile el flujo completo sin errores."""
    print("Verificando dataset...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print("Dataset no encontrado. Generando imágenes de prueba (MOCK)...")
        for class_name in CLASSES:
            class_dir = os.path.join(DATA_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            # Generar 20 imágenes por clase
            for i in range(20):
                # Planta_Sana (Más verde), Tizon (Marrón), Oidio (Blanquecino)
                if class_name == "Planta_Sana":
                    color = (10, random.randint(150, 255), 10)
                elif class_name == "Tizon_Tardio_Papa":
                    color = (random.randint(100, 150), 70, 20)
                else:
                    color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
                
                img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=color)
                img.save(os.path.join(class_dir, f"mock_{i}.jpg"))
        print("Dataset MOCK generado. ¡Para el proyecto final, reemplaza la carpeta 'data/' con fotos reales!")
    else:
        print("Dataset encontrado en 'data/'.")

import random
generate_mock_dataset()

# Transformaciones (Resize, Tensor, Normalización básica)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar dataset usando ImageFolder de torchvision
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------------------------
# 2. DEFINICIÓN DE LA ARQUITECTURA CNN
# -----------------------------------------------------------------------------
class AgricolaCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AgricolaCNN, self).__init__()
        # Bloque Convolucional 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloque Convolucional 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Clasificador (Fully Connected)
        # IMG_SIZE=64 -> pool1(32) -> pool2(16)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO
# -----------------------------------------------------------------------------
def train_model():
    print(f"\nIniciando entrenamiento por {EPOCHS} épocas...")
    model = AgricolaCNN(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()      # Reiniciar gradientes
            outputs = model(inputs)    # Forward pass
            loss = criterion(outputs, labels) # Calcular error
            loss.backward()            # Backpropagation
            optimizer.step()           # Actualizar pesos
            
            running_loss += loss.item()
            
        print(f"Época [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(dataloader):.4f}")
        
    print("Entrenamiento completado.")
    
    # 4. EXPORTACIÓN DEL MODELO
    export_path = "modelo_vision.pth"
    torch.save(model.state_dict(), export_path)
    print(f"\n✅ Modelo exportado exitosamente a '{export_path}'")
    print("Este archivo será cargado por FastAPI en la Fase 2.")

if __name__ == "__main__":
    train_model()
