"""
Fase 2: Backend FastAPI para Inferencia con PyTorch
Proyecto Final - Orquestador Agrícola

Este script carga el modelo "modelo_vision.pth" entrenado en la Fase 1
y expone un endpoint POST para recibir imágenes reales y devolver el diagnóstico.
"""

import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- 1. IMPORTAR LA ARQUITECTURA DE LA FASE 1 ---
# Para cargar los pesos, necesitamos la misma estructura de la clase
class AgricolaCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AgricolaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

CLASS_NAMES = ["Oidio_Vid", "Planta_Sana", "Tizon_Tardio_Papa"]
MODEL_PATH = "modelo_vision.pth"
ACTIVE_MODEL = None

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo PyTorch en memoria antes de aceptar peticiones."""
    global ACTIVE_MODEL
    try:
        model = AgricolaCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval() # Modo inferencia
        ACTIVE_MODEL = model
        print(f"✅ Modelo PyTorch cargado en memoria desde '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró '{MODEL_PATH}'. ¡Debes correr la Fase 1 primero!")
        ACTIVE_MODEL = None
    yield
    print("Apagando API de Visión...")

app = FastAPI(
    title="API de Visión Agrícola (PyTorch)",
    description="Microservicio de diagnóstico foliar",
    lifespan=lifespan
)

@app.post("/predecir_muestra")
async def predecir_muestra(file: UploadFile = File(...)):
    """
    Recibe una imagen (JPG/PNG), la procesa y devuelve la predicción.
    Endpoint optimizado para ser consumido por el nodo HTTP de n8n.
    """
    if ACTIVE_MODEL is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
        
    try:
        # Leer imagen desde los bytes subidos
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocesamiento a Tensor
        input_tensor = transform(image).unsqueeze(0) # Añadir dimensión de batch
        
        # Inferencia sin calcular gradientes
        with torch.no_grad():
            outputs = ACTIVE_MODEL(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Obtener clase con mayor probabilidad
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        diagnostico = CLASS_NAMES[predicted_idx.item()]
        confianza_float = round(confidence.item(), 4)
        
        return JSONResponse(content={
            "status": "success",
            "diagnostico": diagnostico,
            "confianza": confianza_float
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")
