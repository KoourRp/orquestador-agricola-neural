"""
Fase 2: Backend FastAPI para Inferencia con PyTorch
Proyecto Final - Orquestador Agrícola

Este script carga el modelo "modelo_vision.pth" entrenado en la Fase 1
y expone un endpoint POST para recibir imágenes reales y devolver el diagnóstico.

=============================================================================
GLOSA DE COMPONENTES DE DEEP LEARNING PRESENTES EN ESTE ARCHIVO
=============================================================================
- TENSORES: La imagen sube como bytes → se convierte a PIL → se transforma en
  Tensor [1, 3, 64, 64] (batch de 1). unsqueeze(0) añade la dimensión de batch.
- CNN: Réplica exacta de la arquitectura definida en entrenar_cnn.py. DEBE ser
  idéntica para que load_state_dict() pueda mapear los pesos correctamente.
- INFERENCIA: torch.no_grad() desactiva el grafo de gradientes. Reduce consumo
  de memoria ~50% y acelera el forward pass (no necesita backprop en producción).
- SOFTMAX: Convierte logits crudos [B,3] en probabilidades [0,1] que suman 1.
  Permite interpretar la salida como "confianza" del diagnóstico.
=============================================================================
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
# Para cargar los pesos, necesitamos la misma estructura de la clase.
# NOTA CRÍTICA: Esta clase DEBE ser una réplica EXACTA de la definida en
# entrenar_cnn.py. Si se modifica la arquitectura allá (ej: agregar una capa),
# se debe replicar aquí ANTES de cargar el .pth, o load_state_dict() fallará
# con un error de "unexpected key" o "missing key".

class AgricolaCNN(nn.Module):
    """
    Réplica de la arquitectura CNN de entrenar_cnn.py para inferencia.
    Ver docstring completa en entrenar_cnn.py para detalles de cada capa.

    Arquitectura resumida:
        Conv2d(3→16, 3×3) → ReLU → MaxPool(2×2)
        Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)
        Flatten → Linear(8192→64) → ReLU → Linear(64→3)
    """

    def __init__(self, num_classes=3):
        super(AgricolaCNN, self).__init__()

        # ─── BLOQUE CONVOLUCIONAL 1 ───
        # Conv2d(3→16, kernel=3, padding=1): 16 filtros sobre 3 canales RGB.
        # Shape: [B,3,64,64] → [B,16,64,64] → ReLU → MaxPool → [B,16,32,32]
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # ─── BLOQUE CONVOLUCIONAL 2 ───
        # Conv2d(16→32, kernel=3, padding=1): 32 filtros sobre 16 feature maps.
        # Shape: [B,16,32,32] → [B,32,32,32] → ReLU → MaxPool → [B,32,16,16]
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # ─── CLASIFICADOR MLP (Fully Connected) ───
        # Flatten: [B,32,16,16] → [B,8192]. fc1: 8192→64. fc2: 64→3.
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """
        Forward pass para inferencia.

        Flujo del tensor:
            [B,3,64,64] → conv1+relu+pool → [B,16,32,32]
            → conv2+relu+pool → [B,32,16,16]
            → flatten → [B,8192] → fc1+relu → [B,64] → fc2 → [B,3]
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # ─── OPERACIÓN DE TENSOR: FLATTEN ───
        # x.view(x.size(0), -1): [B,32,16,16] → [B,8192]
        # Puente entre CNN (2D espacial) y MLP (vector 1D).
        x = x.view(x.size(0), -1)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ─── CLASS_NAMES: Mapeo índice → nombre de clase ───
# NOTA CRÍTICA: Este orden DEBE coincidir con el orden ALFABÉTICO que
# torchvision.datasets.ImageFolder asigna durante el entrenamiento:
#   Directorio "Oidio_Vid"       → índice 0
#   Directorio "Planta_Sana"     → índice 1
#   Directorio "Tizon_Tardio_Papa" → índice 2
# Si este orden no coincide, las predicciones se mostrarán con etiquetas cruzadas
# (bug que ya fue corregido en la Sesión 2, ver SESION.md).
CLASS_NAMES = [
    "Arana_Roja_Tomate",
    "Mancha_Bact_Pimiento",
    "Mancha_Bact_Tomate",
    "Mancha_Diana_Tomate",
    "Moho_Foliar_Tomate",
    "Mosaico_Tomate",
    "Oidio_Vid",
    "Planta_Sana",
    "Septoria_Tomate",
    "Tizon_Tardio_Papa",
    "Tizon_Tardio_Tomate",
    "Tizon_Temprano_Papa",
    "Tizon_Temprano_Tomate",
    "Virus_Rizo_Tomate",
]

# ─── HIPERPARÁMETRO: MODEL_PATH ───
# Ruta al archivo de pesos serializados (.pth) exportado por entrenar_cnn.py.
MODEL_PATH = "modelo_vision.pth"

# Variable global que mantiene el modelo cargado en RAM durante toda la vida
# del servidor. Patrón "Singleton" para evitar recargar el modelo en cada request.
ACTIVE_MODEL = None

# ─── PIPELINE DE PREPROCESAMIENTO (debe ser IDÉNTICO al de entrenamiento) ───
# Si se cambia el tamaño, normalización o cualquier transformación aquí pero no
# en entrenar_cnn.py (o viceversa), el modelo recibirá datos con distribución
# diferente a la que aprendió, degradando severamente la precisión.
# Flujo: PIL Image → Resize(64×64) → ToTensor [3,64,64] float32 → Normalize [-1,1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo PyTorch en memoria antes de aceptar peticiones.

    Patrón Lifespan de FastAPI (reemplaza @app.on_event("startup")):
    - Al arrancar: carga los pesos del .pth en una instancia de AgricolaCNN.
    - model.eval(): Activa modo evaluación. Desactiva Dropout y fija BatchNorm
      (aunque esta red no los usa, es buena práctica obligatoria en producción).
    - Al apagar: libera recursos (yield finaliza el contexto).
    """
    global ACTIVE_MODEL
    try:
        model = AgricolaCNN(num_classes=len(CLASS_NAMES))

        # ─── CARGA DE PESOS ───
        # torch.load deserializa el state_dict desde disco.
        # load_state_dict() mapea cada tensor de pesos al parámetro correspondiente
        # de la red (conv1.weight, conv1.bias, fc1.weight, etc.).
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))

        # ─── model.eval(): MODO INFERENCIA ───
        # Diferencias con model.train():
        #   - Dropout: Se DESACTIVA (todas las neuronas participan).
        #   - BatchNorm: Usa estadísticas GLOBALES en vez de las del mini-batch.
        # Sin eval(), las predicciones serían no deterministas y degradadas.
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

    Pipeline de inferencia completo:
        bytes → PIL Image RGB → Tensor [3,64,64] → unsqueeze → [1,3,64,64]
        → forward pass (CNN) → logits [1,3] → softmax → probabilidades [1,3]
        → argmax → índice de clase → CLASS_NAMES[idx] → JSON response
    """
    if ACTIVE_MODEL is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")
        
    try:
        # ─── PASO 1: Lectura de bytes crudos ───
        # FastAPI recibe el archivo como stream binario multipart/form-data.
        contents = await file.read()

        # ─── PASO 2: Conversión a PIL Image ───
        # io.BytesIO envuelve los bytes en un file-like object.
        # .convert("RGB") asegura 3 canales (descarta alfa si es PNG con transparencia).
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # ─── PASO 3: Transformación a Tensor ───
        # transform(image): PIL [H,W,3] → Tensor [3,64,64] (float32, rango [-1,1]).
        # .unsqueeze(0): Añade dimensión de batch → [1,3,64,64].
        # La red SIEMPRE espera un tensor 4D [B,C,H,W], incluso para una sola imagen.
        input_tensor = transform(image).unsqueeze(0) # Añadir dimensión de batch
        
        # ─── PASO 4: INFERENCIA SIN GRADIENTES ───
        # torch.no_grad() desactiva el tracking de gradientes (autograd).
        # Beneficios en producción:
        #   1. ~50% menos uso de memoria (no almacena activaciones intermedias).
        #   2. Forward pass más rápido (no construye grafo computacional).
        #   3. No tiene sentido calcular gradientes si no hay backpropagation.
        with torch.no_grad():
            # Forward pass: [1,3,64,64] → CNN → [1,3] logits
            outputs = ACTIVE_MODEL(input_tensor)

            # ─── SOFTMAX: Logits → Probabilidades ───
            # Fórmula: P(clase_i) = e^(z_i) / Σ e^(z_j)
            # Transforma logits crudos (pueden ser negativos o >1) en probabilidades
            # válidas que suman 1.0. Ejemplo: [-0.5, 2.1, 0.3] → [0.05, 0.72, 0.23]
            # dim=1: aplica softmax sobre la dimensión de clases (no sobre el batch).
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # ─── ARGMAX: Selección de la clase predicha ───
            # torch.max retorna (valor_máximo, índice) sobre dim=1.
            # confidence: probabilidad de la clase ganadora (ej: 0.9289).
            # predicted_idx: índice de la clase (0, 1 o 2).
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        # ─── PASO 5: Mapeo índice → etiqueta legible ───
        # .item() extrae el valor escalar del tensor 0-dimensional.
        diagnostico = CLASS_NAMES[predicted_idx.item()]
        confianza_float = round(confidence.item(), 4)
        
        # Respuesta JSON consumida por el nodo HTTP de n8n.
        # n8n usa diagnostico y confianza para alimentar los nodos de Clima y Gemini.
        return JSONResponse(content={
            "status": "success",
            "diagnostico": diagnostico,
            "confianza": confianza_float
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")
