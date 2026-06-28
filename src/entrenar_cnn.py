"""
Fase 1: Entrenamiento de Red Neuronal Convolucional (PyTorch)
Proyecto Final - Orquestador Agrícola

Este script implementa y entrena una CNN (Red Neuronal Convolucional) para clasificar
enfermedades agrícolas, reemplazando el perceptrón multicapa (MLP) hecho en NumPy.

Uso:
1. Colocar las imágenes reales del dataset (ej. PlantVillage) en la carpeta "data/".
2. Ejecutar este script para entrenar la red.
3. Se generará un archivo "modelo_vision.pth" listo para ser consumido por FastAPI.

=============================================================================
GLOSA DE COMPONENTES DE DEEP LEARNING PRESENTES EN ESTE ARCHIVO
=============================================================================
- TENSORES: Estructuras multidimensionales (torch.Tensor) que representan imágenes
  y pesos. Flujo: PIL Image → transforms → Tensor [B, C, H, W].
- CNN: Dos bloques convolucionales (Conv2d + ReLU + MaxPool2d) para extracción
  jerárquica de características visuales (bordes → texturas → patrones de enfermedad).
- MLP: Cabezal clasificador Fully Connected (fc1 → ReLU → fc2) que mapea features
  espaciales aplanadas a probabilidades por clase.
- BACKPROPAGATION: loss.backward() calcula ∂L/∂w para cada peso; optimizer.step()
  actualiza los pesos usando Adam.
- LOSS FUNCTION: CrossEntropyLoss = Softmax + NLLLoss, ideal para clasificación
  multiclase (3 patologías agrícolas).
- ÉPOCAS Y BATCHES: Bucle externo = épocas (pasadas completas por el dataset);
  bucle interno = mini-batches de 32 muestras.
=============================================================================
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

# ─── HIPERPARÁMETRO: DATA_DIR ───
# Ruta al dataset organizado en subdirectorios por clase (formato ImageFolder).
DATA_DIR = "data"

# ─── HIPERPARÁMETRO: CLASSES ───
# Lista de las 3 clases objetivo. Define num_classes=3 para la capa de salida.
# Contexto agrícola: cubre planta sana + 2 enfermedades foliares comunes en Chile.
CLASSES = ["Planta_Sana", "Tizon_Tardio_Papa", "Oidio_Vid"]

# ─── HIPERPARÁMETRO: IMG_SIZE = 64 ───
# Dimensión espacial a la que se redimensionan todas las imágenes (64×64 píxeles).
# Justificación: 64px es un compromiso eficiente entre resolución suficiente para
# detectar patrones foliares y costo computacional bajo para entrenamiento en CPU.
# Valores menores (32) pierden detalle; mayores (224) requieren GPU y más RAM.
IMG_SIZE = 64

# ─── HIPERPARÁMETRO: BATCH_SIZE = 32 ───
# Número de muestras procesadas antes de cada actualización de pesos.
# Justificación: 32 es el estándar empírico (Bengio, 2012). Batches menores (8-16)
# producen gradientes ruidosos; mayores (128+) requieren más memoria y pueden
# converger a mínimos más agudos (peor generalización).
BATCH_SIZE = 32

# ─── HIPERPARÁMETRO: EPOCHS = 10 ───
# Número de pasadas completas por todo el dataset durante el entrenamiento.
# Justificación: 10 épocas es conservador. Con ~2000 imágenes y batch=32 (~63
# iteraciones/época), el modelo tiene 630 actualizaciones totales de pesos.
# Suficiente para convergencia en un dataset pequeño sin riesgo severo de
# overfitting. Si el loss no converge, incrementar a 20-30.
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

# ─── PIPELINE DE TRANSFORMACIÓN DE TENSORES ───
# Convierte imágenes PIL (H, W, C) en tensores PyTorch (C, H, W) normalizados.
# Cada transformación modifica el tensor secuencialmente:
#   1. Resize: Escala la imagen a (IMG_SIZE, IMG_SIZE) = (64, 64) píxeles.
#   2. ToTensor: Convierte PIL Image [H,W,C] uint8 [0,255] → Tensor [C,H,W] float32 [0,1].
#      Shape resultante: [3, 64, 64] (3 canales RGB).
#   3. Normalize: Aplica z = (x - μ) / σ por canal con μ=0.5, σ=0.5.
#      Reescala de [0,1] a [-1,1]. Centra la distribución en 0, facilitando
#      la convergencia del optimizador (gradientes más estables).
# ─── HIPERPARÁMETRO: Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ───
# Normalización genérica. Para mayor precisión se usarían las medias/desviaciones
# reales del dataset PlantVillage o las de ImageNet (0.485, 0.456, 0.406).
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar dataset usando ImageFolder de torchvision.
# ImageFolder espera subdirectorios por clase: data/Oidio_Vid/, data/Planta_Sana/, etc.
# NOTA CRÍTICA: ImageFolder asigna etiquetas en ORDEN ALFABÉTICO de los subdirectorios.
# Orden resultante: 0=Oidio_Vid, 1=Planta_Sana, 2=Tizon_Tardio_Papa.
# Este orden DEBE coincidir con CLASS_NAMES en api_vision.py para que la predicción sea correcta.
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# ─── DATALOADER: Segmentación en mini-batches ───
# Envuelve el dataset en un iterador que entrega tensores de shape [B, C, H, W].
# Con batch_size=32: cada iteración entrega un tensor [32, 3, 64, 64].
# shuffle=True: baraja los datos cada época para evitar que el modelo memorice
# el orden de las muestras (reduce overfitting y mejora generalización).
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------------------------
# 2. DEFINICIÓN DE LA ARQUITECTURA CNN
# -----------------------------------------------------------------------------

class AgricolaCNN(nn.Module):
    """
    Red Neuronal Convolucional para clasificación de patologías foliares agrícolas.

    ARQUITECTURA (topografía de la red):
    ┌─────────────────────────────────────────────────────────────────────┐
    │  INPUT: Tensor [B, 3, 64, 64]  (batch de imágenes RGB 64×64)      │
    │                                                                     │
    │  ► BLOQUE CONV 1: Conv2d(3→16, k=3, p=1) → ReLU → MaxPool(2×2)   │
    │    Shape: [B,3,64,64] → [B,16,64,64] → [B,16,64,64] → [B,16,32,32]│
    │                                                                     │
    │  ► BLOQUE CONV 2: Conv2d(16→32, k=3, p=1) → ReLU → MaxPool(2×2)  │
    │    Shape: [B,16,32,32] → [B,32,32,32] → [B,32,32,32] → [B,32,16,16]│
    │                                                                     │
    │  ► FLATTEN: view(B, -1)                                             │
    │    Shape: [B, 32, 16, 16] → [B, 8192]                              │
    │                                                                     │
    │  ► MLP (Clasificador Fully Connected):                              │
    │    Linear(8192→64) → ReLU → Linear(64→3)                           │
    │    Shape: [B, 8192] → [B, 64] → [B, 64] → [B, 3]                  │
    │                                                                     │
    │  OUTPUT: Tensor [B, 3]  (logits crudos, una puntuación por clase)   │
    └─────────────────────────────────────────────────────────────────────┘

    PARÁMETROS ENTRENABLES TOTALES:
    - conv1: (3×16×3×3) + 16 bias = 448
    - conv2: (16×32×3×3) + 32 bias = 4,640
    - fc1: (8192×64) + 64 bias = 524,352
    - fc2: (64×3) + 3 bias = 195
    - TOTAL: ~529,635 parámetros
    """

    def __init__(self, num_classes=3):
        super(AgricolaCNN, self).__init__()

        # ─── BLOQUE CONVOLUCIONAL 1 (Extracción de características de bajo nivel) ───
        # Conv2d: Aplica 16 filtros de 3×3 sobre la imagen RGB (3 canales).
        # ─── HIPERPARÁMETRO: out_channels=16 ───
        #   16 filtros iniciales es estándar para datasets pequeños. Cada filtro
        #   aprende a detectar un patrón visual elemental (bordes, gradientes de color).
        # ─── HIPERPARÁMETRO: kernel_size=3 ───
        #   Kernel 3×3 es el estándar mínimo efectivo (VGGNet). Captura patrones
        #   locales con campo receptivo pequeño y menor costo que 5×5 o 7×7.
        # ─── HIPERPARÁMETRO: padding=1 ───
        #   Padding "same": preserva dimensiones espaciales (64×64 → 64×64).
        #   Fórmula: out = (in + 2*pad - kernel) / stride + 1 = (64+2-3)/1+1 = 64.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # ─── FUNCIÓN DE ACTIVACIÓN: ReLU (Rectified Linear Unit) ───
        # f(x) = max(0, x). Introduce no-linealidad permitiendo aprender funciones complejas.
        # Ventaja sobre Sigmoid/Tanh: NO sufre "vanishing gradient" porque su derivada
        # es 1 para x>0 (gradiente constante), permitiendo entrenar redes más profundas.
        # Es el estándar de facto desde AlexNet (2012) por su simplicidad y eficiencia.
        self.relu1 = nn.ReLU()

        # ─── POOLING: MaxPool2d(2,2) ───
        # Reduce dimensiones espaciales a la mitad: 64×64 → 32×32.
        # Selecciona el valor máximo en cada ventana 2×2, preservando las activaciones
        # más fuertes (features más relevantes). Aporta invarianza a pequeñas traslaciones
        # y reduce el número de parámetros en las capas siguientes.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ─── BLOQUE CONVOLUCIONAL 2 (Extracción de características de alto nivel) ───
        # ─── HIPERPARÁMETRO: out_channels=32 ───
        #   Duplicar filtros (16→32) es patrón estándar en CNNs. Las capas más profundas
        #   necesitan más filtros para representar combinaciones más complejas de features
        #   (ej: manchas de Tizón = bordes marrones + textura irregular + distribución).
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # MaxPool2d: 32×32 → 16×16. Segundo nivel de reducción espacial.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ─── CLASIFICADOR MLP (Perceptrón Multicapa / Fully Connected) ───
        # Después de 2 bloques conv+pool, el tensor tiene shape [B, 32, 16, 16].
        # Se aplana (flatten) a [B, 32*16*16] = [B, 8192] para alimentar capas lineales.
        #
        # ─── HIPERPARÁMETRO: hidden_layer_sizes = (64,) ───
        # fc1 comprime 8192 features → 64 neuronas. Esta capa oculta actúa como
        # "cuello de botella" que fuerza al modelo a aprender una representación
        # compacta y discriminativa. 64 neuronas ofrecen capacidad suficiente para
        # separar 3 clases sin sobreajustar en un dataset de ~2000 imágenes.
        # Dimensión calculada: IMG_SIZE=64 → pool1(32) → pool2(16), entonces 32*16*16=8192.
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.relu3 = nn.ReLU()

        # ─── CAPA DE SALIDA ───
        # fc2 produce 3 logits crudos (uno por clase). No tiene activación porque
        # CrossEntropyLoss aplica Softmax internamente. Cada logit representa la
        # "puntuación" bruta de confianza para cada patología.
        # ─── HIPERPARÁMETRO: num_classes=3 ───
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        """
        Forward pass: propagación hacia adelante del tensor a través de la red.

        Args:
            x: Tensor de entrada [B, 3, 64, 64] (batch de imágenes normalizadas).

        Returns:
            Tensor [B, 3] con logits crudos (sin Softmax) para cada clase.

        Flujo de transformación del tensor:
            [B, 3, 64, 64] → conv1 → [B, 16, 64, 64] → relu → pool → [B, 16, 32, 32]
            → conv2 → [B, 32, 32, 32] → relu → pool → [B, 32, 16, 16]
            → flatten → [B, 8192] → fc1 → [B, 64] → relu → fc2 → [B, 3]
        """
        # Bloque 1: Convolución → Activación no-lineal → Submuestreo
        x = self.pool1(self.relu1(self.conv1(x)))
        # Bloque 2: Misma secuencia, features más abstractas
        x = self.pool2(self.relu2(self.conv2(x)))

        # ─── OPERACIÓN DE TENSOR: FLATTEN (Aplanado) ───
        # x.view(x.size(0), -1) transforma [B, 32, 16, 16] → [B, 8192].
        # x.size(0) preserva la dimensión del batch. -1 calcula automáticamente
        # el producto de las dimensiones restantes (32×16×16 = 8192).
        # Esta operación es el PUENTE entre la CNN (opera sobre grillas 2D)
        # y el MLP (opera sobre vectores 1D).
        x = x.view(x.size(0), -1) # Flatten

        # MLP: Compresión a representación de 64 dims + activación
        x = self.relu3(self.fc1(x))
        # Capa de salida: 64 → 3 logits (sin activación, CrossEntropyLoss la aplica)
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO
# -----------------------------------------------------------------------------
def train_model():
    """
    Ejecuta el bucle completo de entrenamiento de la CNN.

    Estructura del entrenamiento:
    ┌─ Época (BUCLE EXTERNO): for epoch in range(EPOCHS)
    │   Una época = 1 pasada completa por TODAS las muestras del dataset.
    │   Con ~2000 imágenes y batch_size=32, cada época tiene ~63 iteraciones.
    │
    │   ┌─ Batch/Lote (BUCLE INTERNO): for inputs, labels in dataloader
    │   │   Un batch = subconjunto de 32 muestras procesadas en paralelo.
    │   │   Cada iteración del batch ejecuta: forward → loss → backward → update.
    │   └──────────────────────────────────────────────────────────────────
    └──────────────────────────────────────────────────────────────────────
    """
    print(f"\nIniciando entrenamiento por {EPOCHS} épocas...")

    # Instanciar el modelo con 3 neuronas de salida (una por clase agrícola)
    model = AgricolaCNN(num_classes=len(CLASSES))

    # ─── FUNCIÓN DE PÉRDIDA: CrossEntropyLoss ───
    # Combina LogSoftmax + NLLLoss (Negative Log-Likelihood).
    # Fórmula: L = -log(P(clase_correcta)), donde P = softmax(logits).
    # Ideal para clasificación multiclase (3 patologías). Penaliza más cuando
    # el modelo asigna BAJA probabilidad a la clase verdadera.
    # En contexto agrícola: un diagnóstico equivocado de "Planta_Sana" cuando hay
    # Tizón produciría una pérdida alta, forzando al modelo a ser cauteloso.
    criterion = nn.CrossEntropyLoss()

    # ─── HIPERPARÁMETRO: Optimizador Adam con lr=0.001 ───
    # Adam (Adaptive Moment Estimation, Kingma & Ba 2014) combina:
    #   - Momentum (promedio móvil del gradiente, β1=0.9)
    #   - RMSProp (promedio móvil del gradiente², β2=0.999)
    # Ventajas sobre SGD puro:
    #   1. Learning rate adaptativo POR PARÁMETRO → converge más rápido.
    #   2. Robusto a gradientes ruidosos y sparse → ideal para imágenes.
    #   3. Menos sensible a la elección del learning rate inicial.
    # lr=0.001 es el valor por defecto recomendado por los autores y funciona
    # bien como punto de partida para la mayoría de problemas de visión.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # model.train() activa modo entrenamiento (habilita Dropout y BatchNorm
    # si existieran; aquí es buena práctica aunque no hay Dropout).
    model.train()

    # ─── BUCLE DE ÉPOCAS (EXTERNO) ───
    # Cada época recorre TODO el dataset una vez completa.
    for epoch in range(EPOCHS):
        running_loss = 0.0

        # ─── BUCLE DE BATCHES (INTERNO) ───
        # El DataLoader entrega mini-batches de 32 muestras.
        # inputs: Tensor [32, 3, 64, 64] (imágenes), labels: Tensor [32] (etiquetas).
        for i, (inputs, labels) in enumerate(dataloader):

            # ─── PASO 1: ZERO_GRAD ───
            # Reinicia los gradientes acumulados a cero. PyTorch ACUMULA gradientes
            # por defecto (útil para gradient accumulation), pero en el flujo estándar
            # se deben limpiar antes de cada paso para evitar gradientes incorrectos.
            optimizer.zero_grad()      # Reiniciar gradientes

            # ─── PASO 2: FORWARD PASS ───
            # Propaga el tensor de entrada a través de toda la red (conv→pool→fc).
            # outputs: Tensor [32, 3] con logits crudos para cada clase.
            # PyTorch construye el grafo computacional dinámicamente durante este paso,
            # registrando cada operación para poder calcular gradientes después.
            outputs = model(inputs)    # Forward pass

            # ─── PASO 3: CÁLCULO DE LA PÉRDIDA ───
            # Compara las predicciones (outputs) con las etiquetas reales (labels).
            # loss: Tensor escalar (0-dimensional) con el error promedio del batch.
            loss = criterion(outputs, labels) # Calcular error

            # ─── PASO 4: BACKPROPAGATION (Retropropagación del gradiente) ───
            # loss.backward() ejecuta el algoritmo de retropropagación:
            #   1. Recorre el grafo computacional en orden INVERSO (de loss hacia inputs).
            #   2. Aplica la REGLA DE LA CADENA: ∂L/∂w = ∂L/∂y · ∂y/∂w para cada peso.
            #   3. Almacena el gradiente en el atributo .grad de cada parámetro.
            # Los gradientes fluyen: fc2 ← relu3 ← fc1 ← flatten ← pool2 ← relu2 ← conv2 ← pool1 ← relu1 ← conv1
            # Gracias a ReLU (derivada=1 para x>0), los gradientes NO se desvanecen
            # al propagarse hacia las capas iniciales.
            loss.backward()            # Backpropagation

            # ─── PASO 5: ACTUALIZACIÓN DE PESOS ───
            # optimizer.step() aplica la regla de actualización de Adam:
            #   w_new = w_old - lr * m̂/(√v̂ + ε)
            # donde m̂ y v̂ son los momentos corregidos del gradiente.
            # Modifica in-place los ~529,635 parámetros de la red.
            optimizer.step()           # Actualizar pesos
            
            running_loss += loss.item()
            
        # Log por época: muestra el loss promedio (running_loss / num_batches).
        # Un loss decreciente indica convergencia. Si se estanca, considerar
        # ajustar lr, aumentar épocas, o agregar data augmentation.
        print(f"Época [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(dataloader):.4f}")
        
    print("Entrenamiento completado.")
    
    # 4. EXPORTACIÓN DEL MODELO
    # torch.save serializa el state_dict (diccionario de pesos) a disco.
    # Solo guarda los PESOS, no la arquitectura. Para cargarlos en api_vision.py
    # se necesita redefinir la misma clase AgricolaCNN y llamar model.load_state_dict().
    export_path = "modelo_vision.pth"
    torch.save(model.state_dict(), export_path)
    print(f"\n✅ Modelo exportado exitosamente a '{export_path}'")
    print("Este archivo será cargado por FastAPI en la Fase 2.")

if __name__ == "__main__":
    train_model()
