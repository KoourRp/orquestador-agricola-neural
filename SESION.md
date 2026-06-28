# Bitácora del Proyecto - Orquestador Agrícola Neural

> Documento vivo. Se actualiza cada sesión de trabajo.
> Para contexto visual del código, ver `graphify-out/graph.html` (grafo interactivo) y `graphify-out/GRAPH_REPORT.md`.

---

## ¿Qué es este proyecto?

Sistema de diagnóstico de enfermedades en plantas que combina tres capas:

1. **Red neuronal (Python/PyTorch)** — analiza una foto de hoja y dice si está sana, tiene Tizón Tardío (papa) u Oídio (vid)
2. **API REST (FastAPI)** — expone la red neuronal como microservicio local en el puerto 8001
3. **Orquestador (n8n)** — recibe la imagen, llama a la API, consulta el clima (OpenWeatherMap) y pide recomendación de tratamiento a Gemini (Google AI)

El sistema corre todo local (Edge Computing): mínima latencia, sin depender de la nube para el análisis visual.

---

## Archivos clave

| Archivo | Qué hace |
|---|---|
| `src/api_vision.py` | API FastAPI que carga el modelo `.pth` y expone `/predecir_muestra` |
| `src/entrenar_cnn.py` | Script para entrenar la CNN desde cero (genera `modelo_vision.pth`) |
| `modelo_vision.pth` | Pesos del modelo ya entrenado — listo para usar |
| `n8n_workflow_demo.json` | Flujo n8n con formulario web para demo/presentación |
| `n8n_workflow_final.json` | Flujo n8n para producción (recibe imágenes por webhook) |
| `data/` | Dataset de hojas: `Planta_Sana/`, `Tizon_Tardio_Papa/`, `Oidio_Vid/` |
| `graphify-out/graph.html` | Grafo interactivo de la arquitectura del proyecto (estado actual del código) |
| `mapa_flujo_telegram_agricola.html` | Mockup del flujo **objetivo**: bot de Telegram reemplazando el Form Trigger. Referencia de hacia dónde evoluciona el sistema. Ver junto al grafo. |

---

## Cómo levantar el sistema

### 1. Instalar dependencias Python (solo la primera vez)
```bash
pip install fastapi uvicorn torch torchvision pillow python-multipart
```

### 2. Levantar la API de visión
```bash
cd E:\Programming\orquestador-agricola-neural
uvicorn src.api_vision:app --port 8001
```
Cuando aparezca `✅ Modelo PyTorch cargado en memoria desde 'modelo_vision.pth'`, está lista.

### 3. n8n ya está corriendo
El flujo `Demo Final Orquestador (CNN + Clima + LLM)` ya está importado en n8n local (`localhost:5678`).

---

## Configuración del flujo n8n

El flujo tiene 4 nodos en cadena:

```
[Subir Imagen (UI)] → [CNN Inferencia (FastAPI)] → [API Clima (OpenWeather)] → [Agrónomo (Gemini)]
```

### Nodo 1 — Subir Imagen (UI)
- Tipo: Form Trigger
- Genera un formulario web para subir la foto
- El archivo binario sale con el nombre `data`

### Nodo 2 — CNN Inferencia (FastAPI)
- Tipo: HTTP Request → POST a `http://127.0.0.1:8001/predecir_muestra`
- Body: `n8n Binary File`, Name: `file`, Input Data Field Name: `data`
- Devuelve: `{ "status": "success", "diagnostico": "Oidio_Vid", "confianza": 0.9289 }`

### Nodo 3 — API Clima (OpenWeather)
- Tipo: HTTP Request → GET a `https://api.openweathermap.org/data/2.5/weather`
- Authentication: **None**
- Query params: `q=Santiago`, `appid=<API_KEY>`, `units=metric`
- API Key configurada: ✅

### Nodo 4 — Agrónomo (Gemini)
- Tipo: LangChain Chain LLM
- Modelo: `gemini-2.5-flash`
- Source for Prompt: `Define below`
- Prompt configurado con variables de los nodos anteriores:
  - `{{ $('CNN Inferencia (FastAPI)').item.json.diagnostico }}`
  - `{{ $('CNN Inferencia (FastAPI)').item.json.confianza }}`
  - `{{ $('API Clima (OpenWeather)').item.json.main.temp }}`
  - `{{ $('API Clima (OpenWeather)').item.json.weather[0].description }}`
  - `{{ $('API Clima (OpenWeather)').item.json.main.humidity }}`

---

## Credenciales configuradas

| Servicio | Estado |
|---|---|
| OpenWeatherMap API Key | ✅ Configurada en nodo |
| Google Gemini (AI Studio) | ✅ Credencial conectada, modelo gemini-2.5-flash |

---

## Arquitectura objetivo (referencia)

El archivo `mapa_flujo_telegram_agricola.html` (en la raíz del proyecto) muestra la meta de evolución del sistema: **reemplazar el Form Trigger web por un bot de Telegram**.

El flujo objetivo tiene 9 pasos y 3 rutas según la confianza de la CNN:

| Umbral | Ruta |
|---|---|
| ≥ 0.80 | Ruta feliz → diagnóstico + tratamiento completo |
| 0.60 – 0.79 | Ruta con alerta → alerta de baja confianza + recomendación parcial |
| < 0.60 | Ruta climática → solo contexto climático, sin diagnóstico definitivo |

**Referencia cruzada:**
- `graphify-out/graph.html` / `graphify-out/graph.json` → estado **actual** del código
- `mapa_flujo_telegram_agricola.html` → estado **deseado** del sistema

---

## Grafo de conocimiento

Se generó un grafo interactivo del proyecto con `graphify`. Abre `graphify-out/graph.html` en el browser para navegarlo.

**Comunidades detectadas:**

| Comunidad | Qué agrupa |
|---|---|
| FastAPI Inference Service | `api_vision.py`, `AgricolaCNN`, `lifespan`, `predecir_muestra` |
| CNN Training Pipeline | `entrenar_cnn.py`, `generate_mock_dataset`, `train_model` |
| Shared Model Architecture | La clase `AgricolaCNN` compartida entre ambos scripts + docs teóricos |
| n8n Orchestration Layer | Flujo n8n, Gemini, OpenWeatherMap, arquitectura de 3 capas |
| Model Weights & Loading | `modelo_vision.pth`, `MODEL_PATH`, `lifespan` |
| Dataset & Training Loop | `generate_mock_dataset`, `train_model`, PlantVillage dataset |
| Image Preprocessing | `predecir_muestra`, pipeline de transformación de imagen |

**Nodo más central:** `AgricolaCNN (entrenar_cnn)` con 7 conexiones — es el puente entre entrenamiento, inferencia y la capa de orquestación.

---

## Conexión MCP con n8n local

Desde Claude Code se puede ver y disparar el flujo directamente vía MCP (sin abrir la UI de n8n).

- **Workflow ID:** `eiHrG2muhzHk01Ib` = "Demo Final Orquestador (CNN + Clima + LLM)"
- **Qué puede hacer el MCP:** listar workflows, leer webhooks del flujo, disparar webhooks (`call_webhook_post` / `call_webhook_get`)
- **Qué NO puede:** editar nodos (eso sigue siendo manual en la UI)
- **Para iterar rápido:** conviene activar `n8n_workflow_final.json` (tiene webhook, no Form Trigger) y dispararlo con `call_webhook_post`

**Migración pendiente:** el compañero contrató 14 días gratis de n8n cloud. Una vez lista la disertación (en 2 días), migrar el flujo ahí para que quede autofuncionando sin depender del PC local.

---

## Límites de Gemini 2.5 Flash (free tier)

Relevante para la disertación y las pruebas:

| Límite | Valor aproximado (post-recorte dic 2025) |
|---|---|
| Requests por día (RPD) | ~250 |
| Requests por minuto (RPM) | ~10 |
| Reset diario | Medianoche, hora del Pacífico |
| Alcance del límite | Por proyecto de AI Studio, no por API key |

Cada corrida del flujo completo = 1 llamada a Gemini → 250/día es más que suficiente para ensayar. No disparar más de ~10/min. Verificar el número exacto en el panel de AI Studio.

---

## Historial de sesiones

### Sesión 1 — 2026-06-27

**Objetivo:** Entender el proyecto y dejarlo funcionando end-to-end.

**Lo que se hizo:**
- Clonado el repo desde GitHub (`pablocortinez/orquestador-agricola-neural`)
- Leído y explicado el README y la guía teórica
- Importado `n8n_workflow_demo.json` en n8n local (ya estaba hecho al llegar)
- Instaladas dependencias Python: `fastapi uvicorn torch torchvision pillow python-multipart`
- API FastAPI levantada y funcionando en puerto 8001
- Configurada API Key de OpenWeatherMap en el nodo de clima (Authentication → None)
- Configurada credencial de Google Gemini (gemini-2.5-flash)
- Resuelto bug del nodo CNN: campo file con tipo `n8n Binary File`, Name `file`, Input `data`
- Generado grafo interactivo del proyecto (`graphify-out/`) y agregado a `.gitignore`
- Creado este archivo `SESION.md`

**Estado al final de la sesión:** flujo completo configurado, pendiente prueba end-to-end completa con los 4 nodos en cadena.

---

### Sesión 2 — 2026-06-28

**Objetivo:** Continuar documentando el proyecto y preparar la defensa ante el profe.

**Lo que se hizo:**
- Copiado `mapa_flujo_telegram_agricola.html` a la raíz del proyecto y referenciado como archivo inicial de contexto (flujo objetivo: bot Telegram)
- Confirmada la conexión MCP de Claude Code con el n8n local (`localhost:5678`): ve y puede disparar el workflow `eiHrG2muhzHk01Ib`
- Explicación del código Python (`api_vision.py` / `entrenar_cnn.py` / `AgricolaCNN`) entregada de forma conversacional para que el usuario y sus 2 compañeros puedan defenderlo ante el profe
- Verificados límites de Gemini 2.5 Flash free tier (~250 RPD, ~10 RPM) y registrados aquí para referencia
- Actualizado este `SESION.md`: tabla de archivos clave, sección arquitectura objetivo, sección MCP n8n, sección límites Gemini
- Primer commit local con todos los cambios de las dos sesiones (sin push al remoto)

**Estado al final de la sesión:** documentación completa, MCP conectado a n8n, pendiente prueba end-to-end vía webhook y planificación de la migración a n8n cloud del compañero.

---

*Añadir una entrada en "Historial de sesiones" cada vez que se trabaje en el proyecto.*
