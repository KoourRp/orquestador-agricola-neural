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
| `n8n_workflow_demo.json` | Flujo n8n con formulario web para demo/presentación (Form Trigger) |
| `n8n_workflow_telegram.json` | **Flujo activo**: bot Telegram `@FitoScanAIBot` → CNN → Clima → Gemini → respuesta |
| `n8n_workflow_final.json` | Flujo n8n para producción (recibe imágenes por webhook) |
| `data/` | Dataset PlantVillage: 1000 Oidio_Vid / 1000 Tizon / 152 Planta_Sana (desbalanceado) |
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

### 3. Exponer FastAPI con ngrok (necesario si n8n corre en la nube)
Si n8n está en cloud y FastAPI corre local, ngrok expone el puerto 8001 para que n8n pueda llamarlo:
```bash
ngrok http 8001
# o con dominio estático (recomendado):
ngrok http --domain=tu-subdominio.ngrok-free.app 8001
```
Actualizar la URL del nodo "CNN Inferencia (FastAPI)" en n8n con la URL que entrega ngrok.

Si n8n también corre local, no se necesita ngrok para FastAPI.

### 4. n8n ya está corriendo
Los flujos `Demo Final Orquestador` y `Orquestador Agrícola - Bot Telegram (@FitoScanAIBot)` están importados en n8n local (`localhost:5678`).
- Workflow demo (Form Trigger): ID `eiHrG2muhzHk01Ib`
- Workflow Telegram (activo): ID `0EL3hcEqE0M0LSdV`

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
- Query params: `q=Santiago`, `appid=TU_OPENWEATHERMAP_KEY`, `units=metric`
- Obtén tu key gratis en [openweathermap.org/api](https://openweathermap.org/api) y pégala en el parámetro `appid`

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

## Credenciales requeridas

| Servicio | Cómo configurar |
|---|---|
| OpenWeatherMap API Key | Regístrate en [openweathermap.org](https://openweathermap.org/api), obtén tu key y pégala en el parámetro `appid` del nodo "API Clima" |
| Google Gemini (AI Studio) | Genera una API key en [aistudio.google.com](https://aistudio.google.com) y agrégala como credencial Google AI en n8n (modelo: `gemini-2.5-flash`) |

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

## Arquitectura de despliegue (estado post-disertación)

Una vez migrado a producción, el sistema opera en dos piezas:

### n8n — en la nube
El workflow de Telegram corre en la instancia cloud del equipo. Al tener URL pública propia, Telegram registra el webhook directamente sin necesidad de ngrok. El bot responde mientras la instancia esté activa.

### FastAPI (CNN) — en el PC del equipo, expuesto con ngrok
La inferencia corre local por el peso de las dependencias. Para que n8n cloud pueda llamarla:

```bash
# Terminal 1 — levantar la API
uvicorn src.api_vision:app --port 8001

# Terminal 2 — exponer el puerto al exterior
ngrok http 8001
```

El nodo "CNN Inferencia (FastAPI)" en n8n debe apuntar a la URL pública que entrega ngrok (`https://xxx.ngrok-free.app/predecir_muestra`).

**Tip:** ngrok permite un dominio estático gratuito para no tener que actualizar la URL del nodo cada vez que se reinicia:
```bash
ngrok http --domain=tu-subdominio.ngrok-free.app 8001
```
El subdominio se reserva una vez en el dashboard de ngrok y queda fijo.

### Requisitos en el PC que corre FastAPI
- Python con las dependencias instaladas (`pip install fastapi uvicorn torch torchvision pillow python-multipart`)
- ngrok instalado y autenticado con cuenta gratuita
- El repo clonado con `modelo_vision.pth` presente

### Conexión MCP con n8n local (referencia de desarrollo)
Durante el desarrollo se usó MCP para inspeccionar y disparar workflows sin abrir la UI.
- **Qué puede hacer:** listar workflows, leer webhooks, disparar ejecuciones vía webhook
- **Qué NO puede:** editar configuración de nodos (eso es manual en la UI)

---

## Telegram Trigger y exposición pública de n8n

El Telegram Trigger funciona via **webhook**: Telegram envía las actualizaciones (fotos, mensajes) a una URL de n8n. Para que esto funcione, n8n debe ser accesible desde internet.

### En desarrollo local (ambiente actual)

**Opción A — tunnel incorporado en n8n:**
```bash
npx n8n start --tunnel
```
⚠️ **No funciona en v2.19.5** (versión actual). No genera URL pública.

**Opción B — ngrok manual (solución actual):**
```bash
ngrok http 5678
```
Genera una URL pública (`https://abc123.ngrok-free.app`) que redirige a `localhost:5678`. n8n la detecta automáticamente al activar el workflow de Telegram.

### En producción (n8n cloud / VPS con IP pública)
No se necesita tunnel ni ngrok. La instancia ya tiene URL pública y Telegram llega directamente. Por eso la migración a la instancia cloud del compañero elimina esta complejidad.

### Bot de Telegram del proyecto
- **Bot:** `@FitoScanAIBot` (nombre: FitoScanBot)
- **Token:** guardado en la credencial "FitoScan Bot" de n8n — NO hardcodear en código ni en archivos del repo
- **Workflow:** `n8n_workflow_telegram.json` (ID en n8n local: `0EL3hcEqE0M0LSdV`)

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

**Objetivo:** Documentar, corregir el modelo CNN y construir el flujo de Telegram.

**Lo que se hizo:**
- Copiado `mapa_flujo_telegram_agricola.html` como referencia del flujo objetivo
- Confirmada conexión MCP de Claude Code con n8n local
- **Bug crítico encontrado y corregido:** modelo entrenado con datos sintéticos (`data_train/`) en vez de imágenes reales (`data/`). Reentrenado con PlantVillage real → ~87% precisión
- **Segundo bug corregido:** `CLASS_NAMES` estaba en orden incorrecto respecto al orden alfabético que usa `ImageFolder` → etiquetas trocadas. Corregido en `api_vision.py`
- Identificado desbalance del dataset: 152 `Planta_Sana` vs 1000 de las otras dos clases (punto a mencionar ante el profe como área de mejora)
- README actualizado: nombre raíz, archivos nuevos, paso de instalación, roadmap con Telegram
- Credenciales sensibles limpiadas de `SESION.md` (repo es público)
- Creado bot `@FitoScanAIBot` en BotFather, credencial "FitoScan Bot" en n8n
- Generado `n8n_workflow_telegram.json` e importado en n8n (ID: `0EL3hcEqE0M0LSdV`)
- Descubierto que `npx n8n start --tunnel` no funciona en v2.19.5 → usando ngrok
- Workflow de Telegram configurado: API Clima ✅, credencial Telegram ✅, Gemini ✅
- 5 commits locales en `master` (sin push — remote es repo de Pablo)

**Estado final:** ✅ **Flujo completo operativo y probado desde celular externo.** Bot `@FitoScanAIBot` recibe fotos de cualquier usuario, la CNN diagnostica (ej. Oidio_Vid con alta confianza), cruza con clima real de Santiago vía OpenWeatherMap, Gemini genera recomendación de tratamiento personalizada y la devuelve al chat de Telegram. Probado exitosamente con imagen de rosa con Oídio desde dispositivo externo. Disertación en ~1 día.

---

*Añadir una entrada en "Historial de sesiones" cada vez que se trabaje en el proyecto.*
