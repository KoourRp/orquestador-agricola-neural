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
| `n8n workflows/n8n_workflow_demo.json` | Flujo n8n con formulario web para demo/presentación (Form Trigger) |
| `n8n workflows/n8n_workflow_telegram.json` | **Flujo v1 (estable)**: bot simple, clima Santiago fijo. ID: `0EL3hcEqE0M0LSdV` |
| `n8n workflows/n8n_workflow_telegram_ubicacion.json` | **Flujo v2 (activo)**: bot completo con ubicación GPS, agente de recepción, ayuda contextual. ID: `EKNCDzY2Xf5DPyeE` |
| `n8n workflows/n8n_workflow_final.json` | Flujo n8n para producción (recibe imágenes por webhook) |
| `data/` | Dataset PlantVillage 14 clases (~11.500 imgs, gitignoreado) |
| `graphify-out/graph.html` | Grafo interactivo de la arquitectura del proyecto (estado actual del código) |
| `mapa_flujo_telegram_agricola.html` | Mockup del flujo objetivo original (referencia histórica) |

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

## Configuración del flujo n8n (v2 — activo)

### Arquitectura completa del flujo v2

```
Telegram Trigger
    ↓
IF: ¿Es texto?  (message?.text || '')
    ├─ true  → Gemini: Recepción → Telegram: Bienvenida
    └─ false ↓
IF: ¿Tiene ubicación?  (message.location exists)
    ├─ true  → Guardar Ubicación (Code) → Telegram: Confirma Ubicación
    └─ false ↓
IF: ¿Tiene foto?  (message.photo?.length > 0)
    ├─ true  → Telegram: Descarga Foto → CNN Inferencia (FastAPI)
    │              → Resolver Ubicación (Code)
    │              → API Clima (OpenWeather)
    │              → Agrónomo (Gemini) → Telegram: Envía Veredicto
    └─ false → Telegram: Mensaje de Ayuda
```

### Nodo — IF: ¿Es texto?
- Condición: `{{ $json.message?.text || '' }}` → String → is not empty
- true: cualquier texto activa el agente de recepción (no solo /start)

### Nodo — Gemini: Recepción
- Tipo: Basic LLM Chain (LangChain)
- Source for Prompt: Define below
- Prompt (User Message): `{{ $json.message.text || 'Hola' }}`
- System Message: agente de recepción que explica qué hace el bot, cómo compartir ubicación GPS, y responde preguntas de fitopatología. Máx 4 líneas.
- Modelo: Google Gemini Chat Model → `models/gemini-2.5-flash`, credencial `NicoTY Gemini`

### Nodo — Guardar Ubicación (Code)
- Guarda `{ lat, lon }` en `$getWorkflowStaticData('global')['loc_<chatId>']`
- Persiste entre ejecuciones del workflow

### Nodo — Resolver Ubicación (Code)
- Lee staticData para obtener GPS guardado del chat
- Agrega `lat` y `lon` al JSON (fallback: Santiago `-33.45, -70.67`)
- Pasa también `diagnostico` y `confianza` del CNN hacia adelante

### Nodo — API Clima (OpenWeather)
- Tipo: HTTP Request → GET `https://api.openweathermap.org/data/2.5/weather`
- Parámetros: `lat={{ $json.lat }}`, `lon={{ $json.lon }}`, `appid=KEY`, `units=metric`, `lang=es`
- Devuelve ciudad, país, temp, sensación térmica, humedad, descripción

### Nodo — Agrónomo (Gemini)
- Tipo: LangChain Chain LLM, modelo `gemini-2.5-flash`
- Max tokens: 4096 (sin límite de líneas), Temperature: 0.3, Top P: 0.8
- Respuesta estructurada en **2 bloques**:
  - Bloque 1: 📍 Ciudad, País | 🌡️ temp | 💧 humedad
  - Bloque 2: diagnóstico completo con agente causal, síntomas, tratamiento, consideración climática
- Reglas especiales: lluvia→sistémicos, Tizon_Tardio→metalaxil/mancozeb, virus→eliminación+vectores, Oidio_Vid→azufre mojable

### Nodo — Telegram: Mensaje de Ayuda
- Se activa cuando el mensaje no es texto, ubicación ni foto (sticker, audio, etc.)
- Texto: instrucciones de uso resumidas

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
- **Token:** guardado en la credencial "FitoScanAIBot" de n8n — NO hardcodear en código ni en archivos del repo
- **Workflow v1 (referencia):** `n8n workflows/n8n_workflow_telegram.json` (ID: `0EL3hcEqE0M0LSdV`) — flujo estable, clima Santiago fijo
- **Workflow v2 (activo):** `n8n workflows/n8n_workflow_telegram_ubicacion.json` (ID: `EKNCDzY2Xf5DPyeE`) — flujo completo con GPS, recepción IA, ayuda contextual
- ⚠️ Solo uno puede estar activo a la vez — el webhook de Telegram solo apunta a uno

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
- Confirmada conexión MCP con n8n local
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

### Sesión 3 — 2026-06-28 (continuación)

**Objetivo:** Ampliar el dataset de 3 clases a 14 usando PlantVillage completo.

**Decisiones tomadas:**
- `data/` excluido de git (no estaba trackeado, agregado a `.gitignore` igual como prevención)
- Dataset expandido de 3 → 14 clases usando `C:\Users\nicoa\Downloads\archive\PlantVillage`
- `Planta_Sana` se construye fusionando `Potato___healthy` + `Tomato_healthy` + `Pepper__bell___healthy` (cap ~1000) para corregir el desbalance original de 152 imágenes
- `Oidio_Vid` se mantiene desde el dataset original (no está en el archive de Kaggle emmarex)
- README actualizado con sección Dataset completa y tabla de mapeo Kaggle → carpetas del proyecto

**Completado en esta sesión:**
- Script `preparar_dataset.py` ejecutado — data/ migrada a 14 clases (~11.500 imgs)
- CNN reentrenada con 14 clases, fc1=256, augmentation, WeightedRandomSampler
- Loss final: 0.2104 (convergencia en 15 épocas, partió en 1.4425)
- `modelo_vision.pth` regenerado (8.4MB, era 2MB con 3 clases)
- `CLASS_NAMES` en `api_vision.py` actualizado a 14 clases

**Pendiente:**
- Actualizar prompt de Gemini en n8n para manejar 14 diagnósticos
- Probar bot con las nuevas clases (tomate, pimiento, etc.)
- Push al fork y PR

---

### Sesión 4 — 2026-06-28 (continuación)

**Objetivo:** Agregar ubicación dinámica al nodo de clima + actualizar grafo.

**Lo que se hizo:**
- Grafo regenerado con graphify: 40 nodos, 47 aristas, 8 comunidades. Refleja los 3 scripts Python con 14 clases, WeightedRandomSampler y la arquitectura de capas.
- `n8n_workflow_telegram.json` reestructurado para soportar ubicación dinámica en el nodo API Clima:
  - Nuevo nodo **IF: ¿Foto o Ubicación?** justo después del trigger — divide el flujo según el tipo de mensaje
  - Rama foto (true): flujo existente, pero el nodo HTTP de clima reemplazado por Code node
  - Rama ubicación/texto (false): Code node guarda `lat/lon` en `staticData[loc_<chatId>]`, responde confirmación
  - Code node **API Clima (OpenWeather)**: prioridad → GPS guardado → caption de la foto → Santiago (fallback)

**Lógica de ubicación:**
```
staticData['loc_<chatId>'] = { lat, lon }   ← guardado al enviar 📍
↓
foto llega → Code node lee staticData → si existe, usa lat/lon en OWM
              → si no, lee caption → si no, usa Santiago
```

**Flujo de usuario:**
1. Enviar 📍 ubicación → bot responde "Ubicación guardada ✓"
2. Enviar foto de planta → diagnóstico usa el clima real de esa zona
3. Enviar foto con caption (ej: "Valparaíso") → usa esa ciudad sin necesitar ubicación
4. Enviar foto sin nada → fallback Santiago

**Pendiente:**
- Importar el JSON actualizado en n8n y probar ambas ramas (ubicación GPS + foto)
- Recordar reemplazar `TU_OPENWEATHERMAP_KEY` en el Code node de clima al importar

---

### Sesión 5 — 2026-06-28 (continuación)

**Objetivo:** Construir y depurar el flujo v2 completo con ubicación dinámica, agente de recepción IA y prompts finales.

**Lo que se hizo:**

**Flujo v2 — construcción e importación:**
- Reescrito `n8n_workflow_telegram_ubicacion.json` desde cero: Switch node (typeVersion 3 no soportado en n8n 2.19.5) reemplazado por 3 IF nodes encadenados
- Corregida estructura de `replyKeyboardMarkup` y `replyKeyboardRemove` para importación correcta
- Todos los n8n workflows movidos a carpeta `n8n workflows/`
- Flujo importado en n8n como workflow `EKNCDzY2Xf5DPyeE`, flujo v1 desactivado

**Bugs encontrados y resueltos:**
- `IF: ¿Tiene foto?` — `message.photo` es array, no objeto → condición: `{{ ($json.message.photo || []).length }}` Number → greater than → 0
- `API Clima` con lat/lon dinámico — `$getWorkflowStaticData` no disponible en expresiones HTTP Request → Code node "Resolver Ubicación" intermedio expone `$json.lat` / `$json.lon`
- `IF: ¿Es texto?` — `undefined` se evaluaba como string `"undefined"` (not empty → siempre true) → fix final: `{{ ($json.message.text || '').length }}` Number → greater than → 0
- `Telegram: Bienvenida` con Chat ID incorrecto — `{{ $json.text }}` estaba en Chat ID en vez de en Text
- `Gemini: Recepción` recibía fotos — mismo bug del IF anterior, corregido con `.length`

**Prompts finales:**

**Agrónomo (Gemini)** — variables del nodo:
```
Diagnóstico CNN: {{$node["CNN Inferencia (FastAPI)"].json["diagnostico"]}}
Confianza: {{$node["CNN Inferencia (FastAPI)"].json["confianza"]}}
Ubicación: {{$json["name"]}}, {{$json["sys"]["country"]}}
Clima: {{$json["main"]["temp"]}}°C (sensación {{$json["main"]["feels_like"]}}°C) | Humedad: {{$json["main"]["humidity"]}}% | {{$json["weather"][0]["description"]}}
```
Comportamiento: 3 niveles de confianza (≥0.80 definitivo 4 líneas / 0.65-0.79 tentativo 3 líneas / <0.65 pide mejor foto 2 líneas). Nunca menciona porcentajes al usuario. Respuesta en 2 partes: datos del entorno + diagnóstico. Termina siempre con disclaimer: "⚠️ Este diagnóstico es orientativo. Ante dudas, consulta a un agrónomo certificado." Si la ubicación es un barrio, menciona la ciudad principal. Reglas especiales por enfermedad (virus sin cura, Tizon_Tardio urgente, Oidio_Vid azufre, lluvia→sistémicos).
Configuración: Temperature 0.3, Top P 0.8, Max tokens 4096.

**Gemini: Recepción** — agente de bienvenida:
Responde cualquier texto. Explica que el bot analiza fotos de hojas + usa clima GPS. Para comenzar necesita: 📍 ubicación (una sola vez, vía 📎 → Ubicación) + 📸 foto de la hoja. Máx 4 líneas, sin markdown.

**Agente de recepción — nodo `Gemini: Recepción`:**
- Tipo: Basic LLM Chain, Source: Define below
- Prompt User Message: `{{ ($json.message?.text || 'Hola') }}`
- System: ver prompt completo en n8n
- Modelo: Google Gemini Chat Model, `models/gemini-2.5-flash`, credencial `NicoTY Gemini`
- Respuesta estructurada en 2 bloques: datos del entorno + diagnóstico
- Agregados campos: `sys.country`, `feels_like`, `name` (ciudad)
- Reglas específicas por enfermedad: Tizon_Tardio → metalaxil/mancozeb, Oidio_Vid → azufre mojable, virus → eliminación + vectores
- Humedad > 80% → productos sistémicos o biológicos
- Max tokens: 4096, Temperature: 0.3, Top P: 0.8

**Agente de recepción (nuevo):**
- Nodo `Gemini: Recepción` (Basic LLM Chain) insertado antes de `Telegram: Bienvenida`
- Responde inteligentemente a cualquier texto: saludo, preguntas sobre el bot, fitopatología general, fuera de alcance
- System prompt explica el GPS: el usuario comparte ubicación desde Telegram (📎 → Ubicación), el bot la recuerda automáticamente para todas las fotos siguientes
- `IF: ¿Es texto?` ahora responde a CUALQUIER texto (no solo /start) — más intuitivo
- Nodo `Telegram: Mensaje de Ayuda` para mensajes que no son texto/foto/ubicación (stickers, audio, etc.)

**Estado final:** ✅ **Flujo v2 completamente operativo.** Bot responde inteligentemente a cualquier mensaje de texto, guarda ubicación GPS, diagnostica enfermedades en 14 clases con clima real localizado, y entrega respuesta estructurada con datos del entorno + diagnóstico completo con agente causal, síntomas y tratamiento.

---

### Sesión 6 — 2026-06-29

**Objetivo:** Agregar contexto de ubicación al agente de recepción — el bot sabe en qué ciudad está el usuario y puede responder preguntas sobre el clima actual.

**Lo que se hizo:**

**Migración a producción (completada en esta sesión):**
- FastAPI CNN desplegada en HuggingFace Spaces (Docker, puerto 7860) como servicio permanente
- Workflow migrado a n8n cloud (instancia de Pablo, v2.27.4) — sin ngrok, sin PC encendido

**Feature: Contexto de ubicación en Gemini: Recepción:**

Nueva arquitectura del flujo de texto:
```
IF: ¿Es texto? → true → Preparar Contexto (Code) → IF: ¿Tiene GPS?
                                                          true  → OWM Recepción (HTTP) → Gemini: Recepción
                                                          false ──────────────────────→ Gemini: Recepción
                                                                                         → Telegram: Bienvenida
```

**Nodo — Preparar Contexto (Code node, nuevo):**
```javascript
const msg = $input.first().json.message;
const chatId = msg.chat.id;
const staticData = $getWorkflowStaticData('global');
const loc = staticData[`loc_${chatId}`];
return [{ json: { ...$input.first().json, loc_lat: loc?.lat ?? null, loc_lon: loc?.lon ?? null, tiene_gps: !!loc } }];
```

**Nodo — IF: ¿Tiene GPS?**
- Condition: `{{ $json.tiene_gps }}` → Boolean → is true

**Nodo — OWM Recepción (HTTP Request, nuevo):**
- GET `https://api.openweathermap.org/data/2.5/weather`
- `lat`: `{{ $json.loc_lat }}`, `lon`: `{{ $json.loc_lon }}`
- Mismo appid/units/lang que API Clima del flujo de fotos

**Nodo — OWM Confirma (HTTP Request, nuevo):**
- Insertado entre "Guardar Ubicación" y "Telegram: Confirma Ubicación"
- `lat`: `{{ $json.lat }}`, `lon`: `{{ $json.lon }}` (sin prefijo `loc_`, viene directo de Guardar Ubicación)

**Nodo — Telegram: Confirma Ubicación (actualizado):**
```
{{ `📍 Ubicación guardada: ${$json.name}, ${$json.sys.country}. Usaré el clima aproximado de tu zona para los análisis. 🌱` }}
```
Chat ID: `{{ $('Guardar Ubicación').first().json.chatId }}`

**Nodo — Gemini: Recepción — Prompt User Message (actualizado):**
```
{{ $('Preparar Contexto').first().json.message?.text || 'Hola' }}
[Contexto ubicación: {{ $json.name ? $json.name + ', ' + $json.sys.country + ' (GPS guardado)' : 'Sin ubicación guardada' }}]
{{ $json.main ? `[Clima actual: ${$json.weather[0].description}, ${$json.main.temp}°C, sensación ${$json.main.feels_like}°C, humedad ${$json.main.humidity}%]` : '' }}
```

**Retry configurado en Gemini: Recepción:** Settings → Retry on Fail → 3 intentos, 10s entre cada uno (Gemini 2.5 Flash Lite devuelve 503 en picos de demanda).

**Estado final:** ✅ **Sistema completamente operativo en la nube.** El bot sabe la ciudad del usuario, responde preguntas de clima, confirma la ubicación guardada con nombre real al recibirla, y el flujo completo (Telegram → n8n cloud → HuggingFace CNN → OWM → Gemini → Telegram) corre sin PC local.

---

*Añadir una entrada en "Historial de sesiones" cada vez que se trabaje en el proyecto.*
