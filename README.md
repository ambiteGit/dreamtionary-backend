# Dreamtionary — Backend

API que conecta la app móvil con Claude para generar interpretaciones de sueños.

## Desplegar en Render (recomendado, capa gratuita disponible)

1. Crea una cuenta en https://render.com
2. Sube esta carpeta (`dreamtionary-backend`) a un repositorio de GitHub
3. En Render: **New +** → **Web Service** → conecta tu repositorio
4. Configuración:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. En la sección **Environment**, añade la variable de entorno:
   - `ANTHROPIC_API_KEY` = tu clave de la API de Anthropic (consíguela en https://console.anthropic.com)
6. Despliega. Render te dará una URL tipo `https://dreamtionary-api.onrender.com`

## Conectar la app móvil a este backend

Una vez desplegado, copia la URL que te dio Render y pégala en el archivo
`dreamtionary-app/src/services/dreamService.js`, reemplazando:

```js
const API_BASE_URL = "https://TU-BACKEND.example.com";
```

por tu URL real, por ejemplo:

```js
const API_BASE_URL = "https://dreamtionary-api.onrender.com";
```

## Probarlo en local antes de desplegar

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="tu-clave-aqui"     # En Windows: set ANTHROPIC_API_KEY=tu-clave-aqui
uvicorn main:app --reload
```

Luego abre http://localhost:8000 en el navegador — deberías ver `{"status": "ok", ...}`.

Para probar el endpoint real:
```bash
curl -X POST http://localhost:8000/interpretar \
  -H "Content-Type: application/json" \
  -d '{"palabras": ["coche", "casa", "avion"]}'
```
