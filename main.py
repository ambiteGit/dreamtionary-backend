"""
Dreamtionary - Backend API
---------------------------
Expone dos endpoints:
- POST /interpretar          → gratis, hasta 5 palabras clave, usa Haiku (barato)
- POST /interpretar-premium  → de pago, texto libre del sueño completo, usa Sonnet (mejor calidad)

Ambos aceptan un campo "idioma" (es, en, fr, pt, zh) para responder en ese idioma.
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from anthropic import Anthropic

app = FastAPI(title="Dreamtionary API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Anthropic()  # Lee ANTHROPIC_API_KEY de las variables de entorno del servidor

# Modelos: Haiku para el intérprete gratuito (barato), Sonnet para el premium (mejor calidad)
MODEL_FREE = "claude-haiku-4-5-20251001"
MODEL_PREMIUM = "claude-sonnet-5"

IDIOMAS_VALIDOS = {"es", "en", "fr", "pt", "zh"}
NOMBRE_IDIOMA = {
    "es": "español",
    "en": "English",
    "fr": "français",
    "pt": "português",
    "zh": "中文 (Chinese)",
}

SYMBOLS_PATH = os.path.join(os.path.dirname(__file__), "dream_symbols_i18n.json")
with open(SYMBOLS_PATH, "r", encoding="utf-8") as f:
    SYMBOLS_DB = json.load(f)


class InterpretarRequest(BaseModel):
    palabras: list[str] = Field(..., min_length=1, max_length=5)
    idioma: str = Field(default="es")


class InterpretarPremiumRequest(BaseModel):
    texto: str = Field(..., min_length=10, max_length=3000)
    idioma: str = Field(default="es")


class DiccionarioRequest(BaseModel):
    palabra: str = Field(..., min_length=1, max_length=60)
    idioma: str = Field(default="es")


def normalizar_idioma(idioma: str) -> str:
    return idioma if idioma in IDIOMAS_VALIDOS else "es"


def normalizar(palabra: str) -> str:
    return palabra.strip().lower().replace(" ", "_")


def buscar_significado(palabra: str, idioma: str):
    """Busca la palabra por su 'key' canónico o por el nombre en el idioma dado."""
    palabra_normalizada = normalizar(palabra)
    for simbolos in SYMBOLS_DB.values():
        for entrada in simbolos:
            nombre_en_idioma = normalizar(entrada[idioma]["simbolo"])
            if entrada["key"] == palabra_normalizada or nombre_en_idioma == palabra_normalizada:
                return entrada[idioma]
    return None


def construir_contexto(palabras: list[str], idioma: str):
    lineas = []
    no_encontradas = []
    for palabra in palabras:
        entrada = buscar_significado(palabra, idioma)
        if entrada:
            lineas.append(f"- {entrada['simbolo']}: {entrada['significado']}")
        else:
            no_encontradas.append(palabra)
    contexto = "\n".join(lineas) if lineas else "(No reference symbols found)"
    return contexto, no_encontradas


@app.get("/")
def health_check():
    return {"status": "ok", "servicio": "Dreamtionary API"}


@app.post("/interpretar")
def interpretar(request: InterpretarRequest):
    idioma = normalizar_idioma(request.idioma)
    if not request.palabras:
        raise HTTPException(status_code=400, detail="Debes enviar al menos una palabra")

    contexto, no_encontradas = construir_contexto(request.palabras, idioma)
    palabras_str = ", ".join(request.palabras)
    nombre_idioma = NOMBRE_IDIOMA[idioma]

    prompt = f"""You are a warm, thoughtful dream interpreter. You are never deterministic \
or alarmist: you present interpretations as possibilities to explore, not absolute truths.

IMPORTANT: Write your entire response in {nombre_idioma}.

The user dreamed about these elements: {palabras_str}.

Reference meanings for each symbol (traditional dream symbolism):
{contexto}

Instructions:
- Write a short 3-4 paragraph interpretation.
- Do not list the symbols separately; weave them into one coherent narrative, \
as if they all appeared in the same dream.
- If a symbol has no reference meaning above, interpret it using your own knowledge \
of dream symbolism, integrating it naturally.
- End with a short reflective question inviting the user to think about their current \
life situation.
- Tone: warm and close, like a knowledgeable friend, not an oracle.
- Remember: respond entirely in {nombre_idioma}."""

    try:
        respuesta = client.messages.create(
            model=MODEL_FREE,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        texto_interpretacion = "".join(
            bloque.text for bloque in respuesta.content if bloque.type == "text"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al generar la interpretación: {e}")

    return {
        "interpretacion": texto_interpretacion,
        "simbolos_usados": contexto,
        "palabras_sin_definir": no_encontradas,
    }


@app.post("/interpretar-premium")
def interpretar_premium(request: InterpretarPremiumRequest):
    """
    Versión premium: el usuario describe el sueño completo con sus propias palabras.
    La IA identifica los símbolos relevantes por sí misma (no dependemos del diccionario)
    y genera una interpretación más extensa y detallada, con un modelo de mayor calidad.
    """
    idioma = normalizar_idioma(request.idioma)
    nombre_idioma = NOMBRE_IDIOMA[idioma]

    prompt = f"""You are an expert, warm dream interpreter with deep knowledge of dream \
symbolism and psychology (Jungian and modern approaches). You are never deterministic or \
alarmist: you present interpretations as possibilities to explore, not absolute truths.

IMPORTANT: Write your entire response in {nombre_idioma}.

The user wrote a full description of their dream:
\"\"\"{request.texto}\"\"\"

Instructions:
- Identify the key symbols, emotions, and narrative arc in the dream yourself.
- Write a rich, structured interpretation (5-7 paragraphs) that:
  1. Reflects back the emotional tone of the dream
  2. Explores the 2-4 most meaningful symbols/themes and how they connect
  3. Offers a possible overall meaning, tied to common life situations (without assuming \
specific facts about the user's life)
  4. Ends with 1-2 reflective questions to help the user connect it to their own life
- Tone: warm, insightful, like a knowledgeable friend — never clinical, never absolute.
- Remember: respond entirely in {nombre_idioma}."""

    try:
        respuesta = client.messages.create(
            model=MODEL_PREMIUM,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        texto_interpretacion = "".join(
            bloque.text for bloque in respuesta.content if bloque.type == "text"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al generar la interpretación: {e}")

    return {"interpretacion": texto_interpretacion}


@app.post("/diccionario")
def diccionario_ia(request: DiccionarioRequest):
    """
    Fallback del Diccionario: si una palabra no está en dream_symbols_i18n.json,
    se le pide a la IA (Haiku, barato) una definición corta en estilo simbología
    onírica, coherente con el resto del diccionario. El resultado se cachea en
    el propio dispositivo del usuario (AsyncStorage), así que esta llamada solo
    ocurre la primera vez que alguien busca esa palabra en ese idioma.
    """
    idioma = normalizar_idioma(request.idioma)
    nombre_idioma = NOMBRE_IDIOMA[idioma]
    palabra = request.palabra.strip()

    # Primero comprobamos si ya existe en la base de datos local (evita gastar IA de más)
    entrada_local = buscar_significado(palabra, idioma)
    if entrada_local:
        return {"simbolo": entrada_local["simbolo"], "significado": entrada_local["significado"], "fuente": "local"}

    prompt = f"""You are a dream symbolism dictionary. Given a single word or short \
phrase, respond with its traditional dream-symbolism meaning in ONE short sentence \
(max 20 words), in the same neutral, warm style as a dream dictionary entry.

IMPORTANT: Respond ONLY in {nombre_idioma}. Respond ONLY with the meaning sentence, \
no preamble, no quotes, no extra text.

Word: {palabra}"""

    try:
        respuesta = client.messages.create(
            model=MODEL_FREE,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        significado = "".join(
            bloque.text for bloque in respuesta.content if bloque.type == "text"
        ).strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar el diccionario: {e}")

    return {"simbolo": palabra.capitalize(), "significado": significado, "fuente": "ia"}
