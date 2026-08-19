"""
Dreamtionary - Backend API
---------------------------
Expone estos endpoints:
- POST /interpretar          → gratis, hasta 5 palabras clave, usa Haiku (barato)
- POST /interpretar-premium  → de pago, texto libre del sueño completo, usa Sonnet (mejor calidad)
- POST /diccionario          → fallback IA para términos no presentes en la base local
- POST /diccionario-foto     → foto concepto (Unsplash) para un término, cacheada en servidor
- POST /diccionario-ampliado → interpretación ampliada con matices/variantes de un término

Todos aceptan un campo "idioma" con cualquiera de los 24 idiomas soportados por la app
(es, en, fr, pt, zh, de, it, hi, ar, ja, ru, id, no, sv, pl, fi, ur, ko, tr, tl, vi, th, fa, nl).
"""

import json
import os
import httpx
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

UNSPLASH_ACCESS_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")

# Caché en memoria del proceso para las fotos concepto del diccionario.
# Como el número de términos posibles es limitado (73 base + los que se
# busquen por IA), cada término solo necesita pedirse a Unsplash una vez;
# el resto de peticiones (de cualquier usuario) se sirven desde aquí.
_cache_fotos = {}

# Modelos: Haiku para el intérprete gratuito (barato), Sonnet para el premium (mejor calidad)
MODEL_FREE = "claude-haiku-4-5-20251001"
MODEL_PREMIUM = "claude-sonnet-5"

IDIOMAS_VALIDOS = {
    "es", "en", "fr", "pt", "zh", "de", "it", "hi", "ar", "ja", "ru", "id",
    "no", "sv", "pl", "fi", "ur", "ko", "tr", "tl", "vi", "th", "fa", "nl",
}
NOMBRE_IDIOMA = {
    "es": "español",
    "en": "English",
    "fr": "français",
    "pt": "português",
    "zh": "中文 (Chinese)",
    "de": "Deutsch",
    "it": "italiano",
    "hi": "हिन्दी (Hindi)",
    "ar": "العربية (Arabic)",
    "ja": "日本語 (Japanese)",
    "ru": "русский (Russian)",
    "id": "Bahasa Indonesia",
    "no": "norsk (Norwegian)",
    "sv": "svenska (Swedish)",
    "pl": "polski (Polish)",
    "fi": "suomi (Finnish)",
    "ur": "اردو (Urdu)",
    "ko": "한국어 (Korean)",
    "tr": "Türkçe (Turkish)",
    "tl": "Tagalog",
    "vi": "Tiếng Việt (Vietnamese)",
    "th": "ไทย (Thai)",
    "fa": "فارسی (Persian)",
    "nl": "Nederlands (Dutch)",
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


class DiccionarioFotoRequest(BaseModel):
    termino: str = Field(..., min_length=1, max_length=60)


class DiccionarioAmpliadoRequest(BaseModel):
    termino: str = Field(..., min_length=1, max_length=60)
    significado_base: str = Field(..., min_length=1, max_length=300)
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


@app.post("/diccionario-foto")
async def diccionario_foto(request: DiccionarioFotoRequest):
    """
    Devuelve una foto concepto (Unsplash) para un término del diccionario de
    sueños. Cachea el resultado en el servidor: sin importar cuántos usuarios
    busquen el mismo término, solo se llama a Unsplash una vez.
    """
    clave = request.termino.lower().strip()

    if clave in _cache_fotos:
        return _cache_fotos[clave]

    if not UNSPLASH_ACCESS_KEY:
        return {"url": None}

    try:
        async with httpx.AsyncClient() as http_client:
            respuesta = await http_client.get(
                "https://api.unsplash.com/search/photos",
                params={"query": request.termino, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al consultar Unsplash: {e}")

    if respuesta.status_code != 200:
        return {"url": None}

    datos = respuesta.json()
    resultados = datos.get("results", [])
    if not resultados:
        resultado = {"url": None}
    else:
        foto = resultados[0]
        resultado = {
            "url": foto["urls"]["regular"],
            "urlThumb": foto["urls"]["small"],
            "autor": foto["user"]["name"],
            "autorUrl": foto["user"]["links"]["html"],
        }

    _cache_fotos[clave] = resultado
    return resultado


@app.post("/diccionario-ampliado")
def diccionario_ampliado(request: DiccionarioAmpliadoRequest):
    """
    Genera una explicación ampliada y matizada de un símbolo de sueños,
    cubriendo los distintos contextos/variantes en los que puede aparecer
    (p. ej. "matar" no es lo mismo que "que te maten", que "soñar que
    matan a un ser querido", etc.).
    """
    idioma = normalizar_idioma(request.idioma)
    nombre_idioma = NOMBRE_IDIOMA[idioma]

    prompt = f"""You are an expert in dream symbolism. The user is viewing the entry \
for the term "{request.termino}" in a dream dictionary.

Short meaning already shown: "{request.significado_base}"

Write an EXPANDED explanation of this symbol, in {nombre_idioma} (use that language \
for the entire response).

IMPORTANT — the explanation should cover the different nuances/variants depending on \
the exact context of the dream, for example (adapt to the actual term):
- Being the active subject of the action (you do something) vs. being the passive \
subject (it happens to you) vs. witnessing it happen to a third party (especially if \
it's someone close/loved).
- If relevant, the emotional tone of the dream (fear, calm, relief...) can change the \
meaning.
- Any other nuance specific and relevant to this particular symbol.

Format: 3-4 short paragraphs, warm and reflective tone (not clinical), in the style of \
a dream dictionary for a general audience. Do not use headings or bullet lists, just \
flowing prose paragraphs.

Do not include any legal disclaimer — that is already shown elsewhere in the app.
Remember: respond entirely in {nombre_idioma}."""

    try:
        respuesta = client.messages.create(
            model=MODEL_FREE,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        texto_ampliado = "".join(
            bloque.text for bloque in respuesta.content if bloque.type == "text"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error al ampliar el significado: {e}")

    return {"significado_ampliado": texto_ampliado}
