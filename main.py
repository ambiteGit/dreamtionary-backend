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


class DiccionarioFotoUsoRequest(BaseModel):
    download_location: str = Field(..., min_length=1, max_length=500)


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
        # Unsplash exige que los enlaces de atribución (tanto al fotógrafo
        # como a Unsplash) lleven parámetros UTM identificando la app.
        utm = "?utm_source=Dreamtionary&utm_medium=referral"
        resultado = {
            "url": foto["urls"]["regular"],
            "urlThumb": foto["urls"]["small"],
            "autor": foto["user"]["name"],
            "autorUrl": foto["user"]["links"]["html"] + utm,
            "unsplashUrl": "https://unsplash.com/" + utm,
            # Se envía al cliente para que él mismo lo devuelva al llamar a
            # /diccionario-foto-uso. Así el registro de "uso" no depende de
            # una caché en memoria del servidor, que se pierde cada vez que
            # Render reinicia el proceso por inactividad (plan gratuito).
            "downloadLocation": foto["links"]["download_location"],
        }

    _cache_fotos[clave] = resultado
    return resultado


@app.post("/diccionario-foto-uso")
async def diccionario_foto_uso(request: DiccionarioFotoUsoRequest):
    """
    Registra ante Unsplash que una foto se está mostrando/usando en la app,
    tal como exigen sus guías de la API ("Triggering downloads"). El cliente
    envía el download_location que recibió en /diccionario-foto — así este
    endpoint no depende de ninguna caché del servidor (que se perdería cada
    vez que Render reinicia el proceso por inactividad en el plan gratuito).

    Validamos que la URL sea realmente de la API de Unsplash, para no
    convertir esto en un proxy abierto a cualquier URL.
    """
    download_location = request.download_location

    if not download_location.startswith("https://api.unsplash.com/"):
        raise HTTPException(status_code=400, detail="URL de destino no válida")

    if not UNSPLASH_ACCESS_KEY:
        return {"ok": False}

    try:
        async with httpx.AsyncClient() as http_client:
            await http_client.get(
                download_location,
                headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
            )
    except Exception as e:
        # No interrumpimos la experiencia del usuario si esto falla; solo
        # registramos el error en los logs del servidor.
        print(f"Error registrando uso de foto en Unsplash: {e}")
        return {"ok": False}

    return {"ok": True}


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

import asyncio

LISTA_SIMBOLOS_250 = [{"key": "madre", "en": "Mother"}, {"key": "padre", "en": "Father"}, {"key": "ex_pareja", "en": "Ex-partner"}, {"key": "bebe", "en": "Baby"}, {"key": "desconocido", "en": "Stranger"}, {"key": "amigo", "en": "Friend"}, {"key": "enemigo", "en": "Enemy"}, {"key": "multitud", "en": "Crowd"}, {"key": "profesor", "en": "Teacher"}, {"key": "jefe", "en": "Boss"}, {"key": "hermano_hermana", "en": "Sibling"}, {"key": "novio_novia", "en": "Partner"}, {"key": "anciano", "en": "Elder"}, {"key": "nino", "en": "Child"}, {"key": "hijo_a", "en": "Child"}, {"key": "abuelos", "en": "Grandparents"}, {"key": "pareja_actual", "en": "Current partner"}, {"key": "companero_trabajo", "en": "Coworker"}, {"key": "dientes", "en": "Teeth"}, {"key": "caida_de_pelo", "en": "Hair loss"}, {"key": "embarazo", "en": "Pregnancy"}, {"key": "sangre", "en": "Blood"}, {"key": "desnudez", "en": "Nakedness"}, {"key": "ceguera", "en": "Blindness"}, {"key": "manos", "en": "Hands"}, {"key": "corazon", "en": "Heart"}, {"key": "envejecer", "en": "Aging"}, {"key": "enfermedad", "en": "Illness"}, {"key": "ojos", "en": "Eyes"}, {"key": "unas", "en": "Nails"}, {"key": "desmayo", "en": "Fainting"}, {"key": "vomito", "en": "Vomiting"}, {"key": "agua", "en": "Water"}, {"key": "fuego", "en": "Fire"}, {"key": "tierra_terremoto", "en": "Earthquake"}, {"key": "serpiente", "en": "Snake"}, {"key": "tormenta", "en": "Storm"}, {"key": "arbol", "en": "Tree"}, {"key": "oceano_mar", "en": "Ocean/sea"}, {"key": "sol", "en": "Sun"}, {"key": "luna", "en": "Moon"}, {"key": "lluvia", "en": "Rain"}, {"key": "montana", "en": "Mountain"}, {"key": "bosque", "en": "Forest"}, {"key": "flores", "en": "Flowers"}, {"key": "arana", "en": "Spider"}, {"key": "lobo", "en": "Wolf"}, {"key": "pajaro", "en": "Bird"}, {"key": "nieve", "en": "Snow"}, {"key": "viento", "en": "Wind"}, {"key": "arcoiris", "en": "Rainbow"}, {"key": "cueva", "en": "Cave"}, {"key": "desierto", "en": "Desert"}, {"key": "volar", "en": "Flying"}, {"key": "caer", "en": "Falling"}, {"key": "ser_perseguido", "en": "Being chased"}, {"key": "examen", "en": "Exam"}, {"key": "llegar_tarde", "en": "Being late"}, {"key": "perderse", "en": "Getting lost"}, {"key": "no_poder_gritar_o_moverse", "en": "Unable to scream or move"}, {"key": "pelea_lucha", "en": "Fight/struggle"}, {"key": "boda", "en": "Wedding"}, {"key": "viaje", "en": "Journey"}, {"key": "repetir_la_misma_accion", "en": "Repeating the same action"}, {"key": "esconderse", "en": "Hiding"}, {"key": "ahogarse", "en": "Drowning"}, {"key": "infidelidad", "en": "Infidelity"}, {"key": "ganar_loteria", "en": "Winning the lottery"}, {"key": "volver_escuela", "en": "Going back to school"}, {"key": "perder_algo_importante", "en": "Losing something important"}, {"key": "casa", "en": "House"}, {"key": "coche", "en": "Car"}, {"key": "avion", "en": "Airplane"}, {"key": "dinero", "en": "Money"}, {"key": "escaleras", "en": "Stairs"}, {"key": "puerta", "en": "Door"}, {"key": "llaves", "en": "Keys"}, {"key": "espejo", "en": "Mirror"}, {"key": "telefono", "en": "Phone"}, {"key": "colegio_universidad", "en": "School/university"}, {"key": "hospital", "en": "Hospital"}, {"key": "reloj", "en": "Clock"}, {"key": "comida", "en": "Food"}, {"key": "barco", "en": "Boat"}, {"key": "puente", "en": "Bridge"}, {"key": "ascensor", "en": "Elevator"}, {"key": "carcel", "en": "Jail/prison"}, {"key": "iglesia_templo", "en": "Church/temple"}, {"key": "ropa", "en": "Clothes"}, {"key": "cama", "en": "Bed"}, {"key": "muerte", "en": "Death"}, {"key": "funeral", "en": "Funeral"}, {"key": "renacimiento", "en": "Rebirth"}, {"key": "fantasma", "en": "Ghost"}, {"key": "cementerio", "en": "Cemetery"}, {"key": "esqueleto", "en": "Skeleton"}, {"key": "reencarnacion", "en": "Reincarnation"}, {"key": "alma_espiritu", "en": "Soul/spirit"}, {"key": "angel", "en": "Angel"}, {"key": "demonio", "en": "Demon"}, {"key": "sombra", "en": "Shadow"}, {"key": "suegros", "en": "In-laws"}, {"key": "cunado_a", "en": "Sibling-in-law"}, {"key": "sobrino_a", "en": "Niece/nephew"}, {"key": "tio_a", "en": "Aunt/uncle"}, {"key": "primo_a", "en": "Cousin"}, {"key": "vecino", "en": "Neighbor"}, {"key": "doctor", "en": "Doctor"}, {"key": "policia", "en": "Police officer"}, {"key": "sacerdote_pastor", "en": "Priest/pastor"}, {"key": "rey_reina", "en": "King/queen"}, {"key": "soldado", "en": "Soldier"}, {"key": "ladron", "en": "Thief"}, {"key": "celebridad", "en": "Celebrity"}, {"key": "gemelo", "en": "Twin"}, {"key": "mentor", "en": "Mentor"}, {"key": "rival", "en": "Rival"}, {"key": "persona_del_pasado", "en": "Person from the past"}, {"key": "extraterrestre", "en": "Alien"}, {"key": "cabello", "en": "Hair"}, {"key": "piel", "en": "Skin"}, {"key": "piernas", "en": "Legs"}, {"key": "pies", "en": "Feet"}, {"key": "espalda", "en": "Back"}, {"key": "estomago", "en": "Stomach"}, {"key": "cabeza", "en": "Head"}, {"key": "boca", "en": "Mouth"}, {"key": "oidos", "en": "Ears"}, {"key": "lengua", "en": "Tongue"}, {"key": "isla", "en": "Island"}, {"key": "volcan", "en": "Volcano"}, {"key": "cascada", "en": "Waterfall"}, {"key": "lago", "en": "Lake"}, {"key": "rio", "en": "River"}, {"key": "cielo", "en": "Sky"}, {"key": "nubes", "en": "Clouds"}, {"key": "estrellas", "en": "Stars"}, {"key": "playa", "en": "Beach"}, {"key": "colina", "en": "Hill"}, {"key": "rayo", "en": "Lightning"}, {"key": "niebla", "en": "Fog"}, {"key": "sequia", "en": "Drought"}, {"key": "huracan", "en": "Hurricane"}, {"key": "eclipse", "en": "Eclipse"}, {"key": "amanecer", "en": "Sunrise"}, {"key": "atardecer", "en": "Sunset"}, {"key": "gato", "en": "Cat"}, {"key": "perro", "en": "Dog"}, {"key": "caballo", "en": "Horse"}, {"key": "vaca", "en": "Cow"}, {"key": "cerdo", "en": "Pig"}, {"key": "raton", "en": "Mouse"}, {"key": "oso", "en": "Bear"}, {"key": "leon", "en": "Lion"}, {"key": "tigre", "en": "Tiger"}, {"key": "elefante", "en": "Elephant"}, {"key": "mono", "en": "Monkey"}, {"key": "conejo", "en": "Rabbit"}, {"key": "tortuga", "en": "Turtle"}, {"key": "toro", "en": "Bull"}, {"key": "oveja", "en": "Sheep"}, {"key": "cabra", "en": "Goat"}, {"key": "ciervo", "en": "Deer"}, {"key": "zorro", "en": "Fox"}, {"key": "abeja", "en": "Bee"}, {"key": "mariposa", "en": "Butterfly"}, {"key": "hormiga", "en": "Ant"}, {"key": "mosca", "en": "Fly"}, {"key": "rana", "en": "Frog"}, {"key": "ballena", "en": "Whale"}, {"key": "delfin", "en": "Dolphin"}, {"key": "tiburon", "en": "Shark"}, {"key": "cocodrilo", "en": "Crocodile"}, {"key": "buho", "en": "Owl"}, {"key": "aguila", "en": "Eagle"}, {"key": "cuervo", "en": "Crow/raven"}, {"key": "paloma", "en": "Dove"}, {"key": "cisne", "en": "Swan"}, {"key": "gallina_gallo", "en": "Hen/rooster"}, {"key": "pulpo", "en": "Octopus"}, {"key": "camaleon", "en": "Chameleon"}, {"key": "mudanza", "en": "Moving house"}, {"key": "divorcio", "en": "Divorce"}, {"key": "accidente_coche", "en": "Car accident"}, {"key": "incendio", "en": "Fire (blaze)"}, {"key": "inundacion", "en": "Flood"}, {"key": "ser_robado", "en": "Being robbed"}, {"key": "ganar_competencia", "en": "Winning a competition"}, {"key": "ser_rechazado", "en": "Being rejected"}, {"key": "ser_ignorado", "en": "Being ignored"}, {"key": "recibir_regalo", "en": "Receiving a gift"}, {"key": "dar_a_luz", "en": "Giving birth"}, {"key": "cirugia", "en": "Surgery"}, {"key": "hablar_publico", "en": "Public speaking"}, {"key": "ser_filmado", "en": "Being filmed"}, {"key": "quedarse_sin_bateria", "en": "Phone dying"}, {"key": "ser_invisible", "en": "Being invisible"}, {"key": "ordenador", "en": "Computer"}, {"key": "movil", "en": "Phone"}, {"key": "redes_sociales", "en": "Social media"}, {"key": "camara", "en": "Camera"}, {"key": "television", "en": "Television"}, {"key": "arma", "en": "Weapon"}, {"key": "cuchillo", "en": "Knife"}, {"key": "libro", "en": "Book"}, {"key": "carta", "en": "Letter"}, {"key": "maleta", "en": "Suitcase"}, {"key": "joyas", "en": "Jewelry"}, {"key": "ventana", "en": "Window"}, {"key": "sotano", "en": "Basement"}, {"key": "jardin", "en": "Garden"}, {"key": "piscina", "en": "Swimming pool"}, {"key": "supermercado", "en": "Supermarket"}, {"key": "aeropuerto", "en": "Airport"}, {"key": "estacion_tren", "en": "Train station"}, {"key": "pozo", "en": "Well"}, {"key": "faro", "en": "Lighthouse"}, {"key": "castillo", "en": "Castle"}, {"key": "torre", "en": "Tower"}, {"key": "laberinto", "en": "Labyrinth"}, {"key": "rojo", "en": "Red"}, {"key": "azul", "en": "Blue"}, {"key": "verde", "en": "Green"}, {"key": "negro", "en": "Black"}, {"key": "blanco", "en": "White"}, {"key": "amarillo", "en": "Yellow"}, {"key": "morado", "en": "Purple"}, {"key": "naranja", "en": "Orange"}, {"key": "rosa", "en": "Pink"}, {"key": "dorado", "en": "Gold"}, {"key": "ataud", "en": "Coffin"}, {"key": "tumba", "en": "Grave"}, {"key": "sacrificio", "en": "Sacrifice"}, {"key": "luz_final_tunel", "en": "Light at the end of the tunnel"}, {"key": "karma", "en": "Karma"}, {"key": "purgatorio", "en": "Purgatory"}, {"key": "espejismo", "en": "Mirage"}, {"key": "sueno_dentro_sueno", "en": "Dream within a dream"}, {"key": "deja_vu", "en": "Déjà vu"}, {"key": "voz_desconocida", "en": "Unknown voice"}, {"key": "silencio", "en": "Silence"}, {"key": "oscuridad", "en": "Darkness"}, {"key": "luz_brillante", "en": "Bright light"}, {"key": "instrumento_musical", "en": "Musical instrument"}, {"key": "musica", "en": "Music"}, {"key": "danza", "en": "Dance"}, {"key": "mascara", "en": "Mask"}, {"key": "disfraz", "en": "Costume"}, {"key": "tatuaje", "en": "Tattoo"}, {"key": "cicatriz", "en": "Scar"}]


@app.get("/admin/precargar-fotos-diccionario")
async def precargar_fotos_diccionario(clave: str = "", inicio: int = 0, cantidad: int = 40):
    if clave != "dreamtionary2026":
        raise HTTPException(status_code=403, detail="No autorizado")

    if not UNSPLASH_ACCESS_KEY:
        return {"error": "UNSPLASH_ACCESS_KEY no configurada en este servidor"}

    lote = LISTA_SIMBOLOS_250[inicio : inicio + cantidad]

    utm = "?utm_source=Dreamtionary&utm_medium=referral"
    resultados = {}
    errores = []

    async with httpx.AsyncClient() as http_client:
        for item in lote:
            try:
                respuesta = await http_client.get(
                    "https://api.unsplash.com/search/photos",
                    params={"query": item["en"], "per_page": 1, "orientation": "landscape"},
                    headers={"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"},
                    timeout=15.0,
                )
                datos = respuesta.json()
                fotos = datos.get("results", [])
                if fotos:
                    foto = fotos[0]
                    resultados[item["key"]] = {
                        "url": foto["urls"]["regular"],
                        "urlThumb": foto["urls"]["small"],
                        "autor": foto["user"]["name"],
                        "autorUrl": foto["user"]["links"]["html"] + utm,
                        "unsplashUrl": "https://unsplash.com/" + utm,
                        "downloadLocation": foto["links"]["download_location"],
                    }
                else:
                    errores.append(item["key"])
            except Exception as e:
                errores.append(f"{item['key']} ({str(e)})")

            await asyncio.sleep(0.8)

    return {
        "lote_inicio": inicio,
        "lote_cantidad": cantidad,
        "total_en_lista": len(LISTA_SIMBOLOS_250),
        "total_procesados_en_este_lote": len(lote),
        "total_exitosos": len(resultados),
        "errores": errores,
        "siguiente_inicio": inicio + cantidad if inicio + cantidad < len(LISTA_SIMBOLOS_250) else None,
        "fotos": resultados,
    }
