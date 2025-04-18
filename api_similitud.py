import re
import logging
from functools import lru_cache
from typing import Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


# --------------------------------------------------
# Configuration & Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CORS origins (configurable)
origins = [
    "http://localhost",
    "http://localhost:3000",
]

# --------------------------------------------------
# App Initialization
# --------------------------------------------------
# api_similitud.py

app = FastAPI(
    title="API de Similitud de Títulos",
    description="Compara dos títulos y retorna un score de similitud semántica",
    version="1.0.0",
    debug=True                      
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Models
# --------------------------------------------------
class TituloPair(BaseModel):
    titulo_db: str = Field(..., min_length=1, example="Sistema de gestión de clientes")
    titulo_input: str = Field(..., min_length=1, example="Aplicación para administración de usuarios")


class SimilitudResponse(BaseModel):
    similitud: float
    clasificacion: str
    titulo_db_normalizado: str
    titulo_input_normalizado: str


# --------------------------------------------------
# Utilities
# --------------------------------------------------
@lru_cache()
def get_model() -> SentenceTransformer:
    """
    Carga y cachea el modelo de sentence-transformers
    """
    logger.info("Cargando modelo paraphrase-multilingual-MiniLM-L12-v2...")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


equivalencias: Dict[str, List[str]] = {
    "sistema": ["software", "aplicación", "herramienta", "plataforma", "entorno"],
    "gestión": ["administración", "control", "organización", "seguimiento", "monitoreo"],
    # Agrega más equivalencias según necesidad
}
# Precompile regex patterns for performance
compiled_patterns = []
for base, synonyms in equivalencias.items():
    for synonym in synonyms:
        compiled_patterns.append((re.compile(rf"\b{re.escape(synonym)}\b", re.IGNORECASE), base))


def normalizar_titulo(texto: str) -> str:
    """
    Reemplaza sinónimos en el texto por su palabra base definida en equivalencias.
    """
    texto_normalizado = texto.strip()
    for pattern, base in compiled_patterns:
        texto_normalizado = pattern.sub(base, texto_normalizado)
    return texto_normalizado.lower()


def clasificar_similitud(score: float) -> str:
    """
    Clasifica el score de similitud en categorías descriptivas.
    """
    if score >= 90:
        return "✨ Casi idéntico"
    if score >= 80:
        return "✅ Muy similar"
    if score >= 70:
        return "⚠️ Similar pero diferente"
    return "❌ No similar"


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.post(
    "/comparar",
    response_model=SimilitudResponse,
    status_code=status.HTTP_200_OK,
    summary="Compara dos títulos",
    tags=["Similitud"]
)
async def comparar_titulos(pair: TituloPair):
    """
    Endpoint para comparar dos títulos y retornar 
    la similitud semántica en porcentaje y una clasificación.
    """
    try:
        model = get_model()

        t1 = normalizar_titulo(pair.titulo_db)
        t2 = normalizar_titulo(pair.titulo_input)
        logger.debug(f"Títulos normalizados: DB='{t1}' INPUT='{t2}'")

        vec1 = model.encode(t1, convert_to_tensor=True)
        vec2 = model.encode(t2, convert_to_tensor=True)

        similitud_score = util.cos_sim(vec1, vec2).item() * 100
        similitud_rounded = round(similitud_score, 2)
        etiqueta = clasificar_similitud(similitud_rounded)

        logger.info(
            f"Similitud calculada: {similitud_rounded}% - Clasificación: {etiqueta}"
        )

        if similitud_rounded >= 60:
            return SimilitudResponse(
                similitud=similitud_rounded,
                clasificacion=etiqueta,
                titulo_db_normalizado=t1,
                titulo_input_normalizado=t2,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Similitud baja, no se retorna el resultado"
            )

    except Exception as exc:
        logger.error(f"Error al comparar títulos: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al calcular similitud"
        )


# --------------------------------------------------
# Fragmento para run de local python api_similitud.py en terminal
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_similitud:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
