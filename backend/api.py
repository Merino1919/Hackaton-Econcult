import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# Importar tu motor RAG
from backend.RAG.engine import RAGEngine

# Importar script del recomendador de equipamientos
from backend.recomendadores.recomendador_equi import RecomendadorCultural, PerfilUsuario, RespuestaRecomendacion
from backend.recomendadores.recomendador_ubi import RecomendadorUbicacion, PerfilEquipamiento, RespuestaUbicacion

# Librerías noticias
from gnews import GNews
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()

# 1. Modelos de Pydantic para Validar Entradas y Salidas

# Reto 4: Catalogador de noticias
class SolicitudAnalisis(BaseModel):
    centro_cultural: str = Field(..., example="IVAM Valencia")

class ImpactoNoticia(BaseModel):
    fecha: str
    titular: str
    impactos: List[str]
    razonamiento: str

# Reto 7: RAG Multimodal
class RespuestaAnalisis(BaseModel):
    centro: str
    resultados: List[ImpactoNoticia]
    
class SolicitudRAG(BaseModel):
    pregunta: str = Field(..., example="¿Qué metodologías se usan para evaluar el arte contemporáneo?")

# 2. Inicializar la aplicación FastAPI y scripts necesarios
app = FastAPI(title="Super APP Inteligencia Cultural")

# --- INICIALIZAR EL RAG ---
rag_engine = RAGEngine()

# --- INICIALIZAR RECOMENDADORES ---
recomendador_engine = RecomendadorCultural(db_path="./data/BDD_equipamientos_v9.xlsx")
recomendador_ubicacion_engine = RecomendadorUbicacion(db_path="./data/BDD_territorios_v4.xlsx")

# 3. Configuración de endpoints: 
# --- ENDPOINT NOTICIAS --- 
dimensiones = {
    "D1": "Transformación del espacio y del territorio",
    "D2": "Generación de conocimiento y conexión entre expertos",
    "D3": "Notoriedad, marca territorial y soft power",
    "D4": "Influencia en la agenda política y opinión pública",
    "D5": "Articulación de redes y comunidades sociales",
    "D6": "Robustecimiento de sectores culturales",
    "D7": "Satisfacción de los derechos culturales"
}

# Configuración del Parser de Langchain
response_schemas = [
    ResponseSchema(name="dimensiones_detectadas", description="Lista de IDs (D1-D7) detectadas en la noticia"),
    ResponseSchema(name="justificacion", description="Breve explicación de por qué encaja en esas categorías")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = """
Analiza el siguiente titular de noticia sobre un centro cultural y clasifícalo según estas 7 dimensiones de impacto:
{dimensiones_info}

Noticia: {titular}

{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(template)

@app.post("/noticias/analizar", response_model=RespuestaAnalisis, tags=["Analizador de noticias"])
async def analizar_impactos_endpoint(solicitud: SolicitudAnalisis):
    # Validar que la API Key esté configurada
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Falta la API Key de OpenAI en las variables de entorno.")

    try:
        # Configurar GNews
        google_news = GNews(language='es', country='ES', max_results=10)
        
        # Buscar noticias
        noticias_raw = google_news.get_news(solicitud.centro_cultural)
        
        if not noticias_raw:
            return RespuestaAnalisis(centro=solicitud.centro_cultural, resultados=[])

        # Inicializar el modelo
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        resultados_finales = []
        format_instructions = output_parser.get_format_instructions()

        for entry in noticias_raw:
            # GNews devuelve diccionarios, así que extraemos los valores con .get()
            titulo = entry.get('title', 'Sin título')
            fecha = entry.get('published date', 'Fecha desconocida')

            input_data = prompt.format_messages(
                dimensiones_info=str(dimensiones),
                titular=titulo,
                format_instructions=format_instructions
            )
            
            response = llm.invoke(input_data)
            analisis = output_parser.parse(response.content)
            
            # Asegurarnos de que dimensiones_detectadas sea siempre una lista
            impactos_detectados = analisis.get('dimensiones_detectadas', [])
            if isinstance(impactos_detectados, str):
                impactos_detectados = [impactos_detectados]

            resultados_finales.append(
                ImpactoNoticia(
                    fecha=fecha,
                    titular=titulo,
                    impactos=impactos_detectados,
                    razonamiento=analisis.get('justificacion', 'Sin justificación')
                )
            )

        return RespuestaAnalisis(centro=solicitud.centro_cultural, resultados=resultados_finales)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el análisis: {str(e)}")
    

# --- ENDPOINT RECOMENDADOR EQUIPAMIENTOS ---
@app.post("/recomendar/equipamiento", response_model=RespuestaRecomendacion, tags=["Recomendador de equipamientos"])
async def recomendar_equipamientos(perfil: PerfilUsuario):
    """
    Recibe el perfil del usuario y devuelve los equipamientos culturales
    más similares utilizando similitud del coseno.
    """
    try:
        resultado = recomendador_engine.recomendar(perfil)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT RECOMENDADOR UBICACION ---
@app.post("/recomendar/ubicacion", response_model=RespuestaUbicacion, tags=["Recomendador de ubicaciones"])
async def recomendar_ubicaciones(perfil: PerfilEquipamiento):
    """
    Recibe las características del equipamiento y devuelve los territorios más similares.
    """
    try:
        resultado = recomendador_ubicacion_engine.recomendar(perfil)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINTS RAG ---
@app.post("/rag/upload", tags=["RAG Multimodal"])
async def upload_pdf(file: UploadFile = File(...)):
    """Recibe un PDF, lo guarda temporalmente y lo inyecta en el RAG."""
    os.makedirs("./data/uploads", exist_ok=True)
    file_path = f"./data/uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Ejecutar tu pipeline de ingesta
        categorias = rag_engine.run_complete_ingestion_pipeline(file_path)
        # Limpiar el archivo tras procesarlo
        os.remove(file_path)
        return {"mensaje": "Documento indexado con éxito", "categorias": categorias}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ask", tags=["RAG Multimodal"])
async def ask_rag(solicitud: SolicitudRAG):
    """Consulta a la base de datos vectorial multimodal."""
    try:
        respuesta = rag_engine.get_response_with_score(solicitud.pregunta)
        return respuesta
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))