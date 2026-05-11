import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str: 
    """
    Crea una descripción técnica y ultra-especializada para mejorar la recuperación
    de contenido multimodal en el dominio de la cultura.
    """
    try: 
        # Carga de configuración
        # Se asume que en el .env LLM=gemini-1.5-flash o gemini-2.5-flash-lite
        model_name = os.getenv("LLM")
        api_key = os.getenv("OPENAI_API_KEY")
        
        llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.2)
        
        # PROMPT ESPECIALIZADO
        prompt_instructions = """Actúa como un Especialista en Documentación Cultural. 
Tu tarea es generar un RESUMEN AMPLIADO de un fragmento de documento (chunk). Este fragmento contiene información mixta (texto, tablas e imágenes).

INSTRUCCIONES:
1. Integra toda la información: No resumas solo el texto; incluye los datos relevantes de las tablas y describe lo que se visualiza en las imágenes.
2. Cohesión: Redacta un texto continuo donde la información de las imágenes y tablas complemente al texto principal.
3. Precisión técnica: Mantén los nombres propios, fechas, términos artísticos y datos numéricos exactos.
4. Objetivo RAG: Asegúrate de que el resumen sea rico en palabras clave para que este fragmento sea fácilmente recuperable mediante búsquedas.

SALIDA ESPERADA:
Un resumen denso y detallado que consolide el contenido textual, visual y tabular del chunk.

CONTENIDO A PROCESAR:
"""
        
        content_to_analyze = f"{prompt_instructions}\n\nTEXTO DEL DOCUMENTO:\n{text}\n"
        
        if tables: 
            content_to_analyze += "\nTABLAS (HTML/Texto):\n"
            for i, table in enumerate(tables): 
                content_to_analyze += f"--- Tabla {i+1} ---\n{table}\n"
                
        message_content = [{"type": "text", "text": content_to_analyze}]
        
        # Inclusión de imágenes en base64 para el modelo multimodal
        for image_base64 in images: 
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
    
    except Exception as e: 
        print(f"Error generando descripción experta: {e}")
        # Fallback básico en caso de error de API
        summary = f"Fragmento técnico: {text[:200]}..."
        if tables: summary += f" | Tablas: {len(tables)}"
        if images: summary += f" | Imágenes: {len(images)}"
        return summary