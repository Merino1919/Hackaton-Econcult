import os
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from unstructured.chunking.title import chunk_by_title

from backend.RAG.utils.parsers import select_loader, separate_content_types
from backend.RAG.src.ai_summary import create_ai_enhanced_summary

load_dotenv()

class RAGEngine:
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL")
        self.text_embedder = OpenAIEmbeddings(model=self.embedding_model_name, api_key=self.api_key)
        self.llm = ChatOpenAI(model=os.getenv("LLM"), api_key=self.api_key)
        self.vector_store = self.create_vector_store()

    def create_vector_store(self):
        
        CHROMA_PATH = os.getenv("CHROMA_PATH")
        COLLECTION_NAME = os.getenv("COLLECTION_NAME")
                
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.text_embedder,
            collection_name=COLLECTION_NAME
        )
    

    def summarise_chunks(self, chunks): 
        """Procesa todos los chunks con resúmenes IA"""
        print("Procesando chunks con resúmenes IA... ")
    
        langchain_documents = []
        total_chunks = len(chunks)
    
        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            print(f" Procesando chunk {current_chunk} / {total_chunks}")
            
            # Analyze chunk content
            content_data = separate_content_types(chunk)
            
            # Debug prints
            print(f"    Tipos encontrados: {content_data['types']}")
            print(f"    Tablas: {len(content_data['tables'])}, Imágenes: {len(content_data['images'])}")
            
            # Create AI-Enhanced summary if chunk has tables/images
            if content_data['tables'] or content_data['images']: 
                print(f" Creando resumen con IA para contenido mixto...")
                
                try: 
                    enhanced_content = create_ai_enhanced_summary(
                        content_data['text'],
                        content_data['tables'],
                        content_data['images']
                    )    
                    
                    print(f"    Resumen IA creado exitosamente")
                    print(f"    Contenido mejorado: {enhanced_content[:200]}")
                
                except Exception as e: 
                    print(f"     Resumen IA falló: {e}")
                    enhanced_content = content_data['text']
            
            else: 
                print(f"    Usando texto crudo (sin imágenes o tablas)")
                enhanced_content = content_data['text']
                    
            # Forzamos la conversión a string por si viene algún otro tipo de objeto
            enhanced_content = str(enhanced_content).strip()
            # -----------------------------------------------

            if enhanced_content: 
                
                doc = Document(
                    page_content=enhanced_content, 
                    metadata={
                        "original_content": json.dumps({
                            "raw_text": content_data['text'],
                            "tables_html": content_data['tables'],
                            "images_base64": content_data['images']  
                        })
                    }
                )
                langchain_documents.append(doc)
            
            else: 
                print(f"    [Skipped] El chunk {current_chunk} está vacío tras el procesado.")
            
            
        print(f"Processed {len(langchain_documents)} chunks")
        return langchain_documents

    def run_complete_ingestion_pipeline(self, file_path, status_callback = None):
        
        # Función auxiliar para actualizar el estado si existe el callback
        def update_status(msg):
            if status_callback:
                status_callback(msg)
        
        # Step 1: Partition
        update_status("🔍 Parseando estructura del archivo...")
        elements = select_loader(file_path)
        
        # Step 2: Chunking
        update_status("✂️ Dividiendo el contenido en chunks...")
        chunks = chunk_by_title(elements, max_characters=3000, new_after_n_chars=2400, combine_text_under_n_chars=500)
        
        all_categories = []
        for chunk in chunks:
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                for el in chunk.metadata.orig_elements:
                    # 'el' is a Unstructuredobject, we will get the type
                    all_categories.append(el.to_dict().get("type"))
        
        # Step 3: AI Summarisation
        update_status("🤖 Generando resúmenes IA para tablas e imágenes...")
        summarised_chunks = self.summarise_chunks(chunks)
        
        # Step 4: Ingest documents into vector database
        update_status("💾 Indexando en la BDD Vectorial (ChromaDB)...")
        if summarised_chunks:
            total = len(summarised_chunks)
            exitosos = 0
            
            # Ingestamos documento a documento para aislar cualquier fallo de la API
            for i, doc in enumerate(summarised_chunks):
                try:
                    self.vector_store.add_documents(documents=[doc])
                    exitosos += 1
                except Exception as e:
                    print(f"    ⚠️ [Error] No se pudo crear el embedding para el chunk {i}: {e}")
            
            update_status(f"✅ Se han indexado {exitosos} de {total} chunks correctamente.")
        
        return all_categories
    
    def get_response_with_score(self, query):
        """Retrieve multimodal context with similarity score and generates the response."""
    
        # 1. Similarity search (Get the 5 closest chunks)
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=5)
        
        if not docs_and_scores:
            return {
                    "answer": "No hay información relevante en los documentos",
                    "best_chunk": "N/A",
                    "chunk_id": "N/A",  
                    "score": 0
                }
        
        # 2. Prepare the metadata
        prompt_context = ""
        message_content = []
        
        for i, (doc, score) in enumerate(docs_and_scores):
            prompt_context += f"--- Documento {i + 1} (Score: {round(score, 4)}) ---\n"
            
            # Extract the content of the metadata (JSON)
            if "original_content" in doc.metadata:
                try:
                    original_data = json.loads(doc.metadata["original_content"])
                    
                    # Extract text
                    raw_text = original_data.get("raw_text", "")
                    if raw_text:
                        prompt_context += f"TEXT:\n{raw_text}\n"
                    
                    # Extract tables
                    tables_html = original_data.get("tables_html", [])
                    if tables_html:
                        prompt_context += "TABLES:\n"
                        for j, table in enumerate(tables_html):
                            prompt_context += f"Table {j+1}:\n{table}\n"
                    
                    # Extract images 
                    images_base64 = original_data.get("images_base64", [])
                    for img_b64 in images_base64:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        })
                except Exception as e:
                    # If JSON fails, we use the basic page_content
                    prompt_context += f"CONTENT:\n{doc.page_content}\n"
            else:
                # If there's no complex metadata, we use the standard content
                prompt_context += f"CONTENT:\n{doc.page_content}\n"
            
            prompt_context += "\n"

        # 3. Final prompt building
        full_prompt_text = f"""Actúa como un experto en documentación cultural y papers artísticos. Utiliza el siguiente contexto (que incluye texto, tablas e imágenes) para responder a la pregunta.
        Si no encuentras la respuesta a la consulta del usuario, sé humilde y di que no la sabes.

        Contexto:
        {prompt_context}

        Pregunta: {query}
        Respuesta:"""

        # 4. Combine text and images using the format desired by the multimodal LLM.
        # Insert the text prompt at the begining of the content list.
        message_content.insert(0, {"type": "text", "text": full_prompt_text})
        
        # 5. Invoke
        try:
            message = HumanMessage(content=message_content)
            # Note: Ensure that this is a vision model
            response = self.llm.invoke([message])
            answer = response.content
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"

        # 6. Give back the best result metadata
        best_doc, best_score = docs_and_scores[0]
        chunk_id = docs_and_scores[0][0].id
        
        return {
            "answer": answer,
            "score": round(best_score, 4),
            "best_chunk": best_doc.page_content,
            "chunk_id": chunk_id
        }

    # --- ANALYSIS FUNCTIONALITIES AND CHUNKS MANAGEMENT ---
    
    def get_document_count(self):
        """Give back the total number of indexed chunks."""
        try:
            if self.vector_store:
                return self.vector_store._collection.count()
            return 0
        except Exception as e:
            print(f"Error counting documents: {e}")
            return 0

    def get_all_chunks(self):
        """Retrieve all the chunks to be visualized"""
        try:
            return self.vector_store.get()
        except Exception as e:
            print(f"Error getting chunks: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def clear_collection(self):
        """Deletes the collection and create it again in a secure way."""
        try:
            
            CHROMA_PATH = os.getenv("CHROMA_PATH")
            COLLECTION_NAME = os.getenv("COLLECTION_NAME")
            
            # 1. Trying to delete the collection using the official method of LangChain.
            self.vector_store.delete_collection()
                
            # 2. Re-initialize the object in order to let the system ready for new documents. 
            self.vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PATH,
                embedding_function=self.text_embedder)
                
            # Line for debugging to check the count after deleting. 
            count = self.vector_store._collection.count()
            print(f"DEBUG - Documents in the collection after deleting: {count}")
                
            return True
            
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
            
        
    def get_detailed_stats(self, chunks):
        """Calculate statistics over the chunks."""
        # Dictionary to count the categories.
        category_counts = {}
        
        for element in chunks: 
            
            try:
                # Trying to get the type from the Unstructured object
                category_dict = element.to_dict()
                tipo = category_dict.get("type", "Unknown")
            
            except AttributeError:
                # If it is a Langchain document: 
                tipo = element.metadata.get("type", "Table Record")
            
            # Sum to the count
            category_counts[tipo] = category_counts.get(tipo, 0) + 1
        
        # Give back the expected format
        return category_counts