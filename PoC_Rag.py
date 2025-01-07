import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.llms import Ollama

import chromadb
from pypdf import PdfReader

# 1. CONFIGURACIÓN BÁSICA
PDF_PATH = "documentos/ejemplo.pdf"
CHROMA_DB_DIR = "chroma_db"  # Directorio donde Chroma guardará su índice
MODEL_NAME = "mistral"       # Nombre que usas en Ollama para el modelo. 
                             # Por ejemplo: 'mistral' -> si lo has registrado en Ollama con ese alias
                             # o "mistral-7B" según cómo lo tengas configurado.

# 2. LECTOR DE PDF Y EXTRACCIÓN DE TEXTO
def extraer_texto_de_pdf(pdf_path: str) -> str:
    """Lee todo el texto de un PDF y lo devuelve como un string."""
    reader = PdfReader(pdf_path)
    texto_completo = ""
    for page in reader.pages:
        texto_completo += page.extract_text() + "\n"
    return texto_completo

# 3. PROCESAMIENTO Y CHUNKEO DEL TEXTO
def crear_documentos(pdf_text: str, chunk_size=1000, overlap=100):
    """
    Crea una lista de Document (fragmentos de texto),
    cada uno con un tamaño máximo de chunk_size.
    """
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    chunks = splitter.split_text(pdf_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# 4. CREACIÓN/ACTUALIZACIÓN DE CHROMADB CON EMBEDDINGS
def crear_vectorstore(docs):
    """
    Crea (o actualiza) un store vectorial en ChromaDB para 
    poder buscar fragmentos más relevantes en base a embeddings.
    """
    # Usa un modelo de embeddings HF (puedes usar SentenceTransformers, etc.)
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Crea la base vectorial en un directorio local
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_DIR
    )
    # Asegura que se guarde el índice en disco
    vectordb.persist()
    return vectordb

# 5. FUNCIÓN PARA INICIALIZAR EL VECTORSTORE (O RECUPERARLO SI YA EXISTE)
def inicializar_vectorstore():
    """
    Si ya existe una base en CHROMA_DB_DIR, la carga.
    Si no, crea una nueva a partir del PDF.
    """
    if not os.path.exists(CHROMA_DB_DIR):
        os.mkdir(CHROMA_DB_DIR)

    # Verificamos si ya hay contenido en la carpeta
    if any(os.scandir(CHROMA_DB_DIR)):
        # Cargar la DB existente
        print("Cargando índice existente de ChromaDB...")
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(
            embedding_function=embedding_function,
            persist_directory=CHROMA_DB_DIR
        )
    else:
        # Crear la DB a partir del PDF
        print("No existe índice de ChromaDB. Creándolo desde el PDF...")
        pdf_text = extraer_texto_de_pdf(PDF_PATH)
        docs = crear_documentos(pdf_text)
        vectordb = crear_vectorstore(docs)
    
    return vectordb

# 6. CONFIGURAR EL MODELO VIA OLLAMA
def crear_llm_ollama(model_name: str):
    """
    Crea un LLM de Ollama apuntando a Mistral (u otro modelo local).
    Asegúrate de tener ollama instalado y el modelo cargado.
    """
    # LangChain tiene un wrapper para Ollama que podemos usar directamente
    # ¡Asegúrate de tener la versión más reciente de LangChain!
    llm = Ollama(
        model=model_name,
        # Parám en Ollama (tempratura, top_k, etc.) si quieres personalizar
        # Ejemplo:
        # base_url="http://localhost:11411",  # si tienes Ollama corriendo en otro puerto
        # timeout=300
    )
    return llm

# 7. CREAR LA CADENA DE RAG
def crear_cadena_rag(llm, vectordb):
    """
    Crea la cadena RetrievalQA que hace lo siguiente:
      - Toma la pregunta del usuario
      - Hace búsqueda de fragmentos relevantes en vectordb
      - Construye un prompt que combine la pregunta + contexto
      - Llama a Mistral (vía Ollama) para generar la respuesta
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combina el texto directamente. O 'map_reduce'/'refine'
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def main():
    # 1. Iniciar/Cargar el vectorstore
    vectordb = inicializar_vectorstore()
    
    # 2. Configurar el modelo LLM via Ollama (Mistral)
    llm = crear_llm_ollama(MODEL_NAME)
    
    # 3. Crear la cadena de QA con RAG
    qa_chain = crear_cadena_rag(llm, vectordb)
    
    # 4. "Simulación" de un bucle de preguntas del usuario
    while True:
        pregunta = input("\nIngresa tu pregunta (o escribe 'salir' para terminar): ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Saliendo...")
            break
        
        # Obtener respuesta
        respuesta = qa_chain(pregunta)
        
        # Muestra la respuesta final y las fuentes (opcional)
        print("\n--- Respuesta ---\n")
        print(respuesta["result"])
        
        # Si quieres ver qué fragmentos se usaron
        print("\n--- Fuentes usadas ---")
        for doc in respuesta["source_documents"]:
            print("Fragmento:", doc.page_content[:100], "...")
            # Podrías también mostrar la ruta del PDF si tuvieras metadatos

if __name__ == "__main__":
    main()
