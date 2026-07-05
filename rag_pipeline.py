import os
import json
import sys
import time
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Import LangChain core components
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import LangChain Community vector store
from langchain_community.vectorstores import Chroma

# Import Google Gemini integrations
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# -----------------------------------------------------------------------------
# PHASE 1: ENVIRONMENT SETUP
# -----------------------------------------------------------------------------
# Load environment variables from a .env file if it exists.
# This is crucial for securing the GOOGLE_API_KEY.
load_dotenv()

# Verify that the Google API Key is set, otherwise notify the user.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("\n[WARNING] GOOGLE_API_KEY environment variable not found.")
    print("Please create a '.env' file in this directory and add:")
    print("GOOGLE_API_KEY=your_actual_api_key_here\n")


class RateLimitedEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    A rate-limited wrapper for GoogleGenerativeAIEmbeddings to prevent
    RESOURCE_EXHAUSTED errors on the Google Gemini Free Tier.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 2
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"Embedding batch {i // batch_size + 1} of {total_batches} (size {len(batch)})...")
            
            # Simple retry loop (up to 3 retries)
            for attempt in range(3):
                try:
                    batch_embeddings = super().embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if ("RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)) and attempt < 2:
                        sleep_time = 35 * (attempt + 1)
                        print(f"Rate limit hit. Sleeping for {sleep_time}s before retry (attempt {attempt + 1}/3)...")
                        time.sleep(sleep_time)
                    else:
                        raise e
            
            # Sleep between batches to stay under free tier rate limits (RPM and TPM)
            if i + batch_size < len(texts):
                time.sleep(11)
        return embeddings


def initialize_rag_components():
    """
    Initializes the embedding model and LLM using Google Gemini API.
    """
    print("Initializing Google Gemini API components...")
    
    # 1. Initialize the embedding model.
    # We use 'models/gemini-embedding-2' as specified.
    # This model maps textual content to high-dimensional dense vectors representing semantic meaning.
    embeddings = RateLimitedEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=GOOGLE_API_KEY
    )
    
    # 2. Initialize the Chat LLM.
    # We use 'gemini-2.5-flash', which offers a great balance of speed, cost, and high performance.
    # Setting temperature to 0.0 ensures highly deterministic and precise answers.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )
    
    return embeddings, llm


# -----------------------------------------------------------------------------
# PHASE 2: INGESTION PHASE
# -----------------------------------------------------------------------------
def extract_text_from_pdf(file_path: Path) -> str:
    """Extracts text content from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        print(f"Warning: Error reading PDF {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path: Path) -> str:
    """Extracts text content from a Word (.docx) file using python-docx."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = []
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                # Deduplicate cells for merged cells
                row_cells = []
                for cell in row.cells:
                    if not row_cells or row_cells[-1] is not cell:
                        row_cells.append(cell)
                row_text = [cell.text.strip() for cell in row_cells if cell.text.strip()]
                if row_text:
                    text.append(" | ".join(row_text))
        return "\n".join(text)
    except Exception as e:
        print(f"Warning: Error reading Word document {file_path}: {e}")
        return ""


def ingest_all_data_sources(json_path: str, data_dir_path: str, persist_directory: str, embeddings) -> Chroma:
    """
    Loads documents from both the JSON dataset and any supported files within
    the data directory (text, markdown, PDF, and Word document formats), chunks them,
    computes embeddings, and stores them in ChromaDB.
    
    Args:
        json_path (str): Path to the source JSON dataset file.
        data_dir_path (str): Path to the data directory containing source files.
        persist_directory (str): Directory where ChromaDB will store index files.
        embeddings: The embedding model instance.
        
    Returns:
        Chroma: An initialized and populated vector store.
    """
    documents = []
    
    # 1. Load JSON file contents if it exists
    if os.path.exists(json_path):
        print(f"Loading dataset from JSON: {json_path}...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            for index, item in enumerate(dataset):
                text_content = ""
                metadata = {"source": json_path, "item_index": index}
                
                if isinstance(item, dict):
                    # Check for standard 'text' field
                    if "text" in item:
                        text_content = item["text"]
                        if "title" in item:
                            metadata["title"] = item["title"]
                    # Check for QA format (question/answer pairs)
                    elif "question" in item and "answer" in item:
                        text_content = f"Question: {item['question']}\nAnswer: {item['answer']}"
                    else:
                        # Fallback: concatenate all string field values in the dict
                        text_content = "\n".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, str))
                    
                    # Incorporate other keys as metadata (excluding main text keys)
                    for k, v in item.items():
                        if k not in ["text", "question", "answer"] and isinstance(v, (str, int, float, bool)):
                            metadata[k] = v
                else:
                    text_content = str(item)
                    
                if text_content.strip():
                    documents.append(Document(page_content=text_content, metadata=metadata))
            print(f"Loaded {len(documents)} raw document(s) from JSON.")
        except Exception as e:
            print(f"Error loading JSON dataset: {e}")

    # 2. Load files from raw files directory
    data_dir = Path(data_dir_path)
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        # Create a sample file inside data_dir to show how it works
        sample_file = data_dir / "sample_note.txt"
        sample_file.write_text(
            "This is a sample note inside the data folder. "
            "You can place text (.txt), markdown (.md), PDF (.pdf), or Word (.docx) documents here, "
            "and the RAG pipeline will automatically load and index them next time it runs.",
            encoding="utf-8"
        )
        print(f"Created data directory and sample file at: {sample_file}")

    print(f"Loading custom files (text/markdown/pdf/docx) from: {data_dir_path}...")
    dir_docs = []
    # Supported file extensions
    extensions = {".txt", ".md", ".jsonld", ".csv", ".pdf", ".docx"}
    for file_path in data_dir.rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix in extensions:
                try:
                    content = ""
                    if suffix == ".pdf":
                        content = extract_text_from_pdf(file_path)
                    elif suffix == ".docx":
                        content = extract_text_from_docx(file_path)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    
                    if content.strip():
                        dir_docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "title": file_path.name,
                                "file_type": suffix
                            }
                        ))
                except Exception as e:
                    print(f"Warning: Could not read file {file_path}: {e}")
            elif suffix == ".doc":
                print(f"Warning: Legacy Word document (.doc) found at '{file_path}'. Please convert it to '.docx' to enable parsing.")
                
    print(f"Loaded {len(dir_docs)} file(s) from '{data_dir_path}'.")
    documents.extend(dir_docs)

    if not documents:
        raise ValueError("No documents found in JSON or directory to ingest.")

    # 3. Chunking (Text Splitting)
    # Split text to fit context windows and improve retrieval granularity.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=800,
        length_function=len
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(split_docs)} chunks.")

    # 4. Storage & Vector Indexing (ChromaDB)
    # Clear existing db directory to prevent accumulating stale chunks
    import shutil
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            print(f"Warning: Could not clean up old database directory: {e}")

    # Instantiate/populate Chroma database locally.
    print(f"Embedding and storing chunks in local ChromaDB at '{persist_directory}'...")
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("Ingestion complete and database saved.")
    return vector_store


# -----------------------------------------------------------------------------
# PHASE 3: RETRIEVAL & GENERATION (RAG CHAIN BUILDER)
# -----------------------------------------------------------------------------
def build_rag_chain(vector_store, llm):
    """
    Configures the search retrieval and connects it to the Gemini LLM
    using a strict context-adhering prompt format.
    
    Args:
        vector_store: The populated Chroma vector store.
        llm: The initialized Chat LLM.
        
    Returns:
        Chain: The executable RAG retrieval chain.
    """
    # 1. Set up the retriever.
    # search_kwargs={"k": 3} retrieves the top 3 most semantically similar chunks.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # 2. Design system prompt.
    # Instructs the LLM to act as a highly precise assistant that answers ONLY using context.
    system_prompt = (
        "You are a highly precise assistant tasked with answering questions "
        "based ONLY on the provided context.\n"
        "If you do not know the answer or if the context does not contain "
        "the answer, you must state that you do not know. Do not make up "
        "information or extrapolate beyond the context under any circumstances.\n\n"
        "Context:\n"
        "{context}"
    )
    
    # Define a prompt template incorporating system instructions and user query input.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 3. Create the document parsing chain.
    # 'create_stuff_documents_chain' takes retrieved documents, formats them into the prompt,
    # and passes it directly to the LLM.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 4. Create the final retrieval chain.
    # 'create_retrieval_chain' links the retriever with our document chain.
    # Input key: 'input' -> Retrieval -> Context retrieved -> formatted into prompt -> LLM runs -> Output key: 'answer'
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain


# -----------------------------------------------------------------------------
# PHASE 4: COMMAND-LINE INTERFACE LOOP
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("      Gemini-powered LangChain RAG CLI Pipeline")
    print("=" * 60)
    
    # Path configuration
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "dataset.json"
    data_dir_path = base_dir / "data"
    db_path = base_dir / "chroma_db"
    
    # Verify API key is configured before starting the loop
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n[ERROR] GOOGLE_API_KEY is not set.")
        print("Please check your .env file or environment variables, set it, and try again.")
        sys.exit(1)

    try:
        embeddings, llm = initialize_rag_components()
    except Exception as e:
        print(f"Error initializing Gemini components: {e}")
        sys.exit(1)

    # Ingest from both JSON and data/ directory sources
    try:
        vector_store = ingest_all_data_sources(
            json_path=str(dataset_path),
            data_dir_path=str(data_dir_path),
            persist_directory=str(db_path),
            embeddings=embeddings
        )
    except Exception as e:
        print(f"Error during ingestion: {e}")
        print("Trying to load existing vector store as fallback...")
        try:
            vector_store = Chroma(
                persist_directory=str(db_path),
                embedding_function=embeddings
            )
        except Exception as e_load:
            print(f"Could not load vector store: {e_load}")
            sys.exit(1)

    # Build RAG chain
    rag_chain = build_rag_chain(vector_store, llm)
    
    print("\nRAG Pipeline loaded successfully. Type your questions below.")
    print("Type 'exit' to quit the application.\n")
    print("-" * 60)

    # CLI Loop
    while True:
        try:
            user_query = input("\nAsk a question: ").strip()
            
            # Exit condition
            if user_query.lower() == 'exit':
                print("Exiting pipeline. Goodbye!")
                break
                
            if not user_query:
                continue
                
            print("Retrieving context and generating answer...")
            
            # Execute RAG query
            response = rag_chain.invoke({"input": user_query})
            
            # Print output details
            print("\n" + "=" * 40 + " ANSWER " + "=" * 40)
            print(response.get("answer", "No answer generated."))
            print("=" * 88)
            
            # Optional: print referenced sources/metadata
            context_docs = response.get("context", [])
            if context_docs:
                print("\n[Retrieved Context Sources]")
                for idx, doc in enumerate(context_docs, 1):
                    source = doc.metadata.get("title", f"Chunk {idx}")
                    print(f"  - {source} (Length: {len(doc.page_content)} characters)")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nExiting pipeline. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred while answering your query: {e}")
            print("Please ensure your Google Gemini API key is valid and has sufficient quota.")


if __name__ == "__main__":
    main()
