# packages/document_processing.py
import os
# NEU: Literal für Typsicherheit bei Splitter-Auswahl
from typing import List, Literal
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# NEU: Weitere Splitter importieren
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
# NEU: Semantic Chunker (experimentell!)
from langchain_experimental.text_splitter import SemanticChunker
# NEU: Embeddings werden für Semantic Chunker benötigt
from langchain_community.embeddings import HuggingFaceEmbeddings
# NEU: Globals für Default-Werte und Embeddings importieren
from packages.globals import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS

# Typ für unterstützte Splitter definieren
SplitterType = Literal["recursive", "sentence_transformer", "semantic"]

class DocumentProcessing:
    """
    Loads documents from a directory and splits them into chunks 
    using various strategies.
    """
    # NEU: Nimmt Embedding-Instanz entgegen (wird für Semantic Chunker gebraucht)
    def __init__(self, embeddings = EMBEDDINGS):
        # Wir nehmen an, dass EMBEDDINGS meist HuggingFaceEmbeddings sind für BGE.
        self.embeddings = embeddings
        print(f"[INFO] DocumentProcessing initialized with embeddings type: {type(self.embeddings)}")

    def _load_documents(self, directory_path: str) -> List[Document]:
        documents: List[Document] = []
        print(f"[INFO] Loading documents from: {directory_path}")
        if not os.path.isdir(directory_path):
             print(f"[ERROR] Directory not found: {directory_path}")
             return documents
             
        for file in os.listdir(directory_path):
            file_path: str = os.path.join(directory_path, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith(".txt"):
                    # Annahme: UTF-8 Encoding ist sinnvoll
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
            except Exception as e:
                print(f"[ERROR] Failed to load document {file_path}: {e}")
        print(f"[INFO] Loaded {len(documents)} pages/documents.")
        return documents

    # Methode für Recursive Splitter (Baseline)
    def _split_recursive_down(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Splits documents using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len, # Standard: zählt Zeichen
            is_separator_regex=False,
        )
        print(f"[INFO] Splitting with RecursiveCharacterTextSplitter (size: {chunk_size}, overlap: {chunk_overlap})")
        return splitter.split_documents(documents)
    
    def _split_recursive(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Splits documents using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            length_function=len, # Standard: zählt Zeichen
            is_separator_regex=False,
        )
        print(f"[INFO] Splitting with RecursiveCharacterTextSplitter (size: {chunk_size}, overlap: {chunk_overlap})")
        
        base = splitter.split_documents(documents)
        stride = int(chunk_size * 0.8)
        overlapped = []

        for chunk in base:
            text = chunk.page_content
            for start in range(0, len(text), stride):
               window = text[start : start + chunk_size]
               overlapped.append(Document(page_content=window, metadata=chunk.metadata.copy()))
        return overlapped


    # NEU: Methode für Sentence Transformer Token Splitter
    def _split_sentence_transformer(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Splits documents based on tokens using SentenceTransformersTokenTextSplitter."""
        # Versucht, den Modellnamen aus der Embedding-Instanz zu holen
        # Setze einen sinnvollen Fallback, falls nicht vorhanden
        model_name = getattr(self.embeddings, 'model_name', 'BAAI/bge-small-en-v1.5')
        
        # Hinweis: chunk_size ist hier die Anzahl der Tokens! Ggf. kleiner wählen als bei Zeichen.
        print(f"[INFO] Splitting with SentenceTransformersTokenTextSplitter (tokens_per_chunk: {chunk_size}, chunk_overlap: {chunk_overlap} tokens, model: {model_name})")
        try:
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=chunk_overlap,
                model_name=model_name, # Wichtig: Muss zum Embedding passen
                tokens_per_chunk=chunk_size
            )
            # Dieser Splitter arbeitet mit Text, nicht direkt mit Document-Objekten.
            split_docs = []
            for doc in documents:
                 # Behalte Metadaten jedes Originaldokuments für die Chunks
                 text_chunks = splitter.split_text(doc.page_content)
                 for chunk in text_chunks:
                      # Erstelle neue Document-Objekte mit kopierten Metadaten
                      split_docs.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
            return split_docs
        except Exception as e:
            print(f"[ERROR] Error splitting with SentenceTransformersTokenTextSplitter: {e}")
            print("[WARN] Falling back to recursive splitting.")
            return self._split_recursive(documents, CHUNK_SIZE, CHUNK_OVERLAP) # Fallback


    # NEU: Methode für Semantic Chunker
    def _split_semantic(self, documents: List[Document]) -> List[Document]:
        """Splits documents semantically using SemanticChunker."""
        # Semantic Chunker benötigt eine Embedding-Instanz.
        if not hasattr(self.embeddings, 'embed_query'):
             print("[ERROR] Provided embeddings instance for Semantic Chunker is invalid (missing embed_query method).")
             print("[WARN] Falling back to recursive splitting.")
             return self._split_recursive(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        #  "percentile", "standard_deviation", "interquartile"
        threshold_type = "percentile" # Standard ist oft gut
        
        print(f"[INFO] Splitting with SemanticChunker (embeddings: {type(self.embeddings).__name__}, threshold: {threshold_type})")
        try:
            semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type=threshold_type
                # breakpoint_threshold_amount=0.95 # Kann man anpassen
            )
            return semantic_splitter.split_documents(documents)
        except Exception as e:
             print(f"[ERROR] Error splitting with SemanticChunker: {e}")
             print("[WARN] Falling back to recursive splitting.")
             return self._split_recursive(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # GEÄNDERT: Hauptmethode wählt Splitter basierend auf Parameter aus
    def get_chunked_documents(
        self,
        directory_path: str,
        # NEU: Parameter zur Auswahl der Strategie
        splitter_type: SplitterType,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[Document]:
        """Loads and chunks documents using the specified splitter type."""
        
        documents = self._load_documents(directory_path)
        split_documents = []


        if not documents:
             print("[WARN] No documents loaded, returning empty list.")
             return []

        print(f"[INFO] Chunking documents using strategy: {splitter_type}")
        if splitter_type == "recursive":
            # Nutzt chunk_size & chunk_overlap wie übergeben (Zeichen)
            # NOTE: _split_recursive_down = standard RecursiveCharacterTextSplitter
            #        _split_recursive = sliding window (deprecated, produced too many fragments)
            split_documents = self._split_recursive_down(documents, chunk_size, chunk_overlap)
        elif splitter_type == "sentence_transformer":
            # Nutzt chunk_size & chunk_overlap wie übergeben, aber interpretiert sie als TOKENS!
            # Ggf. müssen die Werte angepasst werden, wenn diese Methode aufgerufen wird.
            split_documents = self._split_sentence_transformer(documents, chunk_size, chunk_overlap)
        elif splitter_type == "semantic":
            # chunk_size und chunk_overlap werden hier ignoriert
            split_documents = self._split_semantic(documents)
        else:
            print(f"[WARN] Unknown splitter type '{splitter_type}'. Falling back to 'recursive'.")
            split_documents = self._split_recursive(documents, chunk_size, chunk_overlap)

        print(f"[INFO] Original document pages: {len(documents)}, Split into chunks: {len(split_documents)} (using {splitter_type})")
        return split_documents
    
    @staticmethod
    def _is_multi_hop(question: str) -> bool:
        keywords = ["explain", "compare", "because", "difference", "cause", "factors"]
        q = question.lower()
        return any(kw in q for kw in keywords)

