# packages/vector_store_handler.py
import shutil
# NEU: Union und Literal importieren
from typing import List, Tuple, Union, Literal
from packages.bm25_retriever import BM25Retriever

from langchain.schema.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings # Korrekte Imports prüfen
# from langchain_community.document_loaders import PyPDFDirectoryLoader # Nicht verwendet?
from langchain.schema.vectorstore import VectorStoreRetriever
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Nicht hier benötigt
from packages.globals import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_PATH,
    SEARCH_TYPE,
    SEARCH_KWARGS_NUM,
    EMBEDDINGS,
)
# NEU: Importiere den SplitterType aus document_processing
from packages.document_processing import SplitterType
import os # NEU: Für os.path.join
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class VectorStoreHandler:
    """
    Handles the creation, loading, and deletion of vector stores (FAISS) 
    based on configuration including splitter type.
    """
    def __init__(
        self,
        embeddings = EMBEDDINGS, # Default aus Globals
        # Defaults für chunk_size/overlap etc. werden jetzt eher in den Methoden geholt oder übergeben
        search_type: str = SEARCH_TYPE,
        search_kwargs_num: int = SEARCH_KWARGS_NUM,
    ):
        self.embeddings = embeddings
        self.search_type = search_type
        self.search_kwargs_num = search_kwargs_num
        import os
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Chunk Parameter sind jetzt spezifisch für den Store, nicht für den Handler
        # self.chunk_size = chunk_size
        # self.chunk_overlap = chunk_overlap

    def _get_embedding_name(self) -> str:
        """Gets a string representation of the embedding class name."""
        # Sicherstellen, dass self.embeddings nicht None ist
        if self.embeddings and hasattr(self.embeddings, '__class__'):
             return self.embeddings.__class__.__name__
        return "UnknownEmbeddings" # Fallback

    # GEÄNDERT: Methode zur Pfad-Generierung
    def _get_vector_store_path(
        self,
        vector_store_name: str,
        splitter_type: SplitterType,
        chunk_size: int, # Explizite Parameter hier
        chunk_overlap: int # Explizite Parameter hier
    ) -> str:
        """Constructs a unique path for the vector store based on config."""
        # Baut den Pfad: ./database/EmbeddingName_PersonName/SplitterType_ChunkSize_ChunkOverlap
        # z.B. ./database/HuggingFaceEmbeddings_Albert Einstein/semantic_1000_0
        # (ChunkSize/Overlap bei semantic evtl. 0 oder Standardwert nehmen, da irrelevant)
        base_path = VECTOR_STORE_PATH
        #folder_name = f"{self._get_embedding_name()}_{vector_store_name}"
        # Bei semantic sind chunk_size/overlap nicht relevant, setze sie ggf. auf 0 im Pfad für Konsistenz
        #if splitter_type == "semantic":
            #config_name = f"{splitter_type}_semantic" # Eindeutiger Name ohne size/overlap
        #else:
            #config_name = f"{splitter_type}_{chunk_size}_{chunk_overlap}"
        #config_name = f"{splitter_type}_{chunk_size}_{chunk_overlap}"
        # os.path.join für plattformunabhängige Pfade verwenden
        #return os.path.join(base_path, folder_name, config_name)
        if splitter_type == "semantic":  # Überschreibt Parameter für semantic
            chunk_size, chunk_overlap = 0, 0
        return os.path.join(base_path, f"{splitter_type}_{chunk_size}_{chunk_overlap}")

    def _create_vector_store(
        self,
        split_documents: List[Document], # Das sind die Chunks!
        vector_store_name: str,
        # NEU: Parameter müssen hier übergeben werden
        splitter_type: SplitterType,
        chunk_size: int,
        chunk_overlap: int,
    ) -> FAISS:
        """Creates and saves a FAISS vector store."""
        db_path = self._get_vector_store_path(
            vector_store_name, splitter_type, chunk_size, chunk_overlap
        )
        # Verwende die Parameter für die Log-Ausgabe
        print(
            f"[INFO] Creating vector store for '{vector_store_name}' with splitter '{splitter_type}' "
            f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap}) at path: {db_path}"
        )
        
        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        try:
            db = FAISS.from_documents(documents=split_documents, embedding=self.embeddings)
            db.save_local(folder_path=db_path) # save_local erwartet folder_path
            print(f"[INFO] Vector store saved successfully to {db_path}")
            return db
        except Exception as e:
            print(f"[ERROR] Failed to create or save vector store at {db_path}: {e}")
            return None


    def _get_existing_vector_store(
        self,
        vector_store_name: str,
        # NEU: Parameter müssen hier übergeben werden
        splitter_type: SplitterType,
        chunk_size: int,
        chunk_overlap: int
    ) -> FAISS:
        """Loads an existing FAISS vector store."""
        db_path = self._get_vector_store_path(
            vector_store_name, splitter_type, chunk_size, chunk_overlap
        )
        print(f"[INFO] Attempting to load existing vector store from: {db_path}")
        
        # Prüfen ob Pfad existiert, bevor geladen wird
        if not os.path.isdir(db_path):
             print(f"[INFO] Directory not found, cannot load store: {db_path}")
             return None
             
        try:
            # allow_dangerous_deserialization ist wichtig!
            db = FAISS.load_local(
                folder_path=db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[INFO] Successfully loaded vector store from {db_path}")
            return db
        except Exception as e:
            # Spezifischere Fehlermeldung, wenn Laden fehlschlägt
            print(f"[ERROR] Failed to load vector store from {db_path}: {e}")
            return None


    def _create_retriever(self, db: FAISS) -> VectorStoreRetriever:
        """Creates a retriever from the vector store."""
        if db is None:
            print("[ERROR] Cannot create retriever because database object is None.")
            return None
        print(f"[INFO] Creating Retriever (search_type: {self.search_type}, k: {self.search_kwargs_num})")
        return db.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": self.search_kwargs_num}
        )

    # GEÄNDERT: Nimmt jetzt alle Config-Parameter entgegen
    def get_db_and_retriever(
        self,
        vector_store_name: str,
        splitter_type: SplitterType,
        chunk_size: int,
        chunk_overlap: int
    ) -> Tuple[VectorStoreRetriever | None, FAISS | None]: # Union für Optionale Rückgabe
        """Tries to load an existing vector store and its retriever."""
        print(
            f"[INFO] Getting DB and Retriever for config: "
            f"Name='{vector_store_name}', Splitter='{splitter_type}', "
            f"ChunkSize={chunk_size}, ChunkOverlap={chunk_overlap}, "
            f"Embedding='{self._get_embedding_name()}'"
        )
        db = self._get_existing_vector_store(
            vector_store_name=vector_store_name,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if db:
            retriever = self._create_retriever(db=db)
            return retriever, db
        else:
            # Explizit loggen, dass nichts gefunden/geladen wurde
            print(f"[INFO] Existing vector store not found or failed to load for the specified config.")
            return None, None

    # GEÄNDERT: Nimmt jetzt alle Config-Parameter entgegen
    def create_db_and_retriever(
        self,
        # HINWEIS: 'documents' müssen bereits die gechunkten Dokumente sein!
        chunked_documents: List[Document],
        vector_store_name: str,
        splitter_type: SplitterType,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Tuple[VectorStoreRetriever | None, FAISS | None]:
        """Creates a new vector store and its retriever."""
        print(f"[INFO] Creating NEW DB and Retriever for config: Name='{vector_store_name}', Splitter='{splitter_type}', ChunkSize={chunk_size}, ChunkOverlap={chunk_overlap}")
        db = self._create_vector_store(
            split_documents=chunked_documents,
            vector_store_name=vector_store_name,
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if db:
            retriever = self._create_retriever(db=db)
            return retriever, db
        else:
            print(f"[ERROR] Failed to create vector store, cannot create retriever.")
            return None, None

    # GEÄNDERT: Nimmt jetzt alle Config-Parameter entgegen
    def delete_db(
        self,
        vector_store_name: str,
        splitter_type: SplitterType,
        chunk_size: int,
        chunk_overlap: int
    ):
        """Deletes a specific vector store directory."""
        db_path = self._get_vector_store_path(
            vector_store_name, splitter_type, chunk_size, chunk_overlap
        )
        print(f"[INFO] Attempting to delete vector store directory: {db_path}")
        if os.path.isdir(db_path):
             try:
                  shutil.rmtree(db_path)
                  print(f"[INFO] Successfully deleted directory: {db_path}")
             except OSError as e:
                  print(f"[ERROR] Error deleting directory {db_path}: {e}")
        else:
             print(f"[INFO] Directory not found, nothing to delete: {db_path}")

    def get_hybrid_retriever(
        self,
        vector_store_name: str,
        splitter_type: SplitterType,
        chunk_size: int,
        chunk_overlap: int,
        bm25_index_dir: str,
        alpha: float = 0.5,
):
        """Gibt HybridRetriever + FAISS-DB zurück."""
    # 1) Dense-Teil laden
        dense_retriever, db = self.get_db_and_retriever(
            vector_store_name, splitter_type, chunk_size, chunk_overlap
    )
    # 2) BM25-Teil laden
        bm25 = BM25Retriever(bm25_index_dir)
    # 3) Hybrid-Wrapper konstruieren
        hybrid = HybridRetriever(
            db = db,
            bm25_retriever=bm25,
            alpha=alpha,
            k=self.search_kwargs_num,
    )
        return hybrid, db
    
class HybridRetriever:
        def __init__(self, db: FAISS, bm25_retriever, alpha: float = 0.5, k: int = 3,
                 dense_k: int | None = None, bm25_k: int | None = None):
            
            self.db = db
            self.bm25_retriever = bm25_retriever
            self.alpha = alpha
            self.k = k
            # optional: getrennte Kandidatenmengen
            self.dense_k = int(dense_k) if dense_k is not None else max(2 * self.k, self.k)
            self.bm25_k = int(bm25_k) if bm25_k is not None else max(3 * self.k, self.k)

        @staticmethod
        def _squash_bm25(bscore: float) -> float:
            # Monotone, robuste Squash-Funktion für unskalierte BM25-Scores
            return 1.0 - 1.0 / (1.0 + max(bscore, 0.0))   

        def set_k_init(self, k_init: int):
            self.k = int(k_init)
            self.dense_k = max(2 * k_init, k_init)
            self.bm25_k = max(3 * k_init, k_init) 
        
        def get_relevant_documents(self, query: str):
            b_scores = self.bm25_retriever.get_scores(query, self.bm25_k)
            dense_docs_with_scores = self.db.similarity_search_with_relevance_scores(query, k=self.dense_k)
            
            fused: list[tuple[Document, float]] = []
            miss_map = 0
            for doc, dscore in dense_docs_with_scores:
                doc_id = doc.metadata.get('source', None)
                if doc_id is None:
                    miss_map += 1
                    bscore = 0.0
                else:
                    bscore = b_scores.get(doc_id, 0.0)

                bscore_norm = self._squash_bm25(bscore)
                fused_score = self.alpha * bscore_norm + (1.0 - self.alpha) * float(dscore)
                fused.append((doc, fused_score))
   
            fused.sort(key=lambda x: x[1], reverse=True)
            topk = fused[: self.k]

            # Nur bei Problemen debuggen (alle Debug-Statements zusammen)
            if miss_map > 0:
                print(f"[DEBUG][HybridRetriever] {miss_map}/{len(dense_docs_with_scores)} docs had no matching BM25 id.")
    
                total = len(dense_docs_with_scores)
                matches = sum(1 for (doc, _) in dense_docs_with_scores if doc.metadata.get('source') in b_scores)
                print(f"[DEBUG] BM25 match rate: {matches}/{total}")
    
    # Zusätzliche Debug-Info nur bei Problemen
                if b_scores and dense_docs_with_scores:  # Safety check
                    print("BM25 doc counts:", len(b_scores))
                    print("FAISS doc counts:", len(dense_docs_with_scores))
                    print("Sample BM25 ID:", list(b_scores.keys())[0])
                    print("Sample FAISS ID:", dense_docs_with_scores[0][0].metadata.get('source', 'N/A'))

            return topk
