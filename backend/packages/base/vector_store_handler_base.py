import shutil
from typing import List, Tuple, Union
from langchain.schema.document import Document
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema.vectorstore import VectorStoreRetriever
from packages.globals import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_PATH,
    SEARCH_TYPE,
    SEARCH_KWARGS_NUM,
    EMBEDDINGS,
)


class VectorStoreHandler:
    """
    This class handles the creation of vector stores and retrievers.
    """

    def __init__(
        self,
        embeddings: Union[OpenAIEmbeddings, HuggingFaceEmbeddings] = EMBEDDINGS,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        search_type: str = SEARCH_TYPE,
        search_kwargs_num: int = SEARCH_KWARGS_NUM,
    ):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_type = search_type
        self.search_kwargs_num = search_kwargs_num

    def _get_embedding_name(self) -> str:
        return self.embeddings.__class__.__name__

    def _create_vector_store(
        self,
        split_documents: List[Document],
        vector_store_name: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> FAISS:
        """Creates a vector store from a list of documents."""

        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        print(
            f"[INFO] Creating vector store -> Name: {vector_store_name}, Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}"
        )
        db = FAISS.from_documents(documents=split_documents, embedding=self.embeddings)
        db.save_local(
            VECTOR_STORE_PATH
            + self._get_embedding_name()
            + "_"
            + vector_store_name
            + "/"
            + str(chunk_size)
            + "_"
            + str(chunk_overlap)
        )
        return db

    def _get_existing_vector_store(
        self, vector_store_name: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> FAISS:
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        print("jetzt hier")
        print(VECTOR_STORE_PATH
            + self._get_embedding_name()
            + "_"
            + vector_store_name
            + "/"
            + str(chunk_size)
            + "_"
            + str(chunk_overlap))
        db = FAISS.load_local(
            VECTOR_STORE_PATH
            + self._get_embedding_name()
            + "_"
            + vector_store_name
            + "/"
            + str(chunk_size)
            + "_"
            + str(chunk_overlap),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True

        )
        return db

    def _create_retriever(self, db: FAISS):
        print("[INFO] Creating Retriever...")
        return db.as_retriever(
            search_type=self.search_type, search_kwargs={"k": self.search_kwargs_num}
        )

    def get_db_and_retriever(
        self, vector_store_name: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> Tuple[VectorStoreRetriever, FAISS]:
        """Tries to get an existing vector store if it exists."""

        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        print(
            f"[INFO] Getting existing Vector Store: {vector_store_name}_{chunk_size}_{chunk_overlap}"
        )
        print("hier fangt das dilemma an")
        try:
            db = self._get_existing_vector_store(
                vector_store_name=vector_store_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            

            retriever = self._create_retriever(db=db)
            return retriever, db
        except:
            print("Could not get existing vector store.")
            return None, None

    def create_db_and_retriever(
        self,
        documents: List[Document],
        vector_store_name: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> Tuple[VectorStoreRetriever, FAISS]:
        """Creates a vector store and retriever from a list of documents."""

        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        db = self._create_vector_store(
            split_documents=documents,
            vector_store_name=vector_store_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        retriever = self._create_retriever(db=db)
        return retriever, db

    def delete_db(
        self, vector_store_name: str, chunk_size: int = None, chunk_overlap: int = None
    ):
        """Deletes a vector store. -> Delete the existing folder"""

        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap

        try:
            shutil.rmtree(
                VECTOR_STORE_PATH
                + self._get_embedding_name()
                + "_"
                + vector_store_name
                + "_"
                + str(chunk_size)
                + "_"
                + str(chunk_overlap)
            )
        except OSError as e:
            print(f"Error: {e}")
