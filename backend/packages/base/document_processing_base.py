import os
from typing import List
from langchain.schema.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from packages.globals import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessing:
    """
    This class is responsible for loading the documents from a Directory
    and splitting them into chunks.
    """

    def __init__(self):
        pass

    def _load_documents(self, directory_path: str) -> List[Document]:
        documents: List[Document] = []

        # Loop through all files in the directory and add them to the List
        for file in os.listdir(directory_path):
            file_path: str = os.path.join(directory_path, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            if file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            # Extenadble as needed with more file types

        return documents

    def _split_documents(
        self, documents: List[Document], chunk_size: int, chunk_overlap: int
    ) -> List[Document]:
        splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        split_documents = splitter.split_documents(documents)
        return split_documents

    def get_chunked_documents(
        self,
        directory_path: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> List[Document]:
        documents = self._load_documents(directory_path)
        split_documents = self._split_documents(documents, chunk_size, chunk_overlap)
        return split_documents
