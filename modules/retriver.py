import os
import hashlib
import pickle
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client.models import Distance, VectorParams

class NormalizedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings normalized for cosine similarity."""
    def embed_documents(self, texts):
        vectors = super().embed_documents(texts)
        return [np.array(v) / np.linalg.norm(v) for v in vectors]

    def embed_query(self, text):
        vector = super().embed_query(text)
        return np.array(vector) / np.linalg.norm(vector)


class QdrantPDFRetriever:
    def __init__(
        self,
        pdf_path: str = "D:/Gen AI/Langgraph Architecture/Langgraph_Architecture/data/Ares XXXI CLO Ltd.PDF",
        db_path: str = "qdrant_db",
        collection_name: str = "pdf_docs",
        keywords: Optional[List[str]] = ["Asset Manager", "Issuer", "Trustee", "Placement Agent"],
        embeddings_model=None,
        hash_path: str = "doc_hash_quad.pkl"
    ):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.collection_name = collection_name
        self.keywords = [kw.lower() for kw in (keywords or [])]
        self.embeddings = embeddings_model or NormalizedOpenAIEmbeddings()
        self.hash_path = hash_path
        self.pages: List[Document] = []

        # Qdrant local client
        self.client = QdrantClient(url="http://127.0.0.1:6333")

        self._init_store()

    def _init_store(self):
        """Initialize or load Qdrant vector store."""
        current_hash = self._compute_file_hash(self.pdf_path)
        previous_hash = self._load_prev_hash()

        if self.client.collection_exists(self.collection_name) and current_hash == previous_hash:
            print(f"✅ Using cached Qdrant collection '{self.collection_name}'")
            self.pages = PyPDFLoader(self.pdf_path).load()
        else:
            print(f"🛠 Building Qdrant collection '{self.collection_name}'")
            self.pages = PyPDFLoader(self.pdf_path).load()
            all_docs = self._prepare_documents(self.pages)

            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)


            self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=1536,
                            distance=Distance.COSINE
                        ),)
                        
            embeddings = self.embeddings

            # Create vectorstore by adding embeddings and documents explicitly
            vectordb = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=embeddings,
            )

            # Upsert docs
            vectordb.add_documents(all_docs)



            # Qdrant.from_documents(
            #     all_docs,
            #     embedding=self.embeddings,
            #     client=self.client,
            #     collection_name=self.collection_name
            # )

            self._save_current_hash(current_hash)

    def _prepare_documents(self, pages: List[Document]) -> List[Document]:
        """Split documents into KV and body chunks."""
        all_docs = []

        # KV docs
        for page in pages:
            for line in page.page_content.splitlines():
                if ":" in line:
                    key, value = map(str.strip, line.split(":", 1))
                    if key.lower() in self.keywords:
                        all_docs.append(Document(
                            page_content=f"{key}: {value}",
                            metadata={"section": "kv_block", "key": key.lower()}
                        ))

        # Body docs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ":", " "]
        )

        for page in pages:
            for chunk in splitter.split_text(page.page_content):
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"section": "body"}
                ))

        return all_docs

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve documents by keyword or semantic search."""
        vectordb = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

        # Keyword-based search
        if any(kw in query.lower() for kw in self.keywords):
            print("🔍 Searching KV blocks...")
            results = vectordb.similarity_search(
                query, k=k,
                # filter={
                #     "must": [
                #         {"key": "section", "match": {"value": "kv_block"}}
                #     ]
                # }
            )
        else:
            print("📘 Searching body sections...")
            results = vectordb.similarity_search(
                query, k=k,
                # filter={
                #     "must": [
                #         {"key": "section", "match": {"value": "body"}}
                #     ]
                # }
            )

        print(f"✅ Retrieved {len(results)} docs")
        return results

    def _compute_file_hash(self, file_path: str) -> str:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            sha.update(f.read())
        return sha.hexdigest()

    def _load_prev_hash(self) -> Optional[str]:
        if os.path.exists(self.hash_path):
            with open(self.hash_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_current_hash(self, hash_value: str):
        with open(self.hash_path, "wb") as f:
            pickle.dump(hash_value, f)
