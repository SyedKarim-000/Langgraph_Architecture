import os
import hashlib
import pickle
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Constants
DB_PATH = "vector_store"
HASH_PATH = "doc_hash.pkl"
PDF_PATH = "D:/Gen AI/Langgraph Architecture/Langgraph_Architecture/data/Ares XXXI CLO Ltd.PDF"
KEYWORDS = ["Asset Manager", "Issuer", "Trustee", "Placement Agent"]

# Normalized OpenAI Embeddings for cosine similarity
class NormalizedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        vectors = super().embed_documents(texts)
        return [np.array(v) / np.linalg.norm(v) for v in vectors]

    def embed_query(self, text):
        vector = super().embed_query(text)
        return np.array(vector) / np.linalg.norm(vector)


class Retriever:
    def __init__(self):
        self.pages: list[Document]
        self.vectordb: FAISS

        embeddings = NormalizedOpenAIEmbeddings()
        current_hash = self.compute_files_hash([PDF_PATH])
        previous_hash = self.load_prev_hash()

        if os.path.exists(DB_PATH) and current_hash == previous_hash:
            print("✅ Loading cached FAISS vector store...")
            self.vectordb = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"PDF path {PDF_PATH}")
            self.pages = PyPDFLoader(PDF_PATH).load()
        else:
            print("🛠 Building new FAISS vector store...")
            self.pages = PyPDFLoader(PDF_PATH).load()
            all_docs = self.prepare_documents(self.pages)
            self.vectordb = FAISS.from_documents(all_docs, embeddings)
            self.vectordb.save_local(DB_PATH)
            self.save_current_hash(current_hash)

    def prepare_documents(self, pages: list[Document]) -> list[Document]:
        all_docs = []

        # Key-Value pairs as separate docs
        for page in pages:
            for line in page.page_content.splitlines():
                if ":" in line:
                    key, value = map(str.strip, line.split(":", 1))
                    if key in KEYWORDS:
                        all_docs.append(Document(
                            page_content=f"{key.lower()}: {value}",
                            metadata={"section": "kv_block", "key": key}
                        ))

        # Chunk full body
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", ":", " "]
        )

        for page in pages:
            chunks = splitter.split_text(page.page_content)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"section": "body"}
                ))

        return all_docs

    def retrieve(self, query: str, k: int = 15) -> list[Document]:
        results = []

        # First, look for keyword blocks
        if any(kw.lower() in query.lower() for kw in KEYWORDS):
            for kw in KEYWORDS:
                if kw.lower() in query.lower():
                    print(f"🔍 Searching KV blocks for keyword: {kw}")
                    results = self.vectordb.similarity_search(
                        query, k=k,
                        filter={"section": "kv_block", "key": kw.lower()}
                    )
                    if results:
                        break  # Stop after finding relevant block

        # Otherwise search body text
        if not results:
            print("📘 Searching body sections...")
            results = self.vectordb.similarity_search(query, k=k, filter={"section": "body"})

        # Final fallback: scan PDF manually
        if not results:
            print("⚠️ Fallback: scanning entire document...")
            for page in self.pages:
                if query.lower() in page.page_content.lower():
                    results.append(Document(page_content=page.page_content))

        # Final fallback: just return first few lines with keywords
        if not results:
            print("⚠️ Keyword scan fallback...")
            for page in self.pages:
                for line in page.page_content.splitlines():
                    if any(k in line for k in KEYWORDS):
                        results.append(Document(page_content=line))
            results = results[:k]

        # Log result preview
        print(f"✅ Retrieved {len(results)} docs:")
        for i, doc in enumerate(results[:3], 1):
            print(f"--- Result {i} ---")
            print(doc.page_content[:300])
            print()

        return results

    def compute_files_hash(self, file_paths: list[str]) -> str:
        sha = hashlib.sha256()
        for path in file_paths:
            with open(path, "rb") as f:
                content = f.read()
                sha.update(content)
        return sha.hexdigest()

    def load_prev_hash(self):
        if os.path.exists(HASH_PATH):
            with open(HASH_PATH, "rb") as f:
                return pickle.load(f)
        return None

    def save_current_hash(self, hash_value):
        with open(HASH_PATH, "wb") as f:
            pickle.dump(hash_value, f)
