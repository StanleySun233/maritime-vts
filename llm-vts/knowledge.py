import argparse
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class Knowledge:
    def __init__(self):
        self.path = os.environ.get("VTS_KNOWLEDGE_DIR", str(Path(__file__).parent / "knowledge_store"))
        self.store = None

    def vectorstore(self):
        if self.store is None:
            embeddings = OpenAIEmbeddings(
                model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                check_embedding_ctx_length=False,
            )
            self.store = Chroma(collection_name="maritime", embedding_function=embeddings, persist_directory=self.path)
        return self.store

    def search(self, query):
        store = self.vectorstore()
        if store._collection.count() == 0:
            return {"documents": [], "status": "knowledge base is empty"}
        docs = store.similarity_search_with_relevance_scores(query, k=4, score_threshold=0.3)
        return {"documents": [{"content": doc.page_content, "source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "score": score} for doc, score in docs]}

    def ingest(self, path):
        path = Path(path).resolve()
        if path.suffix.lower() == ".pdf":
            documents = [Document(page_content=page.extract_text() or "", metadata={"source": str(path), "page": i + 1}) for i, page in enumerate(PdfReader(path).pages)]
        elif path.suffix.lower() in {".txt", ".md"}:
            documents = [Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})]
        else:
            raise ValueError("Supported formats: .pdf, .txt, .md")
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(documents)
        if not chunks:
            raise ValueError("The document contains no text")
        store = self.vectorstore()
        previous = store.get(where={"source": str(path)})["ids"]
        store.add_documents(chunks)
        if previous:
            store.delete(ids=previous)
        return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for file in args.files:
        print(f"{file}: {Knowledge().ingest(file)} chunks")
