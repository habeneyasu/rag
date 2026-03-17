import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
from implementation.answer import logger

MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

# Use BAAI/bge-base-en-v1.5: Best for RAG retrieval tasks
# - Optimized for asymmetric retrieval (query vs document)
# - Strong performance on retrieval benchmarks
# - Better semantic search for RAG applications
# - Runs locally, no API costs
load_dotenv(override=True)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def fetch_documents():
    """Load all markdown documents from the knowledge base."""
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    """Chunk documents with dynamic sizes and markdown header preservation."""

    all_chunks = []
    
    # Markdown header splitter to preserve section structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    
    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "unknown")
        
        if doc_type == "contracts":
            chunk_size = 500
            chunk_overlap = 100
        elif doc_type == "products":
            chunk_size = 900
            chunk_overlap = 180
        else:
            chunk_size = 700
            chunk_overlap = 140
        
        try:
            header_splits = markdown_splitter.split_text(doc.page_content)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            
            for header_split in header_splits:
                header_metadata = header_split.metadata.copy()
                header_metadata.update(doc.metadata)
                
                section_parts = []
                for level in ["Header 1", "Header 2", "Header 3"]:
                    if level in header_metadata:
                        section_parts.append(header_metadata[level])
                
                if section_parts:
                    header_metadata["section"] = " > ".join(section_parts)
                
                sub_chunks = text_splitter.split_documents([header_split])
                for idx, chunk in enumerate(sub_chunks):
                    chunk.metadata.update(header_metadata)
                    chunk.metadata["chunk_index"] = idx
                    chunk.metadata["total_chunks_in_section"] = len(sub_chunks)
                
                all_chunks.extend(sub_chunks)
        
        except Exception as e:
            logger.warning(f"Markdown splitting failed for {doc.metadata.get('source', 'unknown')}: {e}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            fallback_chunks = text_splitter.split_documents([doc])
            all_chunks.extend(fallback_chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks with advanced chunking strategy")
    return all_chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    logger.info(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    logger.info("Ingestion complete")
