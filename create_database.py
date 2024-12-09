import argparse
import os
import shutil

from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"
DATA_PATH = "data"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def main():
    
    # Clear the database if required
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear the database.")
    args = parser.parse_args()
    if args.clear:
        clear_database()
    
    # Load data
    documents = load_documents(DATA_PATH)
    
    # Load embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Split documents into smaller chunks
    chunks = split_documents(documents)
    
    # Create embeddings and add to the database
    create_chroma(chunks, embeddings_model)
    

def load_documents(data_path: str) -> list:
    """
    Loads documents from the specified directory path.

    Args:
        data_path (str): The path to the directory containing PDF documents.

    Returns:
        list: A list of loaded documents.
    """
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load() 


def split_documents(documents: list[Document]) -> list:
    """
    Splits a list of Document objects into smaller chunks using a RecursiveCharacterTextSplitter.

    Args:
        documents (list[Document]): A list of Document objects to be split.

    Returns:
        list: A list of smaller chunks obtained from the original documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def create_chroma(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings):
    """
    Create or update a Chroma database with the given document chunks and embeddings model.
    
    Args:
        chunks (list[Document]): A list of Document objects representing the chunks to be added to the database.
        embeddings_model (HuggingFaceEmbeddings): An instance of HuggingFaceEmbeddings used for generating embeddings.
    
    Returns:
        None
    This function performs the following steps:
    1. Creates or loads a Chroma database from the specified directory.
    2. Calculates unique chunk IDs for the provided document chunks.
    3. Retrieves existing chunks from the database and identifies their IDs.
    4. Compares the new chunk IDs with the existing ones to determine which chunks are new.
    5. Adds the new chunks to the database and persists the changes if there are any new chunks.
    """
    
    # Create or load a Chroma database
    chroma = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings_model
    )
    
    # Create page ids
    chunk_ids = calculate_chunk_ids(chunks)
    
    # Get existing chunks
    existing_chunks = chroma.get(include=[])
    existing_ids = set(existing_chunks["ids"])
    print(f"Existing number of documents in database: {len(existing_ids)}")
    
    # Add new chunks to the database
    new_chunks = []
    for chunk in chunk_ids:
        if chunk.metadata["chunk_id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new documents to the database.")
        new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        chroma.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add to the database.")
    

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Calculate and assign unique chunk IDs to each chunk in the provided list.
    Each chunk ID is composed of the source, page, and an index to ensure uniqueness
    within the same page. The chunk ID is added to the metadata of each chunk.
    
    Args:
        chunks (list): A list of chunk objects, where each chunk has a metadata attribute
                       containing 'source' and 'page' keys.
    
    Returns:
        list: The list of chunks with updated metadata including the unique 'chunk_id'.
    """
    last_page_id = None
    current_chunk_index = 0
    
    for chunk in chunks:
        # Create page id
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        # Ensure that the page id is unique
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # Create chunk id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        
        # Add chunk id to metadata
        chunk.metadata["chunk_id"] = chunk_id
    
    return chunks


def clear_database():
    """
    Clears the database by removing the directory specified by CHROMA_PATH.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Database cleared.")


if __name__ == "__main__":
    main()
        