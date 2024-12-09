import argparse

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

CHROMA_PATH = "chroma"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

PROMPT_TEMPLATE = """
Answer the question below:

Here is the retrieved context. Ignore the context that are not relevant to the question: \n\n{context}

Question: {question}

Answer: 
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_prompt", type=str, help="Text for the query")
    args = parser.parse_args()
    query_prompt = args.query_prompt
    query(query_prompt)
    

def query(query_prompt: str):
    # Load database
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)
    
    # Run similarity search
    results = chroma.similarity_search_with_score(query_prompt, k=3)
    
    # Update context with retrieved information
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_prompt)
    
    # Load model
    model = OllamaLLM(model="llama3")
    
    response = model.invoke(prompt)
    
    sources = [doc.metadata.get("chunk_id", None) for doc, _ in results]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(prompt)
    print(formatted_response)
    return response
    

if __name__ == "__main__":
    main()