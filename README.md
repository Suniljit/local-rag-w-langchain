# Local RAG with Ollama and Langchain
## Introduction
This project runs a local RAG system by creating embeddings from a local model and serving a LLM to generate a response using Ollma. It leverages Langchain to run the RAG system. Chroma DB is used to store the embeddings.

## Set Up
1. Clone the repository
2. Install the required packages using the code below<br>
`pip install -r requirements.txt`

## Create or Update Database
1. Store your PDF files in the data folder.
2. Create or update a database in your local machine using a local embeddings model.<br>
`python create_database.py`
3. You can add the ---clear flag to the code above to clear the database. 

## Running a Query
1. Run the below code to run a query on the database, which inserting your query text between the quotation marks.<br>
`python main.py "How to get out of jail?"`
2. The model will output the following details:
   - Retrieved relevant chunks from the database
   - Response generated by the model