# Filename: rag_system.py

import os
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# Constants
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
LLM_MODEL = "TheDrummer/Moistral-11B-v3-GGUF"
CHUNK_SIZE = 512  # Number of tokens in each chunk
OVERLAP = 50  # Number of overlapping tokens between chunks
MAX_TOKENS = 8000  # Max tokens for final prompt to LLM
TOKEN_LIMIT = 512  # Maximum tokens for embedding request
BATCH_SIZE = 512   # Maximum number of tokens to send in a single batch

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# tokenizer used to calculate token count
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def tokenize_text(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_embeddings(texts):
    """
    Function to get embeddings from OpenAI embedding model.
    :param texts: List of texts to embed
    :return: List of embeddings (vectors)
    """
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    embeddings = [data.embedding for data in response.data]
    return np.array(embeddings)

def estimate_token_count(text):
    """
    Estimate the number of tokens in a given text based on its length in characters.
    Approximate 1 token per 4 characters.
    :param text: Input text to estimate token count
    :return: Estimated number of tokens
    """
    return len(text) // 4  # Roughly estimate 1 token per 4 characters

def chunk_text(text, token_limit=TOKEN_LIMIT):
    """
    Splits text into chunks that fit within the token limit, based on character count approximation.
    :param text: The full text to chunk
    :param token_limit: Maximum token length per chunk
    :return: List of text chunks that fit within the token limit
    """
    words = text.split()
    current_chunk = []
    current_token_count = 0
    chunks = []

    for word in words:
        # Estimate the number of tokens for the current word
        word_token_count = estimate_token_count(word)
        
        # If adding this word would exceed the token limit, save the current chunk and start a new one
        if current_token_count + word_token_count > token_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_token_count = word_token_count
        else:
            current_chunk.append(word)
            current_token_count += word_token_count
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def batch_texts(texts, batch_size=BATCH_SIZE):
    """
    Batches the text input while ensuring each batch contains less than the batch_size token count.
    :param texts: List of text inputs to batch
    :param batch_size: Maximum number of tokens per batch (approximated using text length)
    :return: List of batches (each batch is a list of text inputs)
    """
    batches = []
    current_batch = []
    current_batch_token_count = 0

    for text in texts:
        token_count = estimate_token_count(text)

        # If adding this text exceeds the batch size, start a new batch
        if current_batch_token_count + token_count > batch_size:
            batches.append(current_batch)
            current_batch = [text]
            current_batch_token_count = token_count
        else:
            current_batch.append(text)
            current_batch_token_count += token_count

    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)

    return batches

def get_embeddings(texts):
    """
    Function to get embeddings from OpenAI embedding model with batch size and token limit handling.
    :param texts: List of texts to embed
    :return: List of embeddings (vectors)
    """
    all_embeddings = []
    
    # Step 1: Chunk any long text into smaller chunks
    chunked_texts = []
    for text in texts:
        chunked_texts.extend(chunk_text(text, TOKEN_LIMIT))
    
    # Step 2: Batch the chunked texts
    text_batches = batch_texts(chunked_texts, BATCH_SIZE)
    
    # Step 3: Request embeddings for each batch
    for batch in text_batches:
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings)


def chunk_document(document, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Function to chunk a document into smaller segments with overlap.
    :param document: String, the document to chunk
    :param chunk_size: Integer, the number of tokens in each chunk
    :param overlap: Integer, number of tokens to overlap between chunks
    :return: List of chunks (text segments)
    """
    words = document.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def generate_response(prompt):
    """
    Function to generate a response from OpenAI LLM.
    :param prompt: The text prompt to pass to the LLM
    :return: The generated text response
    """
    response = client.chat.completions.create(
        # max_tokens=2048,  # Adjust as needed for the response length
        messages=[
            # {"role": "system", "content": "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."},
            {"role": "user", "content": prompt}
        ],
        model=LLM_MODEL,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def summarize_chunks(chunks):
    """
    Function to summarize document chunks using LLM.
    :param chunks: List of text chunks to summarize
    :return: List of summaries for each chunk
    """
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following content: {chunk}"
        summary = generate_response(prompt)
        summaries.append(summary)
    return summaries


def calculate_token_length(text):
    """
    Estimate the number of tokens in a given text. 
    This can be a simple word count approximation.
    :param text: String
    :return: Integer, number of tokens
    """
    return len(text.split())


def retrieve_relevant_chunks(query, chunk_embeddings, flat_chunks, top_n=15):
    """
    Retrieve the top-n relevant document chunks for a given query.
    :param query: User's query
    :param chunk_embeddings: List of embeddings for document chunks
    :param flat_chunks: List of document chunks (text)
    :param top_n: Number of top relevant chunks to retrieve
    :return: List of top-n relevant chunks
    """
    query_embedding = get_embeddings([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)
    most_relevant_chunk_indices = np.argsort(similarities[0])[-top_n:][::-1]
    relevant_chunks = [flat_chunks[i] for i in most_relevant_chunk_indices]
    return relevant_chunks


def construct_prompt(user_query, relevant_chunks, max_tokens=MAX_TOKENS):
    """
    Construct the prompt for the LLM by ensuring the total length fits the token limit.
    :param user_query: The original user query
    :param relevant_chunks: List of retrieved document chunks
    :param max_tokens: The maximum token length allowed for the prompt
    :return: The prompt to send to the LLM
    """
    final_context = []
    current_token_count = calculate_token_length(user_query)

    for chunk in relevant_chunks:
        chunk_token_count = calculate_token_length(chunk)
        if current_token_count + chunk_token_count > max_tokens:
            break
        final_context.append(chunk)
        current_token_count += chunk_token_count

    # Construct the final prompt
    prompt = f"User query: {user_query}\nRelevant context: {' '.join(final_context)}"
    return prompt

# Function to extract text from a PDF file
def extract_text_from_pdf(file_bytes):
    reader = PdfReader(file_bytes)

    # Check if the PDF is encrypted
    if reader.is_encrypted:
        try:
            reader.decrypt('')  # Attempt to decrypt with no password (empty string)
        except Exception as e:
            raise ValueError("The PDF is encrypted and requires a password.")
    
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_text_file(file_path):
    """
    Load content from a .txt file.
    :param file_path: Path to the .txt file
    :return: Text content as a string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_documents_from_files(file_paths):
    """
    Load documents from a list of file paths.
    Supports both .txt and .pdf files.
    :param file_paths: List of file paths
    :return: List of document contents (strings)
    """
    documents = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(file_path)
            if file_extension.lower() == ".txt":
                documents.append(load_text_file(file_path))
            elif file_extension.lower() == ".pdf":
                documents.append(extract_text_from_pdf(file_path))
            else:
                print(f"Unsupported file format: {file_path}")
        else:
            print(f"File not found: {file_path}")
    return documents


def rag_pipeline(file_paths, user_query):
    """
    Main RAG pipeline.
    :param file_paths: List of file paths to load documents from
    :param user_query: The user's query
    :return: Generated response from the LLM
    """
    # Step 1: Load documents from files
    documents = load_documents_from_files(file_paths)

    # Step 2: Chunk the documents
    chunked_docs = [chunk_document(doc) for doc in documents]
    flat_chunks = [chunk for doc_chunks in chunked_docs for chunk in doc_chunks]

    # Step 3: Get embeddings for document chunks
    chunk_embeddings = get_embeddings(flat_chunks)

    # Step 4: Retrieve the most relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(user_query, chunk_embeddings, flat_chunks)

    # Step 5: Summarize chunks if necessary (optional)
    summarized_chunks = summarize_chunks(relevant_chunks)

    # Step 6: Construct the final prompt with token limit control
    prompt = construct_prompt(user_query, summarized_chunks)

    # Step 7: Generate the final response using the LLM
    response = generate_response(prompt)

    return response


# Example Usage
if __name__ == "__main__":
    # List of file paths to the documents
    file_paths = [
        "databases/document1.txt",
        "databases/document2.txt",
        "databases/itStartsWithUs.pdf"
        # Add more file paths as needed
    ]

    # User query
    user_query = "Give me a detailed summary of what happened in the first chapter of It Ends with Us, featuring Lily and Atlas."
    # "Give me an detailed response about how Tibalt is not just a villian in Romeo and Julliet. Be detailed, and descriptive. Use reference from the text to support your idea."

    # Run RAG pipeline
    final_response = rag_pipeline(file_paths, user_query)

    # Print the response
    print("Generated response:", final_response)
