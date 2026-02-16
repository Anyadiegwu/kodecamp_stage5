# Hybrid RAG Agent (LangChain)

A command-line Retrieval-Augmented Generation (RAG) agent that combines semantic vector search and keyword search to answer questions from local documents and conversation history. The agent uses a hybrid retriever, persistent vector storage, and tool-based reasoning to support document search and simple travel utilities.

## Features

* Hybrid retrieval using semantic embeddings + BM25 keyword search
* Persistent vector database for document and conversation storage
* Automatic document ingestion from a local data folder
* Conversational memory stored and indexed for future retrieval
* Tool-enabled agent with travel booking and currency conversion utilities
* Command-line interface for interactive querying

## Architecture Overview

The agent pipeline consists of:

1. **Document ingestion**

   * Loads PDF, CSV, TXT, and Markdown files
   * Splits text into chunks for embedding

2. **Vector storage**

   * Uses a persistent Chroma vector database
   * Stores both documents and conversation history

3. **Hybrid retrieval**

   * Dense semantic search via embeddings
   * BM25 keyword search
   * Weighted ensemble retriever merges results

4. **Agent execution**

   * LLM-powered reasoning with tool access
   * Conversation memory tracking
   * Retrieval-augmented responses

## Project Structure

```
project/
├── main.py
├── data/            # Place your documents here
├── chroma_db/       # Persistent vector database (auto-created)
├── .env             # API keys
```

## Requirements

* Python 3.9+
* Virtual environment recommended

Install dependencies:

```bash
pip install langchain langchain-openai langchain-huggingface \
langchain-chroma langchain-community langchain-classic \
python-dotenv chromadb sentence-transformers
```

## Environment Setup

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_openrouter_api_key
HF_API_KEY=your_huggingface_api_key
```

Only the OpenRouter key is required for LLM access. The Hugging Face key is optional depending on your embedding configuration.

## Adding Documents

Place supported files inside the `data/` folder:

* `.pdf`
* `.csv`
* `.txt`
* `.md`

On startup, the agent automatically loads and indexes these documents.

## Usage

Run the agent from the command line:

```bash
python main.py "your question here"
```

Example:

```bash
python main.py "What information do you have about Nairobi travel?"
```

The agent will:

1. Load and index documents
2. Retrieve relevant content
3. Use tools if necessary
4. Generate a final response
5. Print conversation history

## Available Tools

### 1. Document Search

Searches indexed documents and conversation history using hybrid retrieval.

### 2. Flight Booking

Returns mock flight schedule and pricing data.

Input format:

```
origin, destination
```

Example:

```
Lagos, Nairobi
```

### 3. Hotel Booking

Returns mock hotel pricing information.

Input format:

```
location, nights
```

Example:

```
Nairobi, 3
```

### 4. Currency Conversion

Converts between supported currencies.

Input format:

```
amount, from_currency, to_currency
```

Example:

```
900, USD, NGN
```

## Configuration

You can tune retrieval and performance by editing:

* Chunk size and overlap in the text splitter
* Number of retrieved documents (`k`)
* Ensemble retriever weights
* LLM temperature

## Persistence

The vector database is stored in `chroma_db/`. This directory:

* Persists across runs
* Stores document embeddings
* Stores conversation history

Delete it to rebuild the index from scratch.

## Troubleshooting

**No documents loaded**

Ensure files exist in the `data/` folder and are in supported formats.

**API errors**

Verify your `.env` file contains valid keys.

**Slow startup**

Large document collections increase indexing time. This happens only on initial ingestion.

## Limitations

* Travel and currency tools use static mock data
* No streaming or interactive shell mode
* Designed for local experimentation and prototyping

## Future Improvements

* Real API integration for travel data
* Web interface or chat UI
* Streaming responses
* Advanced ranking strategies
* Distributed vector storage

## License

This project is provided for educational and experimental use.
