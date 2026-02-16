import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import List
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_db")

DATA_DIR.mkdir(exist_ok=True)


class RAGAgent:
    def __init__(self):

        print("\nInitializing Hybrid RAG Agent...\n")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        self.documents = self._load_documents()
        self.vector_store = self._init_chroma(self.documents)
        self.retriever = self._init_hybrid_retriever()

        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
        )

        self.memory = ChatMessageHistory()

        self.tools = self._create_tools()

        self.agent = create_agent(self.llm, self.tools)
        print("Agent ready.\n")

    # ---------- DOCUMENT LOADING ----------
    def _load_documents(self) -> List[Document]:
        docs = []
        for file in DATA_DIR.rglob("*"):
            if not file.is_file():
                continue
            try:
                if file.suffix == ".pdf":
                    loader = PyPDFLoader(str(file))
                elif file.suffix == ".csv":
                    loader = CSVLoader(str(file))
                elif file.suffix in [".txt", ".md"]:
                    loader = TextLoader(str(file))
                else:
                    continue

                loaded = loader.load()
                docs.extend(loaded)
                print(f"Loaded: {file.name}")

            except Exception as e:
                print(f"Error loading {file.name}: {e}")

        if docs:
            docs = self.text_splitter.split_documents(docs)

        return docs


    def _init_chroma(self, docs: List[Document]):
        store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        if docs:
            store.add_documents(docs)
            print(f"Indexed {len(docs)} document chunks.")

        return store

    def _init_hybrid_retriever(self):

        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        bm25_retriever = BM25Retriever.from_documents(
            self.documents if self.documents else [Document(page_content="")]
        )
        bm25_retriever.k = 5

        hybrid = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        return hybrid

    def _store_message(self, role: str, content: str):

        doc = Document(
            page_content=content,
            metadata={
                "type": "conversation",
                "role": role,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        self.vector_store.add_documents([doc])

    def _create_tools(self):
        @tool
        def rag_query(query: str) -> str:
            """Search internal documents and past conversations."""
            results = self.retriever.invoke(query)  

            if not results:
                return "No relevant information found."

            return "\n\n".join(
                f"[Source {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(results)
            )

        @tool
        def get_flight_booking(origin: str, destination: str) -> str:
            """Get flight booking information including schedule, duration, and pricing in USD for a round trip. 
            Input: origin (departure city) and destination (arrival city) separated by comma, e.g., 'Lagos, Nairobi'."""
            flights = {
                'lagos-nairobi': {
                    'outbound': {
                        'departure': '2026-02-15T08:00:00',
                        'arrival': '2026-02-15T13:30:00',
                        'duration': '5h 30m',
                        'price': 450,
                        'airline': 'Kenya Airways',
                        'flightNumber': 'KQ512'
                    },
                    'return': {
                        'departure': '2026-02-18T15:00:00',
                        'arrival': '2026-02-18T20:30:00',
                        'duration': '5h 30m',
                        'price': 450,
                        'airline': 'Kenya Airways',
                        'flightNumber': 'KQ513'
                    },
                    'totalFlightTime': '11h 0m',
                    'totalPrice': 900,
                    'currency': 'USD'
                }
            }
            
            key = f"{origin.lower()}-{destination.lower()}"
            result = flights.get(key, {'error': 'Route not found'})
            return json.dumps(result, indent=2)

        @tool
        def get_hotel_booking(location: str, nights: int) -> str:
            """Get hotel booking information including pricing in USD per night and total cost. 
            Input: location (city) and nights (number) separated by comma, e.g., 'Nairobi, 3'."""
            hotels = {
                'nairobi': {
                    'name': 'Nairobi Serena Hotel',
                    'pricePerNight': 180,
                    'totalNights': nights,
                    'totalPrice': 180 * nights,
                    'currency': 'USD',
                    'amenities': ['WiFi', 'Breakfast', 'Conference facilities']
                }
            }
            
            result = hotels.get(location.lower(), {'error': 'Location not found'})
            return json.dumps(result, indent=2)

        @tool
        def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
            """Convert an amount from one currency to another. 
            Input: amount, from_currency, to_currency separated by commas, e.g., '900, USD, NGN'."""
            rates = {
                'USD': {'USD': 1, 'NGN': 1550, 'KES': 160, 'EUR': 0.92, 'GBP': 0.79},
                'NGN': {'USD': 0.00065, 'NGN': 1, 'KES': 0.103, 'EUR': 0.0006, 'GBP': 0.00051},
                'KES': {'USD': 0.00625, 'NGN': 9.69, 'KES': 1, 'EUR': 0.0058, 'GBP': 0.0049}
            }
            
            rate = rates.get(from_currency.upper(), {}).get(to_currency.upper(), 1)
            converted_amount = amount * rate
            
            result = {
                'originalAmount': amount,
                'originalCurrency': from_currency.upper(),
                'convertedAmount': round(converted_amount, 2),
                'targetCurrency': to_currency.upper(),
                'exchangeRate': rate
            }
            return json.dumps(result, indent=2)
        

        return [get_flight_booking, get_hotel_booking, convert_currency, rag_query]

    def run(self, query: str):

        self.memory.add_user_message(query)
        self._store_message("user", query)

        result = self.agent.invoke(
            {"messages": self.memory.messages}
        )

        output = result["messages"][-1].content
        self.memory.add_ai_message(output)
        self._store_message("assistant", output)

        print("\n--- Conversation History ---\n")
        for msg in self.memory.messages:
            role = "User" if msg.type == "human" else "Assistant"
            print(f"{role}: {msg.content}\n")

        print("\n--- Final Response ---\n")
        print(output)


def main():

    if len(sys.argv) < 2:
        print('Usage: python main.py "your question"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    agent = RAGAgent()
    agent.run(query)


if __name__ == "__main__":
    main()
