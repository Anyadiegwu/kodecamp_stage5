import os
import sys
import json
import argparse
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class ConversationMemoryStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents: List[Document] = []
        self.store = None

    def _rebuild_store(self):
        if self.documents:
            self.store = Chroma.from_documents(
                self.documents,
                self.embeddings,
                collection_name="conversation_memory",
            )

    def add(self, question: str, answer: str):
        doc = Document(
            page_content=f"User: {question}\nAssistant: {answer}"
        )
        self.documents.append(doc)
        self._rebuild_store()

    def search(self, query: str, k: int = 3) -> List[Document]:
        if not self.store:
            return []
        retriever = self.store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}, 
            encode_kwargs={'normalize_embeddings': True}
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        self.documents: List[Document] = []
        self.vector_store = None
        self.bm25 = None
        self.retriever = None
        self.memory = ConversationMemoryStore(self.embeddings)

    def load_data_dir(self, folder="data"):
        path = Path(folder)

        if not path.exists():
            print(f"⚠ '{folder}' folder not found. Creating it...")
            path.mkdir(exist_ok=True)
            return

        supported = {".txt", ".pdf", ".csv", ".docx", ".pptx"}

        for file in path.iterdir():
            if file.suffix.lower() not in supported:
                continue

            if file.suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix == ".csv":
                loader = CSVLoader(str(file))
            elif file.suffix == ".docx":
                loader = Docx2txtLoader(str(file))
            else:
                loader = UnstructuredPowerPointLoader(str(file))

            docs = self.splitter.split_documents(loader.load())
            self.documents.extend(docs)

        print(f"✓ Indexed {len(self.documents)} document chunks")

    def build_retriever(self):
        if not self.documents:
            print("⚠ No documents loaded.")
            return

        self.vector_store = Chroma.from_documents(
            self.documents,
            self.embeddings,
            collection_name="rag_documents",
        )

        self.bm25 = BM25Retriever.from_documents(self.documents, k=5)

        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 2}
        )

        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25, vector_retriever],
            weights=[0.5, 0.5],
        )


    def create_tools(self):

        @tool
        def rag_query(query: str) -> str:
            """Search internal documents and past conversations."""
            docs = self.retriever.invoke(query) if self.retriever else []
            history = self.memory.search(query)

            combined = docs + history

            if not combined:
                return "No relevant information found."

            return "\n\n".join(
                f"[Source {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(combined)
            )

        @tool
        def get_flight_booking(origin: str, destination: str) -> str:
            """Get round-trip flight booking details."""
            flights = {
                "lagos-nairobi": {
                    "outbound": {
                        "departure": "2026-02-15T08:00:00",
                        "arrival": "2026-02-15T13:30:00",
                        "duration": "5h 30m",
                        "price": 450,
                        "airline": "Kenya Airways",
                        "flightNumber": "KQ512",
                    },
                    "return": {
                        "departure": "2026-02-18T15:00:00",
                        "arrival": "2026-02-18T20:30:00",
                        "duration": "5h 30m",
                        "price": 450,
                        "airline": "Kenya Airways",
                        "flightNumber": "KQ513",
                    },
                    "totalPrice": 900,
                    "currency": "USD",
                }
            }

            key = f"{origin.lower()}-{destination.lower()}"
            return json.dumps(
                flights.get(key, {"error": "Route not found"}),
                indent=2,
            )

        @tool
        def get_hotel_booking(location: str, nights: int) -> str:
            """Get hotel booking information."""
            hotels = {
                "nairobi": {
                    "name": "Nairobi Serena Hotel",
                    "pricePerNight": 180,
                    "totalNights": nights,
                    "totalPrice": 180 * nights,
                    "currency": "USD",
                }
            }

            return json.dumps(
                hotels.get(location.lower(), {"error": "Location not found"}),
                indent=2,
            )

        @tool
        def convert_currency(
            amount: float,
            from_currency: str,
            to_currency: str,
        ) -> str:
            """Convert currency."""
            rates = {
                "USD": {"USD": 1, "NGN": 1550, "KES": 160},
                "NGN": {"USD": 0.00065, "NGN": 1, "KES": 0.103},
                "KES": {"USD": 0.00625, "NGN": 9.69, "KES": 1},
            }

            rate = rates.get(from_currency.upper(), {}).get(
                to_currency.upper(),
                1,
            )

            converted = amount * rate

            result = {
                "originalAmount": amount,
                "convertedAmount": round(converted, 2),
                "exchangeRate": rate,
                "targetCurrency": to_currency.upper(),
            }

            return json.dumps(result, indent=2)

        return [
            rag_query,
            get_flight_booking,
            get_hotel_booking,
            convert_currency,
        ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hybrid RAG Agent CLI"
    )
    parser.add_argument(
        "prompt",
        nargs="+",
        help="Query to send to the agent",
    )
    args = parser.parse_args()
    user_query = " ".join(args.prompt)

    print("\n=== Initializing System ===\n")

    rag = RAGSystem()
    rag.load_data_dir("data")
    rag.build_retriever()

    tools = rag.create_tools()

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    agent_prompt = """You are a helpful travel assistant with document search and booking tools.
    Use RAG tool for document questions, booking tools for flights/hotels, currency tool for conversions.
    Always check documents first if relevant. Respond concisely with JSON when possible."""

    agent = create_agent(
        llm,
        tools,
        system_prompt=agent_prompt,
    )

    print("\n=== Running Query ===\n")
    print("Q:", user_query)

    result = agent.invoke({
        "messages": [
            ("user", user_query)
    ]
    })

    rag.memory.add(user_query, result["messages"][-1].content)

    print("\n=== Agent Response ===\n")
    print(result["messages"][-1].content)

    print("\n=== Retrieved Memory ===\n")
    history = rag.memory.search(user_query)
    print(format_docs(history))
