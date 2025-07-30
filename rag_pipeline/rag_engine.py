from typing import List, Dict
from retriever import Retriever
from generator import generate_answer

class RAGEngine:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever(top_k=top_k)
        self.retriever.load_index()

    def query(self, question: str) -> Dict:
        # Step 1: Retrieve top-k relevant chunks
        retrieved_chunks = self.retriever.retrieve(question)

        # Step 2: Generate answer using Mistral
        answer = generate_answer(question, retrieved_chunks)

        return {
            "question": question,
            "answer": answer,
            "contexts": retrieved_chunks
        }

# CLI example
if __name__ == "__main__":
    rag = RAGEngine(top_k=3)
    while True:
        question = input("\n🔍 Enter a question (or 'exit'): ")
        if question.lower() in {"exit", "quit"}:
            break
        result = rag.query(question)
        print("\n📢 Answer:\n", result["answer"])
