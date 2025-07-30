import gradio as gr
from rag_pipeline.rag_engine import RAGEngine

# Initialize your RAG engine
rag = RAGEngine()

# Define interaction function
def answer_question(question: str):
    result = rag.query(question)
    answer = result["answer"]
    contexts = "\n\n".join([f"• {ctx['text']}" for ctx in result["contexts"]])
    return answer, contexts

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧬 Biology Conference Q&A")
    gr.Markdown("Ask a question based on the conference abstracts:")

    with gr.Row():
        question_input = gr.Textbox(label="Your Question", placeholder="e.g., What were the key topics in tumor immunotherapy?")
    
    with gr.Row():
        answer_output = gr.Textbox(label="Generated Answer")
        context_output = gr.Textbox(label="Retrieved Contexts", lines=10)

    submit_btn = gr.Button("Get Answer")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input],
        outputs=[answer_output, context_output]
    )

# Launch app
if __name__ == "__main__":
    demo.launch()
