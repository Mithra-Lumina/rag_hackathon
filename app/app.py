import os
import gradio as gr
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_chroma import Chroma

# === Setup & Retrieval Logic ===

def load_environment():
    load_dotenv()
    return {
        "API_KEY": os.getenv("API_KEY"),
        "IBM_URL": os.getenv("URL"),
        "PROJECT_ID": os.getenv("PROJECT_ID")
    }

def get_embedder(api_key, url, project_id):
    return WatsonxEmbeddings(
        url=url,
        apikey=api_key,
        project_id=project_id,
        model_id="ibm/granite-embedding-278m-multilingual"
    )

def initialize_vector_store(embedder):
    return Chroma(
        collection_name="climate_edu",
        persist_directory="data/chroma_db",
        embedding_function=embedder
    )

def retrieve_relevant_docs(vector_store, query, k=2):
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def format_context(docs):
    return "\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(docs)])

def build_prompt(context, user_query):
    return f"""<|begin_of_text|><|header_start|>system<|header_end|>
You are a teacher who provide information on climate change, its impacts, and actions individuals can take to mitigate it. Use the provided context to answer each user question as accurately and briefly as possible. If the context does not contain the answer, respond with: "Sorry, I cannot assist you."
Answer the queries with the same user's input language no matter the context's language.
Always answer in markdown format, using bullet points for lists and **bold** for important facts. If the context is not sufficient to answer the question, politely inform the user that you cannot assist them.
Keep the answers like you are teaching and add a little interaction to the answer, like asking a question or giving a suggestion to the user.

Context:
{context}<|eot|>
<|header_start|>user<|header_end|>

{user_query}<|eot|>
<|header_start|>assistant<|header_end|>
"""

def get_llm(api_key, url, project_id):
    return WatsonxLLM(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        url=url,
        apikey=api_key,
        project_id=project_id,
        params={"max_new_tokens": 1024, "decoding_method": "greedy"}
    )

def chatbot_interface(user_query, prev_output):
    config = load_environment()
    embedder = get_embedder(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])
    vector_store = initialize_vector_store(embedder)
    llm = get_llm(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])

    docs = retrieve_relevant_docs(vector_store, user_query)
    context = format_context(docs)
    prompt = build_prompt(context, user_query)
    response = llm.invoke(prompt)

    updated_output = f"{prev_output}\n\n### ❓ {user_query}\n{response.strip()}"
    return updated_output

# === Gradio Interface ===

def main():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Climate Change Assistant")
        gr.Markdown("Ask about the causes, effects, and solutions for climate change.")

        with gr.Row():
            input_box = gr.Textbox(
                label="Ask a question about climate change",
                placeholder="Type your question here...",
                lines=1,
                scale=4
            )
            flag_btn = gr.Button("Flag", scale=1)

        output_box = gr.Markdown(value="### Climate Assistant\nWelcome! Ask your question below 👇")

        with gr.Row():
            clear_btn = gr.Button("Clear")
            submit_btn = gr.Button("Submit")

        # Submit functionality
        submit_btn.click(
            fn=chatbot_interface,
            inputs=[input_box, output_box],
            outputs=output_box
        )

        # Clear input after submit
        submit_btn.click(
            fn=lambda: "",
            inputs=None,
            outputs=input_box
        )

        # Clear all
        clear_btn.click(
            fn=lambda: ["", ""],
            inputs=None,
            outputs=[input_box, output_box]
        )

    demo.launch()

if __name__ == "__main__":
    main()
