import os
import gradio as gr
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings, ChatWatsonx
from langchain_chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)
import fitz
import utils
load_dotenv()

class StreamingGradioCallbackHandler(BaseCallbackHandler):
    def __init__(self, chatbot_messages_list):
        self.chatbot_messages_list = chatbot_messages_list
        self.current_bot_message_content = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.current_bot_message_content += token
        if self.chatbot_messages_list and self.chatbot_messages_list[-1]['role'] == 'assistant':
            self.chatbot_messages_list[-1]['content'] = self.current_bot_message_content
        else:
            pass

def load_environment():
    return {
        "API_KEY": os.getenv("API_KEY"),
        "IBM_URL": os.getenv("URL"),
        "PROJECT_ID": os.getenv("PROJECT_ID")
    }
config = load_environment()

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
        persist_directory="app/data/chroma_db",
        embedding_function=embedder
        )

embedder = get_embedder(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])
vector_store = initialize_vector_store(embedder)
def retrieve_relevant_docs(query, k=5):
    docs = vector_store.similarity_search_with_relevance_scores(query, k=k, score_threshold=0.65)
    page_content = [doc[0].page_content for doc in docs]
    page_number = [doc[0].metadata['page'] for doc in docs]
    file_name = [doc[0].metadata['file_name'] for doc in docs]
    gallery_images(file_name, page_number)
    return page_content, file_name, page_number


def format_context(docs):
    return "\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(docs)])

def format_chat_history(chat_history, limit=10):
    recent_history = chat_history[-limit:]
    formatted = ""
    for msg in recent_history:
        if msg["role"] == "user":
            formatted += f"user:\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted += f"assistant:\n{msg['content']}\n"
    return formatted

def build_prompt(context, user_query, history):
    SYSTEM_PROMPT = f"""
        You are an climate educational system, that only helps user's with topics related to climate change and related environmental concepts.
        Create a comprehensive but preliminary road map from the following context to suggest user's whom do not know about climate change and ask about it.
        Context is bellow: 
        {context}
        Provide accurate, interactive, and easily understandable answers about climate change, its impacts, and practical actions individuals can take to mitigate it. Tailor responses for educational clarity, engaging learners through explanations and examples suitable for diverse age groups and backgrounds.
        Provide your answers in Markdown format and when applicable illustrate with tables.
        When a user's query seems unclear or possibly a mistake, notify them briefly and politely without verbose explanations.
        Chat history is bellow:
        {history}
        Context is provided below. Use it to answer the user's question. If the context does not contain the answer, respond with: "Sorry, I don't know."
        Context:
        {context}
        """
    system_message = SystemMessage(
        content=SYSTEM_PROMPT
        )
    human_message = HumanMessage(
        content=user_query
        )
    return [system_message, human_message]


def get_llm(api_key, url, project_id):
    return ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=url,
        apikey=api_key,
        project_id=project_id,
        params={"max_new_tokens": 100, "decoding_method": "greedy"},
    )

def user_fn(user_message, chat_history):
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=user_message, interactive=True)


llm = get_llm(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])
local_custom_agent = utils.LangGraphApp()
local_custom_agent.set_up(llm)

def chatbot_interface(user_message, chat_history):
    local_custom_agent.query(user_message)
    chat_history = chat_history or []
    chat_history.append({"role": "assistant", "content": ""})

    content, name, page_number = retrieve_relevant_docs(user_message)
    context = "\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(content)])
    history = format_chat_history(chat_history)
    prompt = build_prompt(context, user_message, history)
    streamer = llm.stream(prompt)
    for chunk in streamer:
        chat_history[-1]["content"] += chunk.content
        yield chat_history, gr.Textbox(value="", interactive=True)
def export_page_fitz(file_name, page_number):
    source_path = os.path.join("app/data/climate_edu", file_name + ".pdf")
    doc = fitz.open(source_path)

    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=200)
    os.makedirs("app/output", exist_ok=True)
    output_path = f"app/output/test{page_number}-{file_name}.png"
    pix.save(output_path)


def gallery_images(file_names, page_numbers):
    if len(file_names) >= 1:
        for file_name, page_number in zip(file_names, page_numbers):
            export_page_fitz(file_name, page_number)

    
def show_gallery():
    images = [
        os.path.join("app/output", img)
        for img in os.listdir("app/output")
        if img.endswith(".png")
    ]
    return images

def show_plot():
    images = [
        os.path.join("app/plot_output", img)
        for img in os.listdir("app/plot_output")
        if img.endswith(".png")
    ]
    return images

def cleanup_output():
    if os.path.exists("app/output"):
        for f in os.listdir("app/output"):
            if f.endswith(".png"):
                os.remove(os.path.join("app/output", f))
    if os.path.exists("app/plot_output"):
        for f in os.listdir("app/plot_output"):
            if f.endswith(".png"):
                os.remove(os.path.join("app/plot_output", f))

with gr.Blocks(css_paths="./app/styles.css", theme=gr.Theme.from_hub("JohnSmith9982/small_and_pretty")) as demo:
    gr.Markdown(elem_id="header-main", value="# 🌍 Learn more about Climate Change 🌍")
    with gr.Row(elem_id="chatbot-area"):
        with gr.Group():
            chatbot = gr.Chatbot(show_label=False, elem_id="chatbot-main", type="messages")
            user_input = gr.Textbox(
                placeholder="Type your question here...",
                lines=1,
                show_label=False,
                elem_classes="input",
            )
            send_btn = gr.Button("Send")
        with gr.Group():
            with gr.Row():
                with gr.Group(elem_id="doc-res"):
                    gr.Markdown(elem_id="header-main", value="# 📈 Visuallization and Resources 📄")
                    gallerya = gr.Gallery(
                    value=['app/data/placeholder_img/logo.png'],
                    show_label=False,
                    elem_id="gallerya",
                    object_fit="contain",
                    height="auto",
                    interactive=True
                    )            
            with gr.Row(elem_id="plot-area"):
                gallery = gr.Gallery(
                    value=['app/data/placeholder_img/climate_change.png'],
                    show_label=False,
                    elem_id="gallery",
                    object_fit="contain",
                    height="auto",
                    interactive=True
                )
                demo.load(None, None, None, js="() => { document.body.classList.remove('dark'); }")
    send_btn.click(
        fn=cleanup_output,
        ).then(
        fn=user_fn,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    ).then(
        fn=retrieve_relevant_docs,
        inputs=[user_input],
    ).then(
        fn=show_gallery,
        outputs=[gallery]
    ).then(
        fn= chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
        ).then(
        fn=show_plot,
        outputs=[gallerya]
    )
    user_input.submit(
        fn=cleanup_output,
        ).then(
        fn=user_fn,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    ).then(
        fn=retrieve_relevant_docs,
        inputs=[user_input],
    ).then(
        fn=show_gallery,
        outputs=[gallery]
    ).then(
        fn= chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
        ).then(
        fn=show_plot,
        outputs=[gallerya]
    )
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)