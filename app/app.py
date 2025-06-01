import os
import gradio as gr
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings, ChatWatsonx
from langchain_chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

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
        params={"max_new_tokens": 1024, "decoding_method": "greedy"},
    )

def user_fn(user_message, chat_history):
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=user_message, interactive=True)

config = load_environment()
embedder = get_embedder(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])
vector_store = initialize_vector_store(embedder)

def chatbot_interface(user_message, chat_history):

    chat_history = chat_history or []
    chat_history.append({"role": "assistant", "content": ""})

    llm = get_llm(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])

    docs = retrieve_relevant_docs(vector_store, user_message)
    context = format_context(docs)
    history = format_chat_history(chat_history)
    prompt = build_prompt(context, user_message, history)

    streamer = llm.stream(prompt)
    for chunk in streamer:
        chat_history[-1]["content"] += chunk.content
        yield chat_history, gr.Textbox(value="", interactive=True)


with gr.Blocks(css_paths="./app/styles.css", theme=gr.Theme.from_hub("JohnSmith9982/small_and_pretty")) as demo:
    gr.Markdown(elem_id="header-main", value="# 🌍 Learn more about Climate Change 🌍")
    with gr.Row():
        with gr.Group():
            chatbot = gr.Chatbot(show_label=False, elem_id="chatbot-main", min_height="650px", type="messages")
            user_input = gr.Textbox(
                placeholder="Type your question here...",
                lines=1,
                show_label=False,
                elem_classes="input",
            )
            send_btn = gr.Button("Send")
        with gr.Group():
            with gr.Row():
                gr.TextArea(show_copy_button=True,show_label=False, value= "")
            with gr.Row():
                gr.TextArea(show_copy_button=True,show_label=False, value= "")
                

                demo.load(None, None, None, js="() => { document.body.classList.remove('dark'); }")
    send_btn.click(
        fn=user_fn,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    ).then(
        fn= chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
        )
    user_input.submit(
        fn=user_fn,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
    ).then(
        fn= chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input]
        )

if __name__ == "__main__":
    demo.launch()