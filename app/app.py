import os
import gradio as gr
from dotenv import load_dotenv
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler

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


def build_prompt(context, user_query):
    return f"""<|begin_of_text|><|header_start|>system<|header_end|>
You are a teacher who provides information on climate change, its impacts, and actions individuals can take to mitigate it. Use the provided context to answer each user question as accurately and briefly as possible. If the context does not contain the answer, respond with: \"Sorry, I cannot assist you.\"
Answer the queries with the same user's input language no matter the context's language.
Always answer in markdown format, using bullet points for lists and **bold** for important facts. If the context is not sufficient to answer the question, politely inform the user that you cannot assist them.
Keep the answers like you are teaching and add a little interaction to the answer, like asking a question or giving a suggestion to the user.

Context:
{context}<|eot|>
<|header_start|>user<|header_end|>

{user_query}<|eot|>
<|header_start|>assistant<|header_end|>
"""


def get_llm(api_key, url, project_id, gradio_handler):
    return WatsonxLLM(
        model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        url=url,
        apikey=api_key,
        project_id=project_id,
        streaming=True,
        callbacks=[gradio_handler],
        params={"max_new_tokens": 1024, "decoding_method": "greedy"},
    )


def chatbot_interface(user_message, chat_history):
    config = load_environment()
    embedder = get_embedder(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"])
    vector_store = initialize_vector_store(embedder)

    chat_history = chat_history or []

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})

    gradio_handler = StreamingGradioCallbackHandler(chat_history)
    llm = get_llm(config["API_KEY"], config["IBM_URL"], config["PROJECT_ID"], gradio_handler)

    docs = retrieve_relevant_docs(vector_store, user_message)
    context = format_context(docs)
    prompt = build_prompt(context, user_message)

    for chunk in llm.stream(prompt):
        yield chat_history, gr.Textbox(value="", interactive=True)

    yield chat_history, gr.Textbox(value="", interactive=True)


with gr.Blocks(css_paths="./app/styles.css", theme=gr.Theme.from_hub("JohnSmith9982/small_and_pretty")) as demo:
    gr.Markdown(elem_id="header-main", value="# 🌍 Climate Change Assistant 🌍")
    with gr.Row():
        with gr.Group():
            chatbot = gr.Chatbot(show_label=False, show_copy_button=True, elem_id="chatbot", min_height="650px", type="messages")
            user_input = gr.Textbox(
                placeholder="Type your question here...",
                lines=1,
                show_label=False,
                elem_classes="input",            )
            send_btn = gr.Button("Send")
        with gr.Group():
            with gr.Row():
                gr.TextArea(show_copy_button=True,show_label=False, value= "")
            with gr.Row():
                gr.TextArea(show_copy_button=True,show_label=False, value= "")
                

                demo.load(None, None, None, js="() => { document.body.classList.remove('dark'); }")
    send_btn.click(
        fn=chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
    )
    user_input.submit(
        fn=chatbot_interface,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
    )

if __name__ == "__main__":
    demo.launch()