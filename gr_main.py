import os
import shutil
import gradio as gr

from chatting_model import model_chat_ollama
from reprocessing import (
    pdf_reprocessing,
    docx_reprocessing
)

from db import (
    langchain_chroma
)

from retrievers import (
    retrievers_multiquery,
    retrievers_normal
)


def upload_file(files):
    """
    Process the uploaded files by copying them to the storage path.

    Args:
        files: List of uploaded files.

    Returns:
        List of file paths of the uploaded files.
    """
    storage_path = './storage'
    file_paths = [file.name for file in files]

    for file in files:
        extension = file.name.split('.')[-1]
        abs_file_path = os.path.join(storage_path, file.name)
        shutil.copy(file, storage_path)

        if extension == 'pdf':
            documents_splits = pdf_reprocessing.processing(file_path=abs_file_path, using_splitter=True)
        elif extension == 'docx':
            documents_splits = docx_reprocessing.processing(file_path=abs_file_path, using_splitter=True)
        else:
            raise ValueError
        list_idx = langchain_chroma.add_documents(
            documents=documents_splits
        )

        show_message = f"""
            Extension {extension} === {file.name} === Splitter: {len(documents_splits)} === LENGTH IDX: {len(list_idx)}
        """
        gr.Info(message=show_message)
    return file_paths


def show_all_storage(files):
    storage_path = './storage'
    all_files = list_files_in_folder(storage_path)
    return show_files(all_files)


def list_files_in_folder(folder_path):
    """
    List all files in the given folder.

    Args:
        folder_path: Path to the folder.

    Returns:
        List of file names in the folder.
    """
    try:
        files = os.listdir(folder_path)
        files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]
        return files
    except FileNotFoundError:
        return []


def show_files(files):
    """
    Args:
        None

    Returns:
        List of file path of storage in folder
    """
    return '\n'.join(files)


def rag_chain(url, question, using_multiquery):
    if using_multiquery:
        chat = model_chat_ollama.do_activate(
            query=str(question),
            retriever=retrievers_multiquery.get_relevant_documents()
        )
    else:
        chat = model_chat_ollama.do_activate(
            query=question,
            retriever=retrievers_normal.get_relevant_documents()
        )
    return str(chat)


with gr.Blocks() as all_interfaces:
    gr.Markdown(
        """
        # Testing uploading and receiving the uploaded file
        """
    )

    # upload file
    file_output_uploaded_button = gr.File()
    upload_button = gr.UploadButton("Upload file", file_types=[".pdf", ".csv", ".docx"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output_uploaded_button)

    # show database file
    show_file_output = gr.Textbox(label="Storage Files")
    gr_button = gr.Button("Show all files")
    gr_button.click(fn=show_all_storage, inputs=gr_button, outputs=show_file_output)

    # Gradio interface
    iface = gr.Interface(
        fn=rag_chain,
        inputs=[
            gr.Textbox(label="URL"),
            gr.Textbox(label="Query"),
            gr.Checkbox(label="Using MultiQuery")
        ],
        outputs="text",
        title="RAG Chain Question Answering",
        description="Enter a URL and a query to get answers from the RAG chain."
    )

if __name__ == "__main__":
    all_interfaces.launch(show_api=False, share=True)
