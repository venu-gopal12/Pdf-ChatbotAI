import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import init_chat_model
import os

# Set API Key
os.environ['MISTRAL_API_KEY'] = ''


def getText(docs):
    text = ""
    for doc in docs:
        pdfReader = PdfReader(doc)
        for page in pdfReader.pages:
            text += page.extract_text() or ""
    return text


def getChunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)


def getDb(chunks):
    encoder = HuggingFaceBgeEmbeddings()
    db = Chroma.from_texts(chunks, encoder)
    return db


def getChain(db):
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
    model = init_chat_model("mistral-large-latest", model_provider="mistralai")
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=db.as_retriever(),
        memory=memory
    )


def main():
    st.set_page_config(
        page_title="Chat with your PDFs",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Chat with Your PDFs")
    st.markdown("Upload PDF files, process them into a searchable knowledge base, and ask questions interactively.")

    # Sidebar for PDF Upload
    with st.sidebar:
        st.header("📂 Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload one or more PDF documents",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("🔍 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing your PDFs..."):
                    pdf_text = getText(pdf_docs)
                    chunks = getChunks(pdf_text)
                    db = getDb(chunks)
                    st.session_state.chain = getChain(db)
                st.success("✅ PDFs processed successfully! Start asking questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    # Main Chat Section
    if "chain" in st.session_state:
        query = st.text_input("💬 Ask a question about your PDFs:")

        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({'question': query})

            # Extract only answer text
            answer = response.get("answer", response)

            # Store chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", answer))

        # Display conversation history with styled bubbles
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.subheader("💬 Conversation")
            for role, msg in st.session_state.chat_history:
                if role == "You":
                    st.markdown(
                        f"<div style=' padding:10px; "
                        f"border-radius:10px; margin:5px 0; text-align:right;'>"
                        f"<b>🧑 You:</b> {msg}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='; padding:10px; test-color:black "
                        f"border-radius:10px; margin:5px 0; text-align:left;'>"
                        f"<b>🤖 Bot:</b> {msg}</div>",
                        unsafe_allow_html=True
                    )


if __name__ == '__main__':
    main()
