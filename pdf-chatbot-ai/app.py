import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


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


@st.cache_resource  # caches the model across reruns
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # smaller/faster


def getDb(chunks):
    encoder = load_embeddings()
    return Chroma.from_texts(chunks, encoder)


def getChain(db):
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=5, return_messages=True
    )
    model = ChatGroq(model_name="llama-3.1-8b-instant")
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=db.as_retriever(),
        memory=memory
    )


def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon="📚", layout="wide")
    st.title("📚 Chat with Your PDFs")

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
                st.success("✅ Done! Ask away.")
            else:
                st.warning("Please upload at least one PDF.")

    if "chain" in st.session_state:
        query = st.text_input("💬 Ask a question about your PDFs:")
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke({"question": query})
            answer = response.get("answer", response)

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", answer))

        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.subheader("💬 Conversation")
            for role, msg in st.session_state.chat_history:
                if role == "You":
                    st.markdown(
                        f"<div style='padding:10px; border-radius:10px; margin:5px 0; text-align:right;'>"
                        f"<b>🧑 You:</b> {msg}</div>", unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='padding:10px; color:black; border-radius:10px; margin:5px 0;'>"
                        f"<b>🤖 Bot:</b> {msg}</div>", unsafe_allow_html=True
                    )


if __name__ == "__main__":
    main()
