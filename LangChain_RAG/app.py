import streamlit as st
from chain import build_chain


st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered",
)

st.title("🎥 YouTube Chatbot using RAG")
st.write("Ask questions about the YouTube video transcript.")

# Build the chain once per Streamlit server session/cache.
@st.cache_resource(show_spinner="Fetching transcript and building the RAG index...")
def load_rag_chain():
    return build_chain()


try:
    rag_chain = load_rag_chain()
except Exception as exc:
    st.error(f"Could not initialize the RAG chain: {exc}")
    st.stop()


# Maintain conversation messages in the browser session.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept a new question.
question = st.chat_input("Ask a question about the video...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the transcript and generating an answer..."):
            try:
                answer = rag_chain.invoke(question)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(answer),
                })
            except Exception as exc:
                st.error(f"An error occurred while generating the answer: {exc}")


if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
