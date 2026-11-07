import streamlit as st
from chatbot_youtube_chain import build_chain

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube Chatbot using RAG")

st.write(
    "Paste a YouTube video URL and ask questions "
    "about its content."
)

# Cache chain separately for each video URL
@st.cache_resource(show_spinner=False)
def load_rag_chain(video_url):
    return build_chain(video_url)

# YouTube URL input form
with st.form("youtube_form"):
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    process_video = st.form_submit_button(
        "🚀 Process Video"
    )

# Process video
if process_video:
    if not video_url.strip():
        st.warning("Please enter a YouTube video URL.")
    else:
        try:
            with st.spinner(
                "Fetching transcript and creating vector database..."
            ):
                # Validate and build chain
                rag_chain = load_rag_chain(video_url.strip())
                st.session_state.video_url = video_url.strip()
                st.session_state.messages = []

            st.success("Video processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Unable to process video: {e}")

# Chat interface
if st.session_state.get("video_url"):
    chain = load_rag_chain(st.session_state.video_url)

    st.divider()

    st.subheader("💬 Chat with your video")

    # Display previous messages.
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message["content"])

    # Accept a new question.
    question = st.chat_input("Ask a question about the video...")

    if question:
        st.session_state.messages.append({
            'role': 'user',
            'content': question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Generating an answer..."):
                try:
                    answer = chain.invoke(question)
                    st.markdown(answer)

                    st.session_state.messages.append({
                        'role':'assistant',
                        'content': str(answer)
                    })
                except Exception as exc:
                    st.error(f"An error occurred while generating the answer: {exc}")

if st.sidebar.button("clear chat"):
    st.session_state.messages = []
    st.rerun()