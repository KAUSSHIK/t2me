import streamlit as st
import aiohttp
import asyncio

# Set page config at the very beginning
st.set_page_config(page_title="Kausshik's Assistant", page_icon="🤖", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.chat-message {
    padding: 1rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .message {
    width: 100%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Kausshik's Assistant")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat messages
def display_messages():
    for message in st.session_state['messages']:
        with st.container():
            if message['role'] == 'user':
                st.markdown(f'<div class="chat-message user"><div class="message">🧑‍💼 You: {message["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot"><div class="message">🤖 Assistant: {message["content"]}</div></div>', unsafe_allow_html=True)

async def send_message(question):
    url = 'https://t2me-be-5523b55adae3.herokuapp.com/chat'  # Replace with your backend URL when deployed
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={'question': question}) as resp:
            data = await resp.json()
            return data['answer']

async def main():
    st.sidebar.title("About")
    st.sidebar.info(
        "This AI assistant is designed to help recruiters learn about Kausshik Manojkumar. "
        "It uses information from Kausshik's resume and LinkedIn profile to answer your questions."
    )
    
    display_messages()

    # User input
    with st.container():
        question = st.text_input("Ask a question about Kausshik's experience:", key="user_input")
        send_button = st.button("Send")

    if send_button and question:
        with st.spinner("Thinking..."):
            st.session_state['messages'].append({'role': 'user', 'content': question})
            answer = await send_message(question)
            st.session_state['messages'].append({'role': 'bot', 'content': answer})
        st.rerun()

if __name__ == "__main__":
    asyncio.run(main())