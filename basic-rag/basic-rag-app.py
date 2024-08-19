import streamlit as st
from qa_chat import process_query
from config import load_configs

configs = load_configs()
TITLE = configs.get("title", "My Personal Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


def generate_ai_response(user_text):
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)
    ai_response = process_query(user_text)
    st.session_state.chat_history.append({"role": "ai", "content": ai_response})
    st.chat_message("ai").markdown(ai_response)


def display_chat_history():
    for message in st.session_state.chat_history:
        st.chat_message(message["role"], ).write(message["content"])


def get_user_input():
    return st.chat_input("", key="user_input")    


def main():
    st.set_page_config(
        page_title=TITLE, page_icon=":star:", layout="centered"
    )

    # Create a custom title
    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50; font-size: 50px;'>
            IT Returns Personal Assistant 🌟
        </h1>
        <h2 style='text-align: center; color: #555;'>
            FY - 2024-2025
        </h2>
    """,
        unsafe_allow_html=True,
    )


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append(
            ({"role": "ai", "content": "Hi, I m a bot. How can I help you today."})
        )

    display_chat_history()

    user_text = get_user_input()

    if user_text:
        generate_ai_response(user_text)


if __name__ == "__main__":
    main()
