import streamlit as st
from langchain.chat_models.bedrock import BedrockChat
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Chat with Bedrock", page_icon=":robot:")
st.header("Hey, I'm Bedrock")

if "sessionMessages" not in st.session_state:
    st.session_state.sessionMessages = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]


def load_answer(question):
    st.session_state.sessionMessages.append(HumanMessage(content=question))
    assistant_answer = chat(st.session_state.sessionMessages)
    st.session_state.sessionMessages.append(AIMessage(content=assistant_answer.content))
    return assistant_answer.content


def get_text():
    input_text = st.text_input("Question: ", key=input)
    return input_text


chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0.1})

user_input = get_text()
submit = st.button('Generate')

if submit:
    response = load_answer(user_input)
    st.subheader("Answer:")

    st.write(response, key=1)
