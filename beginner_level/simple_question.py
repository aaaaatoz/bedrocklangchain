import streamlit as st
from langchain.llms.bedrock import Bedrock
import boto3

boto3_bedrock = boto3.client('bedrock-runtime')
def get_answer(question):
    llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':200})
    if question:
        prompt_template = f"""Human:{question}
        Assistant:
        """
        answer = llm(prompt_template)
        return answer
    else:
        return "Please enter a question"


#App UI starts here
st.set_page_config(page_title="LangChain with Bedrock Demo", page_icon=":robot:")
st.header("LangChain meets Bedrock")

#Gets the user input
def get_text():
    input_text = st.text_input("Your question: ", key="input")
    return input_text


user_input=get_text()
response = get_answer(user_input)

submit = st.button('Generate Answer')

if submit:

    st.subheader("Bedrock Response:")

    st.write(response)