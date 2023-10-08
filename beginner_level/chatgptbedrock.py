import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.chat_models.bedrock import BedrockChat
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
import boto3
import os
from langchain.llms.bedrock import Bedrock

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}
boto3_bedrock = boto3.client('bedrock-runtime')
llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=boto3_bedrock,
    model_kwargs=inference_modifier,
)


conversation = ConversationChain(llm=llm,
                                 memory=ConversationBufferMemory(ai_prefix="Assistant"))
conversation("Good Morning AI")
conversation("My name is Rafa")
conversation("I am a computer science student at UC Berkeley")
response = conversation("What is my name?")
print(response['response'])
