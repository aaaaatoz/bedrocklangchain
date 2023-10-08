import boto3
import boto3

import streamlit as st
from langchain.embeddings import BedrockEmbeddings


from langchain.vectorstores import FAISS
from datetime import datetime

st.set_page_config(page_title="Find Similar Things", page_icon="🤖", layout="wide")
st.header("Hey, Find Similar Things")
boto3_bedrock = boto3.client("bedrock-runtime")
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='./beginner_level/4000-most-common-english-words-csv.csv', csv_args={
    'delimiter': ',',
})

data = loader.load()

print(datetime.now())
print("beginning load")
db = FAISS.from_documents(data, embeddings)
print(datetime.now())
print("load done")

def get_text():
    input_text = st.text_input("Your input: ", key= input)
    return input_text


user_input=get_text()
submit = st.button('Find similar words')

if submit:
    
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0])
    st.text(docs[1].page_content)

