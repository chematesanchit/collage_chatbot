from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINE_API_KEY = st.secrets["PINE_API_KEY"]
openai.api_key = OPENAI_API_KEY
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key=PINE_API_KEY, environment="gcp-starter")
index = pinecone.Index('collagechatbot')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.2,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
