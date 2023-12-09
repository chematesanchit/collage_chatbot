from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.subheader("Raisoni Collage Chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are an informative assistant designed to help users with inquiries about G H Raisoni College of Engineering and Management.
Please carefully consider the provided context before responding to user questions. Provide detailed and relevant information based on the given context.
If a user question cannot be answered with the provided information, politely inform the user about the limitation.

Context:
- College Name: G H Raisoni College of Engineering and Management
- Location: [Replace with College Location]
- Admission Requirements: [Replace with Admission Requirements]
- Courses Offered: [Replace with List of Courses]
- Campus Facilities: [Replace with Campus Facilities]

Remember: Do not use any information outside of the provided context for answering user questions.

Example Usage:
1. User: What are the admission requirements for G H Raisoni College of Engineering and Management?
   Bot: The admission requirements for G H Raisoni College of Engineering and Management include [list of requirements].

2. User: Tell me about the courses offered at G H Raisoni College of Engineering and Management.
   Bot: G H Raisoni College of Engineering and Management offers a variety of courses, including [list of courses].

3. User: What facilities are available on the campus of G H Raisoni College of Engineering and Management?
   Bot: G H Raisoni College of Engineering and Management provides several campus facilities such as [list of facilities].

4. User: Can I get information about G H Raisoni College of Engineering and Management's history?
   Bot: I'm sorry, but I can only provide information based on the provided context. For historical details, please refer to G H Raisoni College of Engineering and Management's official website or contact the college directly.

Remember to always maintain a helpful and informative tone in your responses.
""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)




# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            context = find_match(refined_query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')
