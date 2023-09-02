from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st
from handlers import StreamHandler
import os
from streamlit_chatbox import *
import time
from langchain.schema import ChatMessage
# chatgpt api inference
# https://platform.openai.com/docs/api-reference/chat/create

# query=st.text_input("input your query",value="Tell me a joke")
# ask_button=st.button("ask") 

# st.markdown("### streaming box")
# # here is the key, setup a empty container first
# chat_box=st.empty() 
# stream_handler = StreamHandler(chat_box)
# chat = ChatOpenAI(max_tokens=25, streaming=True, callbacks=[stream_handler])

# st.markdown("### together box")  

# if query and ask_button: 
#     response = chat([HumanMessage(content=query)])    
#     llm_response = response.content  
#     st.markdown(llm_response)
    
    
def main_pages():

    st.set_page_config(page_title="LLM + Streamlit Prompt Tuning Tool", layout="wide")
    # build streamHandler & OpenAIChat
    
    with st.sidebar:
        selected_model = st.selectbox("gpt-3.5-turbo", ['gpt-3.5-turbo', 'gpt4', 'spark'])
        max_tokens = st.slider("max_tokens", 128, 2048, 512)
        temperature = st.slider("temperature", 0.0, 1.0, 0.7)
        stop = st.text_input("stop words")
        frequency_penalty = st.slider("frequency_penalty", -2.0, 2.0, 0.0)
        presence_penalty = st.slider("presence_penalty",  -2.0, 2.0, 0.0)
        
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="您好，请问我有什么可以帮您？")]

    for msg in st.session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)


        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            model_kwargs = {
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop
            }
            llm = ChatOpenAI(model_kwargs=model_kwargs,temperature=temperature, streaming=True,max_tokens=max_tokens, callbacks=[stream_handler])
            response = llm(st.session_state.messages)
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
            
        
if __name__ == "__main__":
    main_pages()
        
        
    