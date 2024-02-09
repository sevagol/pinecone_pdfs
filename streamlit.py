import streamlit as st
from openai import OpenAI
import openai
from pinecone import Pinecone
client1 = OpenAI(api_key=st.secrets.openai_key)
from pinecone import Pinecone
pc = Pinecone(api_key=st.secrets.pinecone_key)
index = pc.Index("pdf-index")

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client1.embeddings.create(input = [text], model=model).data[0].embedding
def create_query_body(query_vector):
    query_body = {
        'namespace': 'example-namespace',
        'top_k': 2,
        'vector': query_vector
    }

    return query_body
def query_pinecone(question):
    query_vector = get_embedding(question)
    query = create_query_body(query_vector)
    query_response = index.query(**query, include_metadata=True)
    texts = [match['metadata']['text'] for match in query_response['matches']]
    return texts

def create_prompt(question, document_content):
    return 'You are given a document and a question. Your task is to answer the question based on the document.\n\n' \
           'Document:\n\n' \
           f'{document_content}\n\n' \
           f'Question: {question}'

def get_answer_from_openai(question):
    client = OpenAI(
    # This is the default and can be omitted
    api_key=st.secrets.openai_key)
    document_content = query_pinecone(question)
    prompt = create_prompt(question, document_content)
    print(f'Prompt:\n\n{prompt}\n\n')
    completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-3.5-turbo",
)
    return completion.choices[0].message.content



st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with pdfs 💬")
st.info("Here you can ask chatbot about processed patents", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about patents"}
    ]


# if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
#         st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_answer_from_openai(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history