import streamlit as st
from streamlit_chat import message

from Database import get_redis_connection
from Assistants import RetrievalAssistant, Message

# Initialise database

## Initialise Redis connection
redis_client = get_redis_connection()

# Set instruction

# System prompt requiring Question to be extracted from the user
system_prompt = '''
You are a helpful Aeroseal knowledge base assistant. You need to capture a Question from each customer.
The Question is their query on usiong Aeroseal products.
Think about this step by step:
- The user will ask a Question
- You will generate and respond with the correct answer

'''

### CHATBOT APP

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

st.title('Aeroseal Dealer Training Chatbot')
st.subheader("Help us help you learn about using Aeroseal products")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(question):
    response = st.session_state['chat'].ask_assistant(question)
    return response

prompt = st.text_input(f"What do you want to know: ", key="input")

if st.button('Submit', key='generationSubmit'):

    # Initialization
    if 'chat' not in st.session_state:
        st.session_state['chat'] = RetrievalAssistant()
        messages = []
        system_message = Message('system',system_prompt)
        messages.append(system_message.message())
    else:
        messages = []


    user_message = Message('user',prompt)
    messages.append(user_message.message())

    response = query(messages)

    # Debugging step to print the whole response
    #st.write(response)

    st.session_state.past.append(prompt)
    st.session_state.generated.append(response['content'])

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
