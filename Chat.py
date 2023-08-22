import streamlit as st
from streamlit_chat import message

from Database import get_redis_connection
from Assistants import RetrievalAssistant, Message

# Initialise database

## Initialise Redis connection
redis_client = get_redis_connection()


# System prompt requiring Question to be extracted from the user
system_prompt = '''
You are a helpful Aeroseal product knowledge base assistant. 

Begin by introducing yourself and asking the user for the product being used. 

Once you have the product being used, ask "What can I help you with?"

Use the product being used to answer any questions the dealer has.

Once you know the product, say "Thank you". Answer all following questions with this information.

Example 1:

User: How do I know when to stop sealing?

Assistant: I can help you with that. Which AeroSeal product are you using?

User: HSC 4

Assistant: Thank you. Sealing should be terminated in the following conditions:

            1. Duct leakage has been reduced below the target level
            2. Fan flow cannot be maintained above 70 cfm due to duct pressure constraints
            3. duct pressure has reached 600 Pa
            4. graph shows a "flat-line" that cannot be remedied

User: How frequently should I check the sealing progress

Assistant: Using HSC, be sure to check progress of the sealing job every 10 minutes

'''

# Streamlit Browser for ChatBot App

st.set_page_config(
    page_title="Streamlit Chat",
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
