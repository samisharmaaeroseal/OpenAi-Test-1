# A basic example of how to interact with out ChatCompletion endpoint
# It requires a list of "messages", consisting of a "role" (one of system, user or assistant) and "content"

question = 'How can you help me'


completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": question}
  ]
)
print(f"{completion['choices'][0]['message']['role']}: {completion['choices'][0]['message']['content']}")

from email import contentmanager
from pyexpat import model
from termcolor import colored

# A class to create a message as a dict for chat

class Message:

    def __init__(self, role, content):

        self.role = role
        self.content = content
    def message(self):
        return {"role": self.role,"content": self.content}

    #Assistant class to help converse with bot

    class Assistant:
        def __init__(self):
            self.conversation_history = []

        def _get_assistant_response(self, prompt):
            try:
                completion = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo",
                    messages = prompt
                )
                
                response_message = Message(completion['choices'][0]['message']['role'],completion['choices'][0]['message']['content'])
                return response_message.message()
            except Exception as e:
                return f'Request failed with exception {e}'

            def ask_assistant(self, next_user_prompt, colorsize_assistant_replies = True):
                [self.conversation_history.append(x) for x in next_user_prompt]
                assistant_response = self._get_assistant_response(self.conversation_history)
                self.conversation_history.append(assistant_response)
                return assistant_response

            def pretty_print_conversation_history(self, colorize_assistant_replies = True):
                for entry in self.conversation_history:
                    if entry in self.conversation_history:
                        pass
                    else:
                        prefix = entry['role']
                        content = entry['content']
                        output = colored(prefix + ':\n' + content, 'green') if colorize_assitant_replies and entry['role'] == 'assistant' else prefix + ':\n' + content 
                        print(output)

# Initiate Assistant class

conversation = Assistant()

#Create a list to hold our messages and insert both a system message to guide behavior and our first user question

messages = []
system_message = Message('system', 'You are helpful business assistant who has innovative ideas')
user_message = Message('user', 'What can you do to help me')
messages.append(system_message.message())
messages.append(user_message.message())
messages

#Get back a response from the chat bot to question
response_message = conversation.ask_assistant(messages)
print(response_message['content'])

next_question = 'Tell me more about option 2'

# Initiate a fresh messages list and insert next question
messages = []
user_message = Message('user', next_question)
messages.append(user_message.message())
response_message = conversation.ask_assistant(messages)
print(response_message['content'])

# Print out a log of conversation so far
conversation.pretty_print_conversation_history


# Updated system prompt requiring Question and Year to be extracted from the user
system_prompt = '''
You are a helpful AeroSeal Dealer Support knowledge base assistant. You need to capture a Question from each customer.
The Question is their query on the technical information regarding the AeroSeal residential and commercial equipment. 

Example 1:

User: Can pregnant women be at the sealing site?

Assistant: Searching for answers.
'''

# New assistant class to add a vector database call to its responses
class RetrievalAssistant:
    def __init__(self):
        self.conversation_history = []
    def _get_assistant_response(self, prompt):
        try:
            completion = openai.ChatCompletion.create(
                model = CHAT_MODEL,
                messages = prompt,
                temperature = 0.1
            )

            response_message = Message(completion['choices'][0]['message']['role'], completion['choices'][0]['message']['content'])
            return response_message.message()

        except Exception as e:
            return f'Request failed with exception {e}'

    # The function to retrieve Redis search results
    def _get_search_results(self,prompt):
        latest_question = prompt
        search_content = get_redis_results(redis_client,latest_question,INDEX_NAME)['result'][0]
        return search_content
        

    def ask_assistant(self, next_user_prompt):
        [self.conversation_history.append(x) for x in next_user_prompt]
        assistant_response = self._get_assistant_response(self.conversation_history)
        
        # Answer normally unless the trigger sequence is used "searching_for_answers"
        if 'searching for answers' in assistant_response['content'].lower():
            question_extract = openai.Completion.create(model=COMPLETIONS_MODEL,prompt=f"Extract the user's latest question and the year for that question from this conversation: {self.conversation_history}. Extract it as a sentence stating the Question and Year")
            search_result = self._get_search_results(question_extract['choices'][0]['text'])
            
            # We insert an extra system prompt here to give fresh context to the Chatbot on how to use the Redis results
            # In this instance we add it to the conversation history, but in production it may be better to hide
            self.conversation_history.insert(-1,{"role": 'system',"content": f"Answer the user's question using this content: {search_result}. If you cannot answer the question, say 'Sorry, I don't know the answer to this one'"})
            #[self.conversation_history.append(x) for x in next_user_prompt]
            
            assistant_response = self._get_assistant_response(self.conversation_history)
            print(next_user_prompt)
            print(assistant_response)
            self.conversation_history.append(assistant_response)
            return assistant_response
        else:
            self.conversation_history.append(assistant_response)
            return assistant_response

    def pretty_print_conversation_history(self, colorize_assistant_replies = True):
        for entry in self.conversation_history:
            if entry['role'] == 'system':
                pass
            else : 
                prefix = entry['role']
                content = entry['content']
                output = colored(prefix + ':\n' + content, 'green') if colorize_assistant_replies and entry['role'] == 'assistant' else prefix + ':\n' + content
                #prefix = entry['role']
                print(output)

conversation = RetrievalAssistant()
messages = []
system_message = Message('system',system_prompt)
user_message = Message('user','What is the Safety Certification for the Sealant Material Specifications?')
messages.append(system_message.message())
messages.append(user_message.message())
response_message = conversation.ask_assistant(messages)
response_message

messages = []

response_message = conversation.ask_assistant(messages)
#response_message


#experiment - to be deleted
response = openai.ChatCompletion.create(
    model1 = MODEL
    messages = [
        {"role": "system", "content" : "You are a friendly and helpful teaching assistant. You explain concepts in greate depth"}
        {"role": "user", "content": "Can you explain how fractions work?"}, 
    ], 
    termperature = 0;

)

print(response["choices"][0]["message"]["content"])

response = openai.ChatCompletion.create(
    model = model
    messages = [
        {"role": "system", "content": "You are a helpful, pattern-following assistant."},
        {"role": "user", "content": "Help me translate the following corporate jargon into plain English."},
        {"role": "assistant", "content": "Sure, I'd be happy to!"},
        {"role": "user", "content": "New synergies will help drive top-line growth."},
        {"role": "assistant", "content": "Things working well together will increase revenue."},
        {"role": "user", "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."},
        {"role": "assistant", "content": "Let's talk later when we're less busy about how to do better."},
        {"role": "user", "content": "This late pivot means we don't have time to boil the ocean for the client deliverable."},
        ], 
    temperature = 0, 
)

print(response["choices"][0]["message"]["content"])







                    

                

