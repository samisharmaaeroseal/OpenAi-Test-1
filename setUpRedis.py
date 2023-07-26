import openai
import os
import requests
import numpy as np
import pandas as pd
from typing import Iterator
import tiktoken
import textract
from numpy import array, average

from database import get_redis_connection


# OpenAI Authentication
key = 'OPENAI_API_KEY'
openai.api_key = os.getenv(key)


from config import COMPLETIONS_MODEL, EMBEDDINGS_MODEL, CHAT_MODEL, TEXT_EMBEDDING_CHUNK_SIZE, VECTOR_FIELD_NAME, INDEX_NAME, PREFIX

# Warnings (I am not sure what this does yet)
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



pd.set_option('display.max_colwidth', 0)

data_dir = os.path.join("C:\\Users\\sami.sharma\\source\\repos\\OpenAi\\data\\")
pdf_files = sorted([x for x in os.listdir(data_dir) if 'DS_Store' not in x])
pdf_files
#TO DELETE LATER
print(pdf_files)
print(data_dir)

#Importing from Redis
from redis import Redis
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField,
    NumericField
)
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)

redis_client = get_redis_connection()

# Constants
VECTOR_DIM = 1536  # length of the vectors
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (cosine similarity strategy)

# Create search index

# Define RediSearch fields for each of the columns in the dataset
# Potential to add any metadata (what kind of meta data can i add and in what format)
filename = TextField("filename")
text_chunk = TextField("text_chunk")
file_chunk_index = NumericField("file_chunk_index")

# define RediSearch vector fields to use HNSW index

text_embedding = VectorField(VECTOR_FIELD_NAME,
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC
    }
)
# Add all field objects to a list to be created as an index
fields = [filename,text_chunk,file_chunk_index,text_embedding]

redis_client.ping()

#checks to see if index already exists and creates redisearch index if it does not
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except Exception as e:
    print(e)
    # Create RediSearch Index
    print("Creating Index")
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )

# The transformers.py file contains all of the transforming functions, including ones to chunk, embed and load data
from transformers import handle_file_string


# Initialise tokenizer
# Tiktoken is a Byte-Pair Encoding which compresses texts and is the OpenAI tokenizer for training models
tokenizer = tiktoken.get_encoding("cl100k_base")

# Process each PDF file and prepare for embedding
for pdf_file in pdf_files:
    
    pdf_path = os.path.join(data_dir,pdf_file)
    print(pdf_path)
    
    # Extract the raw text from each PDF using textract
    text = textract.process(pdf_path, method='pdfminer')
    
    # Chunk each document, embed the contents and load to Redis
    handle_file_string((pdf_file,text.decode("utf-8")),tokenizer,redis_client,VECTOR_FIELD_NAME,INDEX_NAME)

    # Check that docs have been inserted
redis_client.ft(INDEX_NAME).info()['num_docs']

from database import get_redis_results

tech_query = 'what are the safety regulations to follow'

result_df = get_redis_results(redis_client, tech_query, index_name = INDEX_NAME)
result_df.head(2)

# Build a prompt to provide the original query, the result and ask to summarise for the user
summary_prompt = '''Summarise this result in a bulleted list to answer the search query a customer has sent.
Search query: SEARCH_QUERY_HERE
Search result: SEARCH_RESULT_HERE
Summary:
'''
summary_prepped = summary_prompt.replace('SEARCH_QUERY_HERE',tech_query).replace('SEARCH_RESULT_HERE',result_df['result'][0])
summary = openai.Completion.create(engine=COMPLETIONS_MODEL,prompt=summary_prepped,max_tokens=500)
# Response provided by GPT-3
print(summary['choices'][0]['text'])

# Requires a list of "messages", consisting of a "role" (one of system, user or assistant) and "content"
question = 'How can you help me'


completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": question}
  ]
)
print(f"{completion['choices'][0]['message']['role']}: {completion['choices'][0]['message']['content']}")

from termcolor import colored

# A basic class to create a message as a dict for chat
class Message:
    
    
    def __init__(self,role,content):
        
        self.role = role
        self.content = content
        
    def message(self):
        
        return {"role": self.role,"content": self.content}
        
# assistant class to converse with the bot
class Assistant:
    
    def __init__(self):
        self.conversation_history = []

    def _get_assistant_response(self, prompt):
        
        try:
            completion = openai.ChatCompletion.create(
              model=CHAT_MODEL,
              messages=prompt
            )
            
            response_message = Message(completion['choices'][0]['message']['role'],completion['choices'][0]['message']['content'])
            return response_message.message()
            
        except Exception as e:
            
            return f'Request failed with exception {e}'

    def ask_assistant(self, next_user_prompt, colorize_assistant_replies=True):
        [self.conversation_history.append(x) for x in next_user_prompt]
        assistant_response = self._get_assistant_response(self.conversation_history)
        self.conversation_history.append(assistant_response)
        return assistant_response
            
        
    def pretty_print_conversation_history(self, colorize_assistant_replies=True):
        for entry in self.conversation_history:
            if entry['role'] == 'system':
                pass
            else:
                prefix = entry['role']
                content = entry['content']
                output = colored(prefix +':\n' + content, 'green') if colorize_assistant_replies and entry['role'] == 'assistant' else prefix +':\n' + content
                print(output)

# Initiate Assistant class
conversation = Assistant()

# Create a list to hold messages and insert both a system message to guide behaviour and first user question
messages = []
system_message = Message('system','You are a helpful business assistant who has innovative ideas')
user_message = Message('user','What can you do to help me')
messages.append(system_message.message())
messages.append(user_message.message())
messages

# Get back a response from the Chatbot to question
response_message = conversation.ask_assistant(messages)
print(response_message['content'])

next_question = 'Tell me more about option 2'

# Initiate a fresh messages list and insert next question
messages = []
user_message = Message('user',next_question)
messages.append(user_message.message())
response_message = conversation.ask_assistant(messages)
print(response_message['content'])

# Print out a log of conversation so far

conversation.pretty_print_conversation_history()

# Updated system prompt requiring Question to be extracted from the user
system_prompt = '''
You are a helpful Aeroseal Product knowledge base assistant. You need to capture a Question from each customer.
The Question is their query on using Aeroseal products.
Once you have the Question, say "searching for answers".

Example 1:

User: I'd like to know if the sealing product is safe for pregnant women

Assistant: Certainly. Searching for answers.
'''

# New Assistant class to add a vector database call to its responses
class RetrievalAssistant:
    
    def __init__(self):
        self.conversation_history = []  

    def _get_assistant_response(self, prompt):
        
        try:
            completion = openai.ChatCompletion.create(
              model=CHAT_MODEL,
              messages=prompt,
              temperature=0.1
            )
            
            response_message = Message(completion['choices'][0]['message']['role'],completion['choices'][0]['message']['content'])
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
            
            # insert an extra system prompt here to give fresh context to the Chatbot on how to use the Redis results
            # This may be better to hide (leave in for testing)
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
            
        
    def pretty_print_conversation_history(self, colorize_assistant_replies=True):
        for entry in self.conversation_history:
            if entry['role'] == 'system':
                pass
            else:
                prefix = entry['role']
                content = entry['content']
                output = colored(prefix +':\n' + content, 'green') if colorize_assistant_replies and entry['role'] == 'assistant' else prefix +':\n' + content
                #prefix = entry['role']
                print(output)


conversation = RetrievalAssistant()
messages = []
system_message = Message('system',system_prompt)
user_message = Message('user','How can a competitor be disqualified from competition')
messages.append(system_message.message())
messages.append(user_message.message())
response_message = conversation.ask_assistant(messages)
response_message

messages = []
user_message = Message('user','For 2023 please.')
messages.append(user_message.message())
response_message = conversation.ask_assistant(messages)
#response_message

conversation.pretty_print_conversation_history()