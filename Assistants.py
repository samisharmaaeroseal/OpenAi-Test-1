import openai
import os
import requests
import numpy as np
import pandas as pd
from typing import Iterator
import tiktoken
import textract
from numpy import array, average

# The transformers.py file contains all of the transforming functions, including ones to chunk, embed and load data
from Transformers import handle_file_string

from Database import get_redis_connection

key = 'OPENAI_API_KEY'
openai.api_key = os.getenv(key)

# Set default models and chunking size
from Config import COMPLETIONS_MODEL, EMBEDDINGS_MODEL, CHAT_MODEL, TEXT_EMBEDDING_CHUNK_SIZE, VECTOR_FIELD_NAME, INDEX_NAME, PREFIX

# Ignore unclosed SSL socket warnings
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

pd.set_option('display.max_colwidth', 0)

data_dir = os.path.join(os.curdir, 'data')
pdf_files = sorted([x for x in os.listdir(data_dir)])
pdf_files

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

# Checks to see if index already exists and creates redisearch index if it does not
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


# Initialise tokenizer
# Tiktoken is a Byte-Pair Encoding which compresses texts and is the OpenAI tokenizer for training models
tokenizer = tiktoken.get_encoding("cl100k_base")

# Process each PDF file and prepare for embedding
for pdf_file in pdf_files:
    
    pdf_path = os.path.join(data_dir,pdf_file)
    print(pdf_path)
    
    # Extract the raw text from each PDF using textract
    text = textract.process(pdf_path, method = 'pdfminer')
    
    # Chunk each document, embed the contents and load to Redis
    handle_file_string((pdf_file,text.decode("utf-8")),tokenizer,redis_client,VECTOR_FIELD_NAME,INDEX_NAME)



# Check that docs have been inserted
redis_client.ft(INDEX_NAME).info()['num_docs']

from Database import get_redis_results

from termcolor import colored

# A basic class to create a message as a dict for chat
class Message:
    
    def __init__(self,role,content):
        
        self.role = role
        self.content = content
        
    def message(self):
        
        return {"role": self.role,"content": self.content}
        



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
        
    #next_user_prompt is always a list of Message objects with at least one Message
    #Message objects are of the form role: "" content: ""
    def ask_assistant(self, next_user_prompt):
        [self.conversation_history.append(x) for x in next_user_prompt]
        
        question_extract = openai.Completion.create(model=COMPLETIONS_MODEL,prompt=f"Extract the user's latest question from this conversation : {self.conversation_history}. Extract the question as a sentence ")
        search_result = self._get_search_results(question_extract['choices'][0]['text'])
            
        self.conversation_history.insert(-1,{"role": 'system',"content": f"Answer the user's question using this content: {search_result}. If you cannot answer the question, say 'Sorry, I don't know the answer to this one'"})
        assistant_response = self._get_assistant_response(self.conversation_history)

        print(next_user_prompt)
        print(assistant_response)
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




    #next_user_prompt is always a list of Message objects with at least one Message
    #Message objects are of the form role: "" content: ""
    def ask_assistant(self, next_user_prompt):
        [self.conversation_history.append(x) for x in next_user_prompt]
        assistant_response = self._get_assistant_response(self.conversation_history)
        product_extract = None
        # Answer normally unless the trigger sequence is used "Thank you."
        if 'thank you' in assistant_response.content.lower():
            product_extract = openai.Completion.create(model=COMPLETIONS_MODEL,prompt=f"Extract the user's product from this conversation : {self.conversation_history}. Use this product for all subsequent questions")
        else:
            question_extract = openai.Completion.create(model=COMPLETIONS_MODEL,prompt=f"Extract the user's latest question from this conversation : {self.conversation_history} for the following product : {product_extract}.")
            search_result = self._get_search_results(question_extract['choices'][0]['text'])
            
            # We insert an extra system prompt here to give fresh context to the Chatbot on how to use the Redis results
            # In this instance we add it to the conversation history, but in production it may be better to hide
            initial_response = Message({"role": 'system',"content": f"Answer the user's question using this content: {search_result}. If you cannot answer the question, say 'to clarify, ', summarize the relevant information from {self.conversation_history} and generate a stand alone question"})
            self.conversation_history.insert(-1, initial_response)
            if 'to clarify' in initial_response.content.lower():
                assistant_response = self.ask_assistant(self.conversation_history)
            
            assistant_response = self._get_assistant_response(self.conversation_history)
            [self.conversation_history.append(x) for x in assistant_response]

            print(next_user_prompt)
            print(assistant_response)
            self.conversation_history.append(assistant_response)
            return assistant_response
