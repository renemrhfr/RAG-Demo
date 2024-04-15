import json
import sys

import numpy as np
import tensorflow as tf
import os
import requests
from helperFunctions import generate_embedding, cosine_similarity

# Collecting the files where we will load our text-embeddings from
files_to_process = []
for file in os.listdir(os.path.join('documents', 'embeddings')):
    full_path = os.path.join(os.path.join('documents', 'embeddings'), file)
    if os.path.isfile(full_path):
        files_to_process.append(full_path)

# We will iterate over this list to find the most relevant data-chunks
embeddings_list = []
for file_entry in files_to_process:
    with open(file_entry, 'r') as file:
        embeddings_list.extend(json.load(file))

linebreak = """
-------------------------------------------------
"""

while True:
    question = input("Enter Question or type 'BYE' to quit: ")
    if question == 'BYE' or question == 'bye':
        break
    question_embedding = generate_embedding(question)
    print(linebreak)
    # Collecting the 5 most relevant Paragraphs from our texts.
    similarities = []

    for entry in embeddings_list:
        saved = np.array(entry["embedding"], dtype=np.float32)
        saved = tf.convert_to_tensor(saved[None, :], dtype=tf.float32)
        similarity = cosine_similarity(question_embedding, saved)
        similarities.append((entry, similarity))

    sorted_cosine_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    # This prompt can now be passed to a Large Language Model
    # Feel free to experiment with the text and maybe try to add more than one context.
    # See how the Large Language Model responds to different Prompts
    # You would also implement a feature from here, where you do not pass the full instructions if the chat continues
    # and follow-up Questions are asked.
    prompt = f"""
Your name is C3PO. You are a friendly, helpful Assistant.
You will now be given a relevant context and a Question from a User.
    
The relevant Context:
{sorted_cosine_similarities[0][0]['content']}

Source:
{sorted_cosine_similarities[0][0]['url']}
    
The Question from the User:
{question}

- Begin your answer by introducing yourself.
- Answer the Question in your own words based on the relevant information.
- Include the source as link at the end of your answer.
"""
    # Debug Mode - prints Prompt instead of sending to LLM
    if len(sys.argv) > 1 and sys.argv[1] == '-D':
        print(prompt)
    # If you use OLLAMA to run your model locally, you can leave those as-is.
    else:
        # Verbose output: Print the prompt AND send to LLM.
        if len(sys.argv) > 1 and sys.argv[1] == '-V':
            print("This prompt will be passed to the LLM: ")
            print(prompt)
            print(linebreak)
        url = 'http://localhost:11434/api/generate'
        data = {
            'model': 'Mistral',
            'prompt': prompt,
        }
        response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'}, stream=True)
        if response.status_code == 200:
            for chunk in response.iter_content( chunk_size=None):
                json_line = json.loads(chunk.decode('utf-8'))
                if not json_line.get("done", True):
                    sys.stdout.flush()
                    print(json_line.get("response", ""), end='')
        else:
            print("Failed to fetch data from LLM. Status Code:", response.status_code)
        print(linebreak)
