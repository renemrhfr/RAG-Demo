import json
import numpy as np
import tensorflow as tf
import os
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

# Use this Prompt for debugging
question = input("Enter your Question: ")
question_embedding = generate_embedding(question)

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

The relevant Link:
{sorted_cosine_similarities[0][0]['url']}

The Question from the User:
{question}

- Begin your answer by introducing yourself.
- Answer the Question in your own words based on the relevant information.
- Include links to the relevant information you used to answer.
"""
print(prompt)
