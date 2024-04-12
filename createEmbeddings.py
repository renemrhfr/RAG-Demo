import json
import os
from helperFunctions import generate_embedding, count_tokens

# Adjust this parameter according to the Token Limit of the Model you use.
max_token_length = 512
# Set this Parameter in your Document to slice it into "chunks" of Data.
paragraph_delimiter = '<NEXT>'

files_to_process = []
for file in os.listdir(os.path.join('documents', 'export')):
    full_path = os.path.join(os.path.join('documents', 'export'), file)
    if os.path.isfile(full_path):
        files_to_process.append(full_path)

for file_entry in files_to_process:
    filename = os.path.basename(file_entry)
    export_filename, _ = os.path.splitext(filename)
    output_file_path = os.path.join('documents', 'embeddings', export_filename + '-embeddings.txt')

    with open(file_entry, 'r', encoding='utf-8') as file:
        content = file.read()
        paragraphs = content.split(paragraph_delimiter)

    processed_paragraphs = []
    for paragraph in paragraphs:
        if not count_tokens(paragraph, max_token_length):
            raise ValueError("Please check the Document " + str(filename) + ", at least one paragraph exceeds the Token limit of " + str(max_token_length))
    for paragraph in paragraphs:
        embedding = generate_embedding(paragraph)
        embedding_as_list = embedding.numpy().tolist()
        processed_paragraphs.append({'url': file_entry, 'content': paragraph, 'embedding': embedding_as_list})

    with open(output_file_path, 'w') as file:
        json.dump(processed_paragraphs, file)