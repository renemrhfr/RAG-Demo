# Retrieval Augmented Generation (RAG)

This Repo should give you a quickstart into Retrieval Augmented Generation (RAG): A great way of using company/domain-specific knowledge in a Large Language Model.

## Why Use RAG?
Having a AI-Chatbot with company specific knowledge is a very common requirement right now. Building a Large Language Model from scratch for each use case is not realistic, and even finetuning models can be challenging, especially for companies without extensive tech resources.

With RAG you can use an existing, already great working Large Language Model as a basis and build your service around it.
As there are already Open-Weights Models like Mistral and Llama available, you can run all of this on-premise, without the need to pass your confidential data
to external servers.

- **LLaMA**: [Visit LLaMA](https://llama.meta.com)
- **Mistral**: [Visit Mistral](https://mistral.ai)

To run models locally with ease, consider using Ollama: [Visit Ollama](https://ollama.com)

## How Does RAG Work?

In applications like ChatGPT and Microsoft Copilot, a "pre-prompt" is sent to the Large Language Model before or with the user input.
This includes instructions about in which way to respond and which topics to avoid to ensure compliance with usage terms. RAG takes this further by integrating relevant information directly into the pre-prompt, allowing the model to access the necessary data to answer queries about any topic we want.

## What do we pass to the Large Language Model?
Of course, it is not possible to just throw the whole knowledge at the Model at once - Large Language Models 
do have Token Limits.
Token limits refer to the maximum number of words or characters the model can process in one go, typically ranging from hundreds to thousands, depending on the model.

This is why we need to slice our knowledge into coherent chunks of knowledge.
When the User asks a question, we search through our chunks for the most relevant one(s) and pass this as context to the Large Language Model.

## Example for a RAG-Prompt
An example for a prompt could look like this:

>Your name is C3PO. You are a friendly, helpful Assistant.
You will now be given a relevant context and a Question from a User.
> 
>The relevant Context:
---> One or more parts of our documents will be printed out right here.
> 
>Source:
---> We will also link the filename from the document that contains the above context.
> 
>The Question from the User:
---> Here we pass the question from the user.
>
> - Begin your answer by introducing yourself.
> - Answer the Question in your own words based on the relevant information.
> - Include the source as link at the end of your answer.

## How do we find the relevant contexts from our knowledge chunks?
We are going to convert the documents into text embeddings, which are numerical vectors where similar texts and paragarphs are positioned close to each other.
For this process i used the model `intfloat/multilingual-e5-large` from https://huggingface.com

Then, when a user asks a question, we convert the question aswell and compare the similarity of the vector to the vectors of your knowledge chunks.
We can then pass the closest one(s) as Context in the prompt.

## Getting Started with This Project

1. Install necessary dependencies listed in `requirements.txt`.
2. Prepare your environment by creating directories: `documents/embeddings`, `documents/export`, `documents/pdf`.
3. Convert PDF documents to text by putting them in `documents/pdf` and running `convertPdf.py`. This script extracts text to `documents/export` as `.txt` files.
4. Prepare Data: In `documents/export`, add `<NEXT>` tokens to segment coherent knowledge chunks and strip extraneous content like page numbers.
5. Create Embeddings: Run `createEmbeddings.py` to process text files and prepare them for retrieval based on user queries. 
6. Evaluate Embeddings: Use `rag.py` to test retrieval by querying your documents. Use flags `-D` to print the prompt instead of sending to the LLM or `-V` for verbose mode, which prints the prompt AND sends it to the LLM.

## How to Expand

This project uses a simplified method of storing embeddings in text files. In a scalable scenario, consider using a vector store and batch processing for creating embeddings. Learn more about these advanced methods:
- **Vector Databases**: [Wikipedia on Vector Databases](https://en.wikipedia.org/wiki/Vector_database)
- **Text Embeddings**: [TensorFlow Tutorials on Word Embeddings](https://www.tensorflow.org/tutorials/text/word_embeddings)
- **Cosine Similarity**: [Scikit-learn on Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

## Contributing

Your contributions and feedback are welcome! Here's how you can contribute:
- **Issues**: Report bugs or suggest enhancements by opening an issue.
- **Pull Requests**: Fork the repository, make your changes, and submit a pull request describing your improvements.