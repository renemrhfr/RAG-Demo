## Retrieval Augmented Generation (RAG)
This Repo should give you a quickstart into RAG: A great way of using company/domain-specific knowledge in a Large Language Model.

### Why use RAG?
Putting new knowledge into a Large Language Model is one of the topics that is extensively researched right now.
It is currently no option to build a Large Language Model from scratch for each use case and even "finetuning" a model is
not feasable for most "non-tech" companies.

With RAG you can use an existing, already great working Large Language Model as a basis and build your services around it.
As there are already Open-Source Models like Llama available, you can run all of this on-premise, without the need to pass your data
to external servers.

LLaMA is a recommended choice due to its open-source nature and the ability to run on-premise, ensuring data privacy.
It is already pre-trained on Natural Language Processing, so it can understand semantics and give meaningful responses.
You can download LLaMA from https://llama.meta.com.

### How does RAG work?
As you may have heard, in Chatbots like ChatGPT, Microsoft Copilot, Bing etc. instructions are passed to the
Large Language Model before you can ask a question. That way it can be assured that the Large Language Model does respond in a certain way and that it
does not give information that would violate the terms of use.

This is done by "injecting" a prompt before the user input, tailoring the responses of the Large Language Model -  also known as "prompt engineering".

Now imagine if the "pre-prompt" would not only contain instructions for the model, but also the information that the Large Language Model needs to answer a question.
This is exactly what we will implement here.

### What do we pass to the Large Language Model?
Of course, it is not possible to just throw the whole company knowledge into the Model at once - Large Language Models 
do have Token Limits.
Token limits refer to the maximum number of words or characters the model can process in one go, typically ranging from hundreds to thousands, depending on the model.

This is why we need to slice our knowledge into coherent chunks of knowledge.
When the User asks a question, we search through our chunks for the most relevant one(s) and pass this as context to the Large Language Model.

### How to use this project
- First, install the required dependencies:
The dependencies are listed in requirements.txt.

- Convert the "knowledge" in a simplified Data-Format. 
To convert your PDF documents into text format, first ensure all relevant PDFs are placed in the documents/pdf folder. 
Then, run convertPdf.py
This script reads each PDF and outputs its text content to the documents/export directory as a .txt file.

- Navigate to documents/export. You will find your Documents as a .txt file. We have to split the knowledge into coherent
chunks. Put \<NEXT\> tokens wherever a new part of the knowledge starts that can stand for itself and also remove any irrelevant information such as
Page Counter, Table of Contents and so on. 
The \<NEXT\> Token is just made up by me in this project and simply serves as a marker to split the text into segments, each of which contains a complete idea or piece of information.
This helps in efficiently retrieving relevant content based on user queries.
The script will first check if all slices are inside the bounds of the Token-Limit and will throw an error if not.
Note: the Token-Limit depends on the Model you choose. Adjust the max_token_length in createEmbeddings.py accordingly.
Note that the Question of the User and the instructions also count as Tokens. So make sure to leave some tokens for this.
You do not need to use Llama for this process.
If you simply want to try out the retrival of the most relevant contexts, feel free to browse https://huggingface.co/models?library=sentence-transformers for a smaller model and filter by your desired language.
Simply change model_name in helperFunctions.py to try a different one.
In this Project i used danielheinz/e5-base-sts-en-de, as I think it works great on legal documents and also works great in german, which I needed for my project.



- Run createEmbeddings.py
This will iterate over all files in documents/export and will put them into a format that can be compared to a User input.
If you want to learn more about this process, look into Tokenization, Text Embeddings and what it means to calculate Cosine Similarity.

- Now you can run evaluateEmbedding: Try asking a Question about your documents in the terminal. 
The sorted_cosine_similarities list will contain the top 5 most relevant chunks from your documents. An example of an instruction
will be printed out. Try to pass this instruction into ChatGPT and see if it responds in a way you would expect.
Feel free to experiment from here!

### How to expand
In this project i show you a simplified Version of storing embeddings - in a bigger, scalable Scenario we would not store 
the data in textfiles, but rather use a vector-store. Also the embeddings could be created as batches instead of one by one.
Vector databases can speed up retrieval of relevant information by efficiently searching through embeddings, and batch processing can enhance the speed and efficiency of creating embeddings.

To further explore the concepts used in this project, consider the following resources:
For a deeper understanding of vector databases, which can enhance scalability, see: https://en.wikipedia.org/wiki/Vector_database
Learn more about text embeddings and their role in machine learning with TensorFlow tutorials: https://www.tensorflow.org/tutorials/text/word_embeddings
Dive into the specifics of cosine similarity and its applications in natural language processing: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

If you want to support full conversations in your chatbot, you would not pass the initial instructions for each User query.
You would rather pass this with the first message and then simply use a conversation history and the next relevant contexts.

### Contributing to this Project
I welcome contributions and feedback on this project! Whether you have suggestions for improvements, found a bug, or want to enhance the functionality, here's how you can contribute:
- Issues: Report bugs or suggest enhancements by opening an issue in the GitHub repository.
- Pull Requests: Want to contribute directly? Fork the repository, make your changes, and submit a pull request with a description of what you've done.