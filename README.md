# PrivateGPT
This is a PrivateGPT built using Llama2 and FAISS DB. The bot is powered by Langchain and Streamlit. The bot runs on a decent CPU machine with a minimum of 16GB of RAM.

## How does PrivateGPT work?
When you upload a document, it will be divided into smaller chunks and stored in a special type of database called a vector index that allows for semantic search and retrieval.

When you ask a question, PrivateGPT will search through the document chunks and find the most relevant ones using the vector index. Then, it will use Llama-2-7B-Chat-GGML (8 bit) to generate a final answer.

## Is my data safe?
Yes, your data is safe. PrivateGPT does not store your documents or questions. All uploaded data is deleted after you close the browser tab.

## Why does it take so long to index my document?
It depends on multiple factore such as the size of the file, number of pages in the file. For example: A 12 page document will be index in few seconds.

## Are the answers 100% accurate?
No, the answers are not 100% accurate. PrivateGPT uses Llama-2-7B-Chat-GGML (8 bit) to generate answers. Llama-2-7B-Chat-GGML (8 bit) is a powerful language model, but it sometimes makes mistakes and is prone to hallucinations. Also, PrivateGPT uses semantic search to find the most relevant chunks and does not see the entire document, which means that it may not be able to find all the relevant information and may not be able to answer all questions (especially summary-type questions or questions that require a lot of context from the document).

But for most use cases, PrivateGPT is very accurate and can answer most questions. Always check with the sources to make sure that the answers are correct.
