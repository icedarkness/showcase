RAG-Powered Financial Product Chatbot
A self-contained, browser-based financial assistant that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about a fictional bank's products. This project demonstrates modern AI techniques in a practical, user-facing application.


(Note: You can create a GIF of the app in action and replace this link)

✨ Features
Conversational AI: Ask questions in natural language about financial products.

Retrieval-Augmented Generation (RAG): The chatbot's knowledge is strictly limited to a provided set of documents, preventing it from making up information.

Source Transparency: Each answer cites the specific text chunks from the knowledge base that were used to generate the response, building user trust.

Fully Client-Side: The entire application, including the embedding model and RAG pipeline, runs directly in the browser using TensorFlow.js. No server-side dependencies are needed.

Modern Tech Stack: Built with modern JavaScript, Gemini for language generation, and the Universal Sentence Encoder for creating semantic text embeddings.

🏛️ Architecture and Tech Stack
This project is designed as a fully self-contained demonstration that can be run by simply opening a single HTML file.

Knowledge Base: A corpus of text describing fictional financial products is stored directly in the HTML file.

Text Chunking: The knowledge base is programmatically split into smaller, semantically relevant chunks.

Embeddings (Client-Side): When the application loads, Google's Universal Sentence Encoder (via TensorFlow.js) is loaded in the browser. It converts all text chunks into 512-dimension numerical vectors (embeddings).

Vector Search (Retrieval): When a user asks a question, their query is also converted into an embedding. The application then performs a cosine similarity search to find the most relevant text chunks from the knowledge base.

Prompt Augmentation: The user's original question and the retrieved text chunks are combined into a detailed prompt.

Generation (LLM Call): This augmented prompt is sent to the Google Gemini API, which generates a coherent, context-aware answer.

A Note on Production Architecture
The client-side architecture of this demo is intentionally chosen for simplicity and portability, making it easy for anyone to run and inspect the code without any setup.

In a real-world, enterprise environment, this architecture would be insecure. A production-grade system would be implemented as follows:

Secure Back-End: The knowledge base, the embedding model, the vector database (e.g., Pinecone, ChromaDB), and the RAG logic would all reside on a secure back-end server.

API-Driven: The front-end user interface would be a lightweight client that only sends the user's query to a protected API endpoint.

Data Privacy: No proprietary data from the knowledge base would ever be exposed to the client. The API would process the request and return only the final, generated answer to the user.

This project intentionally showcases the core RAG logic in a transparent way, while my understanding of system design accounts for the security and scalability needs of a production environment.

🚀 How to Run Locally
Download the rag_chatbot.html file from this repository.

Open the file in a modern web browser (e.g., Google Chrome, Firefox).

The application will take a moment to initialize as it downloads the sentence encoder model.

Once the welcome message appears, you can start asking questions!