## Health & Wellness Assistant Using Llama2

A conversational chatbot that provides personalized health and wellness guidance by combining the open-source Llama2 language model with a retrieval-augmented generation pipeline. It ingests domain-specific documents (PDFs, CSVs, etc.), embeds them with Sentence-Transformers, and stores the vectors in a FAISS index for ultra-fast similarity search.

**Key Features**  
- **Retrieval-Augmented Answers**  
  Pulls in relevant passages from your own PDF/CSV data to ground responses in real information.  
- **Interactive Chat UI**  
  Built with Chainlit for a clean, web-based chat experience—no extra front-end work required.  
- **Easy Data Ingestion**  
  `ingest.py` handles loading and splitting docs, encoding them into embeddings, and populating your vectorstore.  
- **Modular Model Server**  
  `model.py` wires up LangChain’s RAG chain and serves it via Chainlit; swap in any compatible LLM or embedding model.  

**Tech Stack**  
Python • LangChain • Chainlit • Sentence-Transformers • FAISS • PyPDF2 • CSV  

**Getting Started**  
1. `pip install -r requirements.txt`  
2. `python ingest.py`  
3. `chainlit run model.py -w` 
