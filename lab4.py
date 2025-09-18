import streamlit as st
import os
import PyPDF2
from io import BytesIO
from typing import List, Dict, Optional

# SQLite fix for ChromaDB - must be before chromadb import
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ChromaDB import with availability check
CHROMADB_AVAILABLE = False
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from openai import OpenAI

def read_pdf_content(pdf_file) -> str:
    """Extract text content from a PDF file."""
    try:
        if hasattr(pdf_file, 'read'):
            # File upload object
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        else:
            # File path string
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def create_vector_database():
    """
    Create ChromaDB collection with OpenAI embeddings from PDF files.
    This function creates the vector database only once per session.
    """
    
    if not CHROMADB_AVAILABLE:
        st.error("ChromaDB not available. Please install with: pip install chromadb pysqlite3-binary")
        return None
    
    # Initialize OpenAI client
    try:
        openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in secrets.toml")
            return None
    except:
        st.error("Could not access OpenAI API key")
        return None
    
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize ChromaDB
    try:
        # Use EphemeralClient to avoid persistence issues
        chroma_client = chromadb.EphemeralClient()
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name="Lab4Collection",
            metadata={"description": "PDF documents for Lab4 vector search"}
        )
        
        st.success("ChromaDB collection 'Lab4Collection' created successfully")
        
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return None
    
    return {
        'client': chroma_client,
        'collection': collection,
        'openai_client': client
    }

def process_uploaded_pdfs(vector_db, uploaded_files):
    """Process uploaded PDF files and add them to the vector database."""
    
    if not uploaded_files:
        st.warning("No PDF files uploaded")
        return False
    
    if len(uploaded_files) > 7:
        st.warning("Please upload no more than 7 PDF files")
        uploaded_files = uploaded_files[:7]
    
    collection = vector_db['collection']
    openai_client = vector_db['openai_client']
    
    documents = []
    metadatas = []
    ids = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Processing PDF files..."):
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Extract text from PDF
            pdf_text = read_pdf_content(uploaded_file)
            
            if pdf_text:
                # Prepare document data
                documents.append(pdf_text)
                metadatas.append({
                    'filename': uploaded_file.name,
                    'file_size': uploaded_file.size,
                    'document_type': 'pdf',
                    'document_id': f"doc_{idx}"
                })
                ids.append(f"doc_{idx}_{uploaded_file.name.replace('.pdf', '').replace(' ', '_')}")
                
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        if documents:
            try:
                status_text.text("Creating embeddings and storing in ChromaDB...")
                
                # Generate embeddings using OpenAI
                embeddings = []
                for doc_idx, doc in enumerate(documents):
                    # Truncate document if too long
                    truncated_doc = doc[:30000]
                    
                    response = openai_client.embeddings.create(
                        input=truncated_doc,
                        model="text-embedding-3-small"
                    )
                    embeddings.append(response.data[0].embedding)
                
                # Add to ChromaDB collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    ids=ids
                )
                
                status_text.text("All documents processed and stored!")
                progress_bar.progress(1.0)
                
                st.success(f"Successfully processed {len(documents)} PDF files")
                return True
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                return False
        
        else:
            st.warning("No valid text content extracted from PDF files")
            return False

def search_vector_database(vector_db, query: str, n_results: int = 3):
    """Search the vector database and return top results."""
    
    collection = vector_db['collection']
    openai_client = vector_db['openai_client']
    
    try:
        # Create embedding for query
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Search the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
        
    except Exception as e:
        st.error(f"Error searching vector database: {str(e)}")
        return None

def test_vector_database(vector_db):
    """Test the vector database with predefined queries and validate results."""
    st.subheader("Vector Database Validation Testing")
    
    test_queries = ["Generative AI", "Text Mining", "Data Science Overview"]
    
    for query in test_queries:
        st.write(f"**Testing query: '{query}'**")
        
        try:
            results = search_vector_database(vector_db, query, n_results=3)
            
            if results and results['documents']:
                st.write("**Top 3 documents (ordered by relevance):**")
                for i, metadata in enumerate(results['metadatas'][0]):
                    filename = metadata.get('filename', 'Unknown Document')
                    distance = results['distances'][0][i]
                    similarity_score = 1 - distance
                    st.write(f"{i+1}. {filename} (Score: {similarity_score:.3f})")
            else:
                st.write("No results found")
        except Exception as e:
            st.error(f"Error testing query '{query}': {str(e)}")
        
        st.write("---")

def create_rag_prompt(user_question: str, context_docs: List[str], source_files: List[str]) -> str:
    """Create a RAG-enhanced prompt with context from retrieved documents."""
    
    context_text = "\n\n".join([f"Document {i+1}:\n{doc[:2000]}" for i, doc in enumerate(context_docs)])
    source_list = ", ".join(source_files)
    
    prompt = f"""You are a knowledgeable assistant that helps answer questions about course materials and academic content. 

You have been provided with relevant context from the following documents: {source_list}

CONTEXT FROM DOCUMENTS:
{context_text}

USER QUESTION: {user_question}

Please provide a comprehensive answer based on the provided context. Follow these guidelines:
1. Use the context from the documents to inform your answer
2. If the answer comes from the provided context, clearly indicate this by saying "Based on the course materials provided..."
3. If you need to supplement with general knowledge, clearly distinguish this by saying "Additionally, from general knowledge..."
4. If the context doesn't contain relevant information, say "The provided course materials don't contain specific information about this topic, but I can provide general information..."
5. Be specific and cite which type of document or course the information comes from when possible

Answer:"""
    
    return prompt

def run():
    """Lab 4A & 4B - Vector Database Setup, Testing, and RAG Chatbot"""
    
    st.set_page_config(page_title="Lab 4 - RAG Chatbot", page_icon="🔍", layout="wide")
    st.title("Lab 4 - Course Information Chatbot with RAG")
    st.write("Vector database with ChromaDB and conversational AI assistant")
    
    # Show ChromaDB availability status
    if CHROMADB_AVAILABLE:
        st.success("ChromaDB is available and ready to use")
    else:
        st.error("ChromaDB is not available. Please install: pip install chromadb pysqlite3-binary")
        return
    
    # Initialize vector database (only once per session)
    if 'Lab4_vectorDB' not in st.session_state:
        st.info("Initializing vector database...")
        vector_db = create_vector_database()
        if vector_db:
            st.session_state.Lab4_vectorDB = vector_db
            st.success("Vector database initialized and stored in session state")
        else:
            st.error("Failed to initialize vector database")
            return
    else:
        vector_db = st.session_state.Lab4_vectorDB
    
    # Sidebar for PDF management
    st.sidebar.header("PDF Document Management")
    
    # File uploader for PDFs
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload up to 7 PDF files for the knowledge base"
    )
    
    # Process PDFs button
    if st.sidebar.button("Process PDFs", type="primary"):
        if uploaded_files:
            success = process_uploaded_pdfs(vector_db, uploaded_files)
            if success:
                st.balloons()
        else:
            st.warning("Please upload some PDF files first")
    
    # Clear database button
    if st.sidebar.button("Clear Database"):
        if 'Lab4_vectorDB' in st.session_state:
            try:
                collection = st.session_state.Lab4_vectorDB['collection']
                all_docs = collection.get()
                if all_docs['ids']:
                    collection.delete(ids=all_docs['ids'])
                st.success("Vector database cleared")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
    
    # Display collection status
    try:
        collection = vector_db['collection']
        count = collection.count()
        st.sidebar.metric("Documents in Database", count)
        
        if count > 0:
            # Show document list
            sample_results = collection.peek(limit=min(5, count))
            if sample_results['metadatas']:
                st.sidebar.write("**Loaded Documents:**")
                for metadata in sample_results['metadatas']:
                    filename = metadata.get('filename', 'Unknown')
                    st.sidebar.write(f"• {filename}")
    except Exception as e:
        st.sidebar.error(f"Error accessing collection: {str(e)}")
    
    # Show validation testing if documents are loaded but haven't been tested
    collection = vector_db['collection']
    if collection.count() > 0:
        if st.sidebar.button("Run Validation Tests"):
            test_vector_database(vector_db)
    
    # Main chatbot interface
    st.header("Course Information Assistant")
    
    if collection.count() == 0:
        st.warning("Please upload and process some PDF documents first to enable the chatbot.")
        return
    
    st.write("Ask me questions about your course materials. I'll search through the documents and provide informed answers.")
    
    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"Sources: {message['sources']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me about your course materials..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching course materials and generating response..."):
                try:
                    # Search for relevant documents
                    results = search_vector_database(vector_db, prompt, n_results=3)
                    
                    if results and results['documents']:
                        # Extract relevant context
                        context_docs = results['documents'][0]
                        source_files = [meta['filename'] for meta in results['metadatas'][0]]
                        
                        # Create RAG prompt
                        rag_prompt = create_rag_prompt(prompt, context_docs, source_files)
                        
                        # Get response from LLM
                        client = vector_db['openai_client']
                        stream = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": rag_prompt}],
                            stream=True,
                            temperature=0.7
                        )
                        
                        # Display streaming response
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                response_placeholder.markdown(full_response + "▌")
                        
                        response_placeholder.markdown(full_response)
                        
                        # Show sources
                        sources_text = ", ".join(source_files)
                        st.caption(f"📚 Sources: {sources_text}")
                        
                        # Add assistant response to chat history
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": full_response,
                            "sources": sources_text
                        })
                        
                    else:
                        # No relevant documents found
                        fallback_response = "I couldn't find relevant information in your course materials for this question. Could you try rephrasing your question or asking about topics covered in your uploaded documents?"
                        st.markdown(fallback_response)
                        
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": fallback_response
                        })
                
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()
    
    # Example questions
    st.subheader("Example Questions")
    st.write("Try asking questions like:")
    example_questions = [
        "What are the learning objectives for the deep learning course?",
        "What programming languages are covered in these courses?",
        "What are the main topics in artificial intelligence applications?",
        "What assignments or projects are mentioned?",
        "What are the prerequisites for these courses?"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{hash(question)}"):
            # Add to chat input
            st.session_state.chat_messages.append({"role": "user", "content": question})
            st.rerun()

if __name__ == "__main__":
    run()