from app.ui import render, read_pdf, show_sources
from app.config import MISTRAL_API_KEY
from app.mistral_client import MistralClient
from app.rag import chunk_text, build_index, retrieve

import streamlit as st
import uuid
from datetime import datetime

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set")

client = MistralClient(MISTRAL_API_KEY)

@st.cache_resource(show_spinner=False)
def process_document(uploaded_file):
    """Process document once and cache results"""
    with st.spinner("Processing document for the first time..."):
        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)
        embeddings = client.embed(chunks)
        index = build_index(embeddings)
    return chunks, index

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'current_document' not in st.session_state:
    st.session_state.current_document = None

def add_message(role, content, sources=None):
    """Add a message to chat history"""
    st.session_state.chat_history.append({
        'id': str(uuid.uuid4()),
        'role': role,
        'content': content,
        'sources': sources or [],
        'timestamp': datetime.now().strftime("%H:%M")
    })

def clear_chat():
    """Clear chat history but keep document"""
    st.session_state.chat_history = []

def answer_question(question, chunks, index):
    """Generate answer for a question"""
    with st.spinner("🔍 Searching document..."):
        q_embedding = client.embed([question])[0]
        context = retrieve(q_embedding, index, chunks)
    
    with st.spinner("💭 Generating answer..."):
        # Build prompt with conversation history for context
        conversation_context = ""
        if st.session_state.chat_history:
            recent_messages = st.session_state.chat_history[-4:]  # Last 4 messages for context
            for msg in recent_messages:
                if msg['role'] == 'user':
                    conversation_context += f"User: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    conversation_context += f"Assistant: {msg['content']}\n"
        
        prompt = f"""You are a helpful assistant answering questions about a document. 
                    Use ONLY information from the provided context. If the answer is not in the context, say you don't know. Always give answers in paragraph form.

                    Previous conversation context (for reference only):
                    {conversation_context}

                    Current context from document:
                    {chr(10).join(context)}

                    Question: {question}

                    Answer based only on the document context above:"""
                            
        answer = client.chat(prompt)
    
    return answer, context

def main():
    # Get UI components
    uploaded = render()
    
    # Document processing (only once)
    if uploaded and not st.session_state.document_processed:
        with st.container():
            st.subheader("📄 Processing Document")
            
            try:
                # Process document with caching
                chunks, index = process_document(uploaded)
                
                # Store in session state
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.document_processed = True
                st.session_state.current_document = uploaded.name
                
                # Clear any previous chat
                clear_chat()
                
                # Add welcome message
                add_message('assistant', f"I've loaded the document **'{uploaded.name}'**. Ask me anything about it!")
                
                st.success(f"✅ Document loaded successfully!")
                
                # Show document stats
                with st.expander("📊 Document Details", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Sections", len(chunks))
                    with col2:
                        st.metric("Processing Status", "Ready")
                    st.caption(f"*Document processed and ready for questions*")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to process document: {str(e)}")
                return
    
    # Main chat interface
    if st.session_state.document_processed:
        # Sidebar with controls
        with st.sidebar:
            st.header("💬 Chat Controls")
            
            st.subheader(f"📄 {st.session_state.current_document}")
            
            # Chat management
            if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
                clear_chat()
                add_message('assistant', "Chat cleared. How can I help you with the document?")
                st.rerun()
            
            if st.button("🔄 New Document", use_container_width=True):
                for key in ['document_processed', 'chunks', 'index', 'current_document']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.cache_resource.clear()
                st.session_state.chat_history = []
                st.rerun()
            
            st.divider()
            
            # Chat statistics
            st.caption("**Chat Statistics**")
            user_msgs = sum(1 for msg in st.session_state.chat_history if msg['role'] == 'user')
            assistant_msgs = sum(1 for msg in st.session_state.chat_history if msg['role'] == 'assistant')
            st.caption(f"👤 User: {user_msgs} messages")
            st.caption(f"🤖 Assistant: {assistant_msgs} messages")
            
            st.divider()
            
            # Suggested questions
            st.caption("**💡 Try asking:**")
            suggestions = [
                "Summarize the main points",
                "What are the key findings?",
                "Explain the methodology",
                "List the recommendations"
            ]
            for suggestion in suggestions:
                if st.button(suggestion, use_container_width=True, type="secondary"):
                    st.session_state.suggested_question = suggestion
                    st.rerun()
        
        # Main chat area
        st.subheader(f"💬 Chat about: **{st.session_state.current_document}**")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message['role']):
                    # Display message content
                    st.markdown(message['content'])
                    
                    # Show timestamp
                    st.caption(f"*{message['timestamp']}*")
                    
                    # Show sources for assistant messages if available
                    if message['role'] == 'assistant' and message.get('sources'):
                        with st.expander("📚 View sources", expanded=False):
                            show_sources(message['sources'])
        
        # Chat input at bottom
        st.divider()
        
        # Initialize chat input
        chat_input_key = f"chat_input_{len(st.session_state.chat_history)}"
        
        # Handle suggested questions
        if 'suggested_question' in st.session_state:
            question = st.session_state.suggested_question
            del st.session_state.suggested_question
        else:
            # Regular chat input
            question = st.chat_input(
                "Ask a question about the document...",
                key=chat_input_key
            )
        
        # Process question
        if question:
            # Add user message to chat
            add_message('user', question)
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                # Create a placeholder for the streaming effect
                message_placeholder = st.empty()
                
                # Show initial thinking indicator
                message_placeholder.markdown("💭 Thinking...")
                
                # Generate answer
                answer, sources = answer_question(
                    question, 
                    st.session_state.chunks, 
                    st.session_state.index
                )
                
                # Update with actual answer
                message_placeholder.markdown(answer)
                
                # Add timestamp
                st.caption(f"*{datetime.now().strftime('%H:%M')}*")
                
                # Show sources
                with st.expander("📚 View sources", expanded=False):
                    show_sources(sources)
                    st.caption(f"*Retrieved {len(sources)} relevant sections*")
            
            # Add assistant message to history
            add_message('assistant', answer, sources)
            
            # Scroll to bottom (using JavaScript via components)
            st.rerun()
    
    else:
        # Initial state - no document uploaded
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 40px;'>
                <h1>📚 Document Q&A Assistant</h1>
                <p>Upload a PDF document to start chatting with AI about its content</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👆 Use the sidebar to upload a PDF document")

if __name__ == "__main__":
    main()