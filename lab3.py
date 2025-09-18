import streamlit as st
from openai import OpenAI
import os
import tiktoken
import re

def run():
    """Lab 3A – Chatbot with OpenAI"""
    # Page setup
    st.set_page_config(page_title="Lab 3 – Chatbot", page_icon="🤖", layout="wide")
    st.title("Lab 3 – OpenAI Chatbot 🤖")
    st.write("Chat with an AI assistant. The conversation is remembered in your session.")

    # Fixed: Use the correct key name from your secrets.toml
    try:
        API = st.secrets["OPENAI_API_KEY"]  # Changed from "OPENAPI_KEY"
    except Exception:
        API = os.getenv("OPENAI_API_KEY")  # Also update environment fallback

    if not API:
        st.error("🔑 No API key found. Please set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment.")
        return

    client = OpenAI(api_key=API)

    # Sidebar
    st.sidebar.header("Options")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"
    st.sidebar.write(f"**Current model:** {model_name}")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=st.session_state["messages"],
                        stream=True,
                    )
                    response = st.write_stream(stream)
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    # Remove the user message if API call failed
                    st.session_state["messages"].pop()

def manage_conversation_buffer(messages, max_pairs=2):
    """
    Keep only the last N message pairs (user + assistant) plus the system message.
    
    Args:
        messages: List of conversation messages
        max_pairs: Maximum number of user-assistant pairs to keep (default: 2)
    
    Returns:
        Trimmed messages list
    """
    if not messages:
        return messages
    
    # Always keep the system message (first message)
    system_message = messages[0] if messages[0]["role"] == "system" else None
    conversation_messages = messages[1:] if system_message else messages
    
    # Count complete pairs (user + assistant)
    pairs = []
    current_pair = []
    
    for msg in conversation_messages:
        current_pair.append(msg)
        
        # If we have a complete pair (user + assistant), save it
        if len(current_pair) == 2 and current_pair[0]["role"] == "user" and current_pair[1]["role"] == "assistant":
            pairs.append(current_pair)
            current_pair = []
        # If we have just a user message without response, keep it for the current conversation
        elif len(current_pair) == 1 and current_pair[0]["role"] == "user":
            # This is the current message being processed, keep it
            continue
        else:
            # Reset if we get an unexpected pattern
            current_pair = [msg] if msg["role"] == "user" else []
    
    # Keep only the last max_pairs complete pairs
    kept_pairs = pairs[-max_pairs:] if len(pairs) > max_pairs else pairs
    
    # Flatten the pairs back into a message list
    kept_messages = []
    for pair in kept_pairs:
        kept_messages.extend(pair)
    
    # Add any incomplete current pair (user message without response yet)
    if current_pair:
        kept_messages.extend(current_pair)
    
    # Reconstruct final message list
    final_messages = []
    if system_message:
        final_messages.append(system_message)
    final_messages.extend(kept_messages)
    
    return final_messages

def run():
    """Lab 3B – Streaming Chatbot with Conversation Buffer"""
    # Page setup
    st.set_page_config(page_title="Lab 3B – Conversation Buffer", page_icon="🤖", layout="wide")
    st.title("Lab 3B – Streaming Chatbot with Conversation Buffer 🤖")
    st.write("Chat with an AI assistant. Only the last 2 message pairs are kept in memory.")
    
    # API key handling
    try:
        API = st.secrets["OPENAI_API_KEY"]
    except Exception:
        API = os.getenv("OPENAI_API_KEY")

    if not API:
        st.error("🔑 No API key found. Please set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment.")
        return

    client = OpenAI(api_key=API)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"
    st.sidebar.write(f"**Current model:** {model_name}")
    
    # Conversation buffer settings
    st.sidebar.subheader("📚 Conversation Buffer")
    max_pairs = st.sidebar.number_input(
        "Max message pairs to keep", 
        min_value=1, 
        max_value=10, 
        value=2, 
        help="Number of user-assistant message pairs to keep in memory"
    )
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # Apply conversation buffer management
    st.session_state["messages"] = manage_conversation_buffer(
        st.session_state["messages"], 
        max_pairs=max_pairs
    )

    # Display conversation statistics
    user_messages = len([msg for msg in st.session_state["messages"] if msg["role"] == "user"])
    assistant_messages = len([msg for msg in st.session_state["messages"] if msg["role"] == "assistant"])
    total_messages = len(st.session_state["messages"]) - 1  # Exclude system message
    
    st.sidebar.write(f"**Total messages in buffer:** {total_messages}")
    st.sidebar.write(f"**User messages:** {user_messages}")
    st.sidebar.write(f"**Assistant messages:** {assistant_messages}")
    
    # Show buffer status
    if user_messages > max_pairs:
        st.sidebar.info(f"🔄 Buffer active: Keeping last {max_pairs} message pairs")
    else:
        st.sidebar.success("✅ All messages fit in buffer")

    # Display chat history (skip system message)
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Apply buffer management before API call
                    buffered_messages = manage_conversation_buffer(
                        st.session_state["messages"], 
                        max_pairs=max_pairs
                    )
                    
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=buffered_messages,
                        stream=True,
                        temperature=0.7,
                    )
                    response = st.write_stream(stream)
                    
                    # Add assistant response
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    
                    # Apply buffer management after adding response
                    st.session_state["messages"] = manage_conversation_buffer(
                        st.session_state["messages"], 
                        max_pairs=max_pairs
                    )
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    # Remove the user message if API call failed
                    st.session_state["messages"].pop()

    # Display current buffer content in expander (for debugging)
    with st.expander("🔍 Debug: Current Buffer Contents"):
        st.write("**Messages being sent to OpenAI:**")
        for i, msg in enumerate(st.session_state["messages"]):
            role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖"}
            st.write(f"{i+1}. {role_emoji.get(msg['role'], '❓')} **{msg['role'].title()}:** {msg['content'][:100]}...")


def count_tokens(text, model="gpt-4o-mini"):
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: The text to count tokens for
        model: The model name to get the appropriate encoding
    
    Returns:
        Number of tokens
    """
    try:
        # Get the encoding for the model
        if "gpt-4o" in model:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        elif "gpt-4" in model:
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            # Default to cl100k_base encoding used by GPT-4 models
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate of 1 token = 4 characters
        return len(text) // 4

def calculate_messages_tokens(messages, model="gpt-4o-mini"):
    """
    Calculate total tokens for a list of messages.
    Includes overhead for message formatting.
    
    Args:
        messages: List of message dictionaries
        model: Model name for token counting
    
    Returns:
        Total token count
    """
    total_tokens = 0
    
    for message in messages:
        # Count tokens in message content
        content_tokens = count_tokens(message["content"], model)
        
        # Add overhead for message structure (role, formatting, etc.)
        # OpenAI adds ~3-4 tokens per message for formatting
        message_overhead = 4
        
        total_tokens += content_tokens + message_overhead
    
    # Add overhead for the completion (assistant response)
    completion_overhead = 3
    total_tokens += completion_overhead
    
    return total_tokens

def manage_token_buffer(messages, max_tokens, model="gpt-4o-mini"):
    """
    Keep messages within token limit by removing oldest messages first.
    Always preserves the system message.
    
    Args:
        messages: List of conversation messages
        max_tokens: Maximum tokens allowed
        model: Model name for token counting
    
    Returns:
        Tuple of (trimmed_messages, total_tokens, removed_count)
    """
    if not messages:
        return messages, 0, 0
    
    # Always keep the system message
    system_message = messages[0] if messages[0]["role"] == "system" else None
    conversation_messages = messages[1:] if system_message else messages
    
    # Start with system message tokens
    system_tokens = count_tokens(system_message["content"], model) + 4 if system_message else 0
    
    # Work backwards through conversation messages
    selected_messages = []
    current_tokens = system_tokens + 3  # +3 for completion overhead
    removed_count = 0
    
    # Add messages from newest to oldest until we hit the token limit
    for message in reversed(conversation_messages):
        message_tokens = count_tokens(message["content"], model) + 4  # +4 for message overhead
        
        if current_tokens + message_tokens <= max_tokens:
            selected_messages.insert(0, message)  # Insert at beginning to maintain order
            current_tokens += message_tokens
        else:
            removed_count += 1
    
    # Reconstruct final message list
    final_messages = []
    if system_message:
        final_messages.append(system_message)
    final_messages.extend(selected_messages)
    
    return final_messages, current_tokens, removed_count

def run():
    """Lab 3B – Token-Based Conversation Buffer"""
    # Page setup
    st.set_page_config(page_title="Lab 3B – Token Buffer", page_icon="🤖", layout="wide")
    st.title("Lab 3B – Token-Based Conversation Buffer 🤖")
    st.write("Chat with an AI assistant. Messages are buffered to stay within token limits.")
    
    # API key handling
    try:
        API = st.secrets["OPENAI_API_KEY"]
    except Exception:
        API = os.getenv("OPENAI_API_KEY")

    if not API:
        st.error("🔑 No API key found. Please set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment.")
        return

    client = OpenAI(api_key=API)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"
    st.sidebar.write(f"**Current model:** {model_name}")
    
    # Token buffer settings
    st.sidebar.subheader("🎯 Token Buffer")
    max_tokens = st.sidebar.number_input(
        "Max tokens for buffer", 
        min_value=100, 
        max_value=8000, 
        value=1000, 
        step=100,
        help="Maximum tokens to send to the LLM (includes all messages)"
    )
    
    # Temperature control
    temperature = st.sidebar.slider(
        "Temperature", 
        0.0, 2.0, 0.7, 0.1,
        help="Controls randomness of responses"
    )
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state["token_history"] = []
        st.rerun()

    # Initialize chat history and token tracking
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    
    if "token_history" not in st.session_state:
        st.session_state["token_history"] = []

    # Apply token buffer management
    buffered_messages, current_tokens, removed_count = manage_token_buffer(
        st.session_state["messages"], 
        max_tokens, 
        model_name
    )

    # Display token statistics
    st.sidebar.subheader("📊 Token Statistics")
    st.sidebar.write(f"**Current tokens:** {current_tokens}")
    st.sidebar.write(f"**Token limit:** {max_tokens}")
    
    # Token usage bar
    token_percentage = min(current_tokens / max_tokens * 100, 100)
    st.sidebar.progress(token_percentage / 100)
    st.sidebar.write(f"**Usage:** {token_percentage:.1f}%")
    
    if removed_count > 0:
        st.sidebar.warning(f"🔄 Removed {removed_count} old messages to fit token limit")
    else:
        st.sidebar.success("✅ All messages fit within token limit")
    
    # Message count statistics
    total_messages = len(st.session_state["messages"]) - 1  # Exclude system message
    buffered_msg_count = len(buffered_messages) - 1  # Exclude system message
    st.sidebar.write(f"**Total messages:** {total_messages}")
    st.sidebar.write(f"**Buffered messages:** {buffered_msg_count}")

    # Display chat history (use buffered messages for display consistency)
    for msg in buffered_messages:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Apply token buffer management before API call
                    buffered_messages, request_tokens, removed_count = manage_token_buffer(
                        st.session_state["messages"], 
                        max_tokens, 
                        model_name
                    )
                    
                    # Store token count for this request
                    st.session_state["token_history"].append({
                        "request_tokens": request_tokens,
                        "removed_messages": removed_count
                    })
                    
                    # Show token info in sidebar during request
                    st.sidebar.info(f"🚀 Sending {request_tokens} tokens to LLM")
                    
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=buffered_messages,
                        stream=True,
                        temperature=temperature,
                    )
                    response = st.write_stream(stream)
                    
                    # Add assistant response
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    
                    # Calculate response tokens
                    response_tokens = count_tokens(response, model_name)
                    st.session_state["token_history"][-1]["response_tokens"] = response_tokens
                    st.session_state["token_history"][-1]["total_tokens"] = request_tokens + response_tokens
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    # Remove the user message if API call failed
                    st.session_state["messages"].pop()
                    if st.session_state["token_history"]:
                        st.session_state["token_history"].pop()

    # Display token usage history
    if st.session_state.get("token_history"):
        with st.expander("📈 Token Usage History"):
            st.write("**Recent API Calls:**")
            for i, call in enumerate(reversed(st.session_state["token_history"][-5:]), 1):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Call", f"#{len(st.session_state['token_history']) - i + 1}")
                with col2:
                    st.metric("Request", f"{call['request_tokens']} tokens")
                with col3:
                    response_tokens = call.get('response_tokens', 0)
                    st.metric("Response", f"{response_tokens} tokens")
                with col4:
                    total_tokens = call.get('total_tokens', call['request_tokens'])
                    st.metric("Total", f"{total_tokens} tokens")
                
                if call['removed_messages'] > 0:
                    st.caption(f"🔄 Removed {call['removed_messages']} old messages")
                st.divider()

    # Display current buffer content in expander (for debugging)
    with st.expander("🔍 Debug: Current Buffer Contents"):
        buffered_messages, debug_tokens, debug_removed = manage_token_buffer(
            st.session_state["messages"], 
            max_tokens, 
            model_name
        )
        
        st.write(f"**Messages being sent to OpenAI (Total: {debug_tokens} tokens):**")
        for i, msg in enumerate(buffered_messages):
            role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖"}
            msg_tokens = count_tokens(msg["content"], model_name)
            st.write(f"{i+1}. {role_emoji.get(msg['role'], '❓')} **{msg['role'].title()}** ({msg_tokens} tokens): {msg['content'][:100]}...")



def create_system_prompt():
    """Create the system prompt for the interactive chatbot."""
    return """You are a helpful assistant designed to interact with users in a specific way:

1. ANSWER QUESTIONS: When a user asks a question, provide a clear, helpful answer that a 10-year-old can understand. Use simple language, examples, and analogies when appropriate.

2. FOLLOW-UP PATTERN: After answering any question, you MUST ask "DO YOU WANT MORE INFO?" (exactly this phrase).

3. HANDLE RESPONSES:
   - If user says "yes" (or variations like "yeah", "sure", "ok"), provide additional detailed information about the topic, then ask "DO YOU WANT MORE INFO?" again.
   - If user says "no" (or variations like "nope", "not really"), ask "What question can I help you with?" to start a new topic.

4. KEEP IT SIMPLE: All explanations should be understandable by a 10-year-old. Use:
   - Simple words and short sentences
   - Real-world examples and comparisons
   - Avoid complex technical jargon
   - Make it engaging and fun when possible

5. STAY ON TOPIC: When providing "more info", elaborate on the same topic with additional details, examples, or related concepts.

Remember: Always follow the pattern of Answer → "DO YOU WANT MORE INFO?" → More details (if yes) → "DO YOU WANT MORE INFO?" again, or New topic (if no)."""

def detect_yes_no_response(message):
    """
    Detect if a message is a yes/no response.
    Returns: 'yes', 'no', or None
    """
    message_lower = message.lower().strip()
    
    # Yes variations
    yes_patterns = [
        r'\b(yes|yeah|yep|sure|ok|okay|of course|definitely|absolutely|yup|y)\b',
        r'\b(tell me more|more info|continue|go on|please)\b',
        r'^(y|yes|yeah)$'
    ]
    
    # No variations  
    no_patterns = [
        r'\b(no|nope|not really|no thanks|nah|stop|enough|that\'s enough)\b',
        r'^(n|no|nope)$'
    ]
    
    for pattern in yes_patterns:
        if re.search(pattern, message_lower):
            return 'yes'
    
    for pattern in no_patterns:
        if re.search(pattern, message_lower):
            return 'no'
    
    return None

def run():
    """Lab 3C – Interactive Chatbot with Follow-up Questions"""
    # Page setup
    st.set_page_config(page_title="Lab 3C – Interactive Bot", page_icon="🤖", layout="wide")
    st.title("Lab 3C – Interactive Chatbot 🤖")
    st.write("Ask me any question! I'll explain it simply and ask if you want to know more.")
    
    # API key handling
    try:
        API = st.secrets["OPENAI_API_KEY"]
    except Exception:
        API = os.getenv("OPENAI_API_KEY")

    if not API:
        st.error("🔑 No API key found. Please set OPENAI_API_KEY in `.streamlit/secrets.toml` or environment.")
        return

    client = OpenAI(api_key=API)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")
    model_name = "gpt-4o" if use_advanced else "gpt-4o-mini"
    st.sidebar.write(f"**Current model:** {model_name}")
    
    # Conversation flow settings
    st.sidebar.subheader("🔄 Conversation Flow")
    st.sidebar.info("""
    **Bot Behavior:**
    1. Answers your question (for 10-year-olds)
    2. Asks "DO YOU WANT MORE INFO?"
    3. If Yes → More details + ask again
    4. If No → "What question can I help you with?"
    """)
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state["messages"] = [
            {"role": "system", "content": create_system_prompt()}
        ]
        st.session_state["conversation_state"] = "waiting_for_question"
        st.session_state["current_topic"] = None
        st.rerun()

    # Initialize chat history and state tracking
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": create_system_prompt()}
        ]
    
    if "conversation_state" not in st.session_state:
        st.session_state["conversation_state"] = "waiting_for_question"  # or "waiting_for_yes_no"
    
    if "current_topic" not in st.session_state:
        st.session_state["current_topic"] = None

    # Display conversation state in sidebar
    st.sidebar.subheader("📊 Conversation State")
    state_emoji = {"waiting_for_question": "❓", "waiting_for_yes_no": "🤔"}
    state_text = {
        "waiting_for_question": "Waiting for a question",
        "waiting_for_yes_no": "Waiting for yes/no response"
    }
    
    current_state = st.session_state["conversation_state"]
    st.sidebar.write(f"{state_emoji.get(current_state, '🤖')} **State:** {state_text.get(current_state, current_state)}")
    
    if st.session_state["current_topic"]:
        st.sidebar.write(f"📖 **Topic:** {st.session_state['current_topic'][:50]}...")

    # Display chat history (skip system message)
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").markdown(msg["content"])

    # Show initial greeting if no messages yet
    if len(st.session_state["messages"]) == 1:  # Only system message
        with st.chat_message("assistant"):
            st.markdown("Hi! I'm here to answer your questions in a simple way that anyone can understand. What would you like to know about?")

    # User input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        # Analyze the user's input
        yes_no_response = detect_yes_no_response(prompt)
        
        # Update conversation state and current topic
        if st.session_state["conversation_state"] == "waiting_for_question":
            # User asked a new question
            st.session_state["current_topic"] = prompt[:100]  # Store topic
            st.session_state["conversation_state"] = "waiting_for_yes_no"
        elif st.session_state["conversation_state"] == "waiting_for_yes_no":
            if yes_no_response == "no":
                # User said no, reset to waiting for new question
                st.session_state["conversation_state"] = "waiting_for_question"
                st.session_state["current_topic"] = None
            # If yes or unclear, stay in yes_no state for more info

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Add context about conversation state to help the AI
                    context_message = f"""
                    Current conversation state: {st.session_state["conversation_state"]}
                    User's response analysis: {yes_no_response if yes_no_response else "unclear"}
                    Current topic: {st.session_state["current_topic"] if st.session_state["current_topic"] else "none"}
                    
                    Remember to follow the interaction pattern:
                    - Answer questions simply (for 10-year-olds)
                    - Always ask "DO YOU WANT MORE INFO?" after answering
                    - If they say yes, give more details then ask again
                    - If they say no, ask "What question can I help you with?"
                    """
                    
                    # Create a temporary message list with context
                    temp_messages = st.session_state["messages"].copy()
                    temp_messages.append({"role": "system", "content": context_message})
                    
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=temp_messages,
                        stream=True,
                        temperature=0.7,
                        max_tokens=300,  # Keep responses concise
                    )
                    response = st.write_stream(stream)
                    
                    # Add assistant response
                    st.session_state["messages"].append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    # Remove the user message if API call failed
                    st.session_state["messages"].pop()

    # Example questions to help users get started
    st.sidebar.subheader("💡 Example Questions")
    example_questions = [
        "What is baseball?",
        "How do airplanes fly?", 
        "What is the sun made of?",
        "Why is the sky blue?",
        "How do computers work?",
        "What are dinosaurs?"
    ]
    
    for question in example_questions:
        if st.sidebar.button(question, key=f"example_{question}"):
            # Simulate user asking the question
            st.session_state["messages"].append({"role": "user", "content": question})
            st.session_state["current_topic"] = question
            st.session_state["conversation_state"] = "waiting_for_yes_no"
            st.rerun()

    # Display conversation flow diagram
    with st.expander("🔍 How the Conversation Flow Works"):
        st.markdown("""
        **Conversation Pattern:**
        
        1. **User asks question** → Bot answers simply
        2. **Bot asks:** "DO YOU WANT MORE INFO?"
        3. **User says "Yes"** → Bot gives more details → Go to step 2
        4. **User says "No"** → Bot asks: "What question can I help you with?" → Go to step 1
        
        **Example Flow:**
        ```
        User: "What is baseball?"
        Bot: [Simple explanation] "DO YOU WANT MORE INFO?"
        User: "Yes"
        Bot: [More details] "DO YOU WANT MORE INFO?"
        User: "No"
        Bot: "What question can I help you with?"
        ```
        
        The bot is designed to explain everything at a 10-year-old level using simple words and examples!
        """)

if __name__ == "__main__":
    run()