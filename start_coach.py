# app.py
import streamlit as st
import asyncio
from neet_main_agent import NEETCoachAgent, QueryType # Make sure your agent class is importable

# --- Page Configuration ---
st.set_page_config(
    page_title="AI NEET Coach",
    page_icon="🤖",
    layout="wide"
)

st.title("🎓 AI NEET Coach")
st.caption("Your personalized AI tutor for NEET preparation, powered by Bio AI principles.")

# --- Initialization ---
# Use Streamlit's session state to keep the agent object persistent across reruns
if 'agent' not in st.session_state:
    with st.spinner("Warming up the AI Coach... Please wait."):
        # This assumes your agent can be initialized without heavy, slow processes.
        # If initialization is slow, you might need to manage it differently.
        st.session_state.agent = NEETCoachAgent()
    st.success("AI Coach is ready!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Helper function to run async code in Streamlit ---
def run_async(coro):
    """A helper to run asynchronous functions within Streamlit's synchronous environment."""
    return asyncio.run(coro)

# --- UI Rendering ---
# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Handling ---
prompt = st.chat_input("Ask me anything about Physics, Chemistry, or Biology...")

if prompt:
    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call your agent's async method using the helper function
            response_obj = run_async(st.session_state.agent.process_query(prompt))
            
            # Display the main content
            response_content = response_obj.content
            st.markdown(response_content)

            # Display sources if available
            if response_obj.sources:
                sources_str = ", ".join(list(set(response_obj.sources)))
                st.info(f"**Sources:** {sources_str}", icon="📚")

            # Display follow-up questions
            if response_obj.follow_up_questions:
                st.markdown("**Here are some follow-up questions you could ask:**")
                for q in response_obj.follow_up_questions:
                    st.button(q) # In a real app, you'd make these buttons functional

    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response_content})
