import streamlit as st

# streamlit run chat.py 

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Function to handle user input
def handle_input():
    user_input = st.text_input("You:", "", key="input")
    if user_input:
        st.session_state['history'].append({"role": "user", "text": user_input})
        bot_response = generate_response(user_input)
        st.session_state['history'].append({"role": "bot", "text": bot_response})
        st.experimental_rerun()

# Function to generate bot response (simple echo bot for demonstration)
def generate_response(user_input):
    return f"Bot: I heard you say '{user_input}'"

# Display chat history
st.title("Chat with Bot")
for chat in st.session_state['history']:
    if chat['role'] == 'user':
        st.markdown(f"**You:** {chat['text']}")
    else:
        st.markdown(f"**Bot:** {chat['text']}")

# Input box for user to type their message
handle_input()
