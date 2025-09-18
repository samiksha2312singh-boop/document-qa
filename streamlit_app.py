import streamlit as st
from openai import OpenAI

# Show title and description.
#st.title("Lab 1 Samiksha Singh")
#st.write(
#    "Upload a document below and ask a question about it – GPT will answer! "
#    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
#openai_api_key = st.text_input("OpenAI API Key", type="password")
#if not openai_api_key:
#    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
#else:

    # Create an OpenAI client.
#    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
#    uploaded_file = st.file_uploader(
#        "Upload a document (.txt or .md)", type=("txt", "md")
#    )

    # Ask the user for a question via `st.text_area`.
#    question = st.text_area(
#        "Now ask a question about the document!",
#        placeholder="Can you give me a short summary?",
#        disabled=not uploaded_file,
#    )

#    if uploaded_file and question:

        # Process the uploaded file and question.
#        document = uploaded_file.read().decode()
#        messages = [
#            {
#                "role": "user",
#                "content": f"Here's a document: {document} \n\n---\n\n {question}",
#            }
#        ]

        # Generate an answer using the OpenAI API.
#        stream = client.chat.completions.create(
#            model="gpt-4.1",
#            messages=messages,
#            stream=True,
#        )

        # Stream the response to the app using `st.write_stream`.
#        st.write_stream(stream)

# Configure the page
#st.set_page_config(
#    page_title="Multi-page Labs App",
#    page_icon="📚",
#    layout="wide"
#)

# Create navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Lab:", ["Lab4","Lab 3","Lab 2 (Default)", "Lab 1"])

# Import and run the selected page
if page == "Lab 1":
    import lab1
    lab1.run()
elif page == "Lab 2 (Default)":
    import lab2
    lab2.run()

elif page == "Lab 3":
    import lab3
    lab3.run()

elif page == "Lab 4":
    import lab4
    lab4.run()