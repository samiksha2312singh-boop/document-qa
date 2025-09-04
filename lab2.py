import streamlit as st
from openai import OpenAI

def run():
    """Main function to run Lab 2"""
    # Show title and description.
    st.title("Lab 2 - Multi-Page App - Samiksha Singh")
    st.write(
        "This is Lab 2, which demonstrates a multi-page Streamlit application. "
        "Use the navigation in the sidebar to switch between Lab 1 and Lab 2."
    )
    
    st.header("About This Lab")
    st.write("""
    **Lab 2A Objectives:**
    - ✅ Create a multi-page Streamlit application
    - ✅ Use sidebar navigation
    - ✅ Separate each lab into its own page
    - ✅ Set Lab 2 as the default page
    - ✅ Maintain functionality from Lab 1
    """)
    
    st.header("Navigation Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Lab 1")
        st.write("""
        - Document QA functionality
        - .txt and .md file support
        - GPT-4.1 model
        - Original Lab 1 features
        """)
    
    with col2:
        st.subheader("🏠 Lab 2 (Current)")
        st.write("""
        - Default landing page
        - Navigation overview
        - Multi-page demonstration
        - Lab information
        """)
    
    st.success("✅ Lab 2A Multi-Page App Implementation Complete!")