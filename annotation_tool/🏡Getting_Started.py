import streamlit as st
from utils import nav_page

st.set_page_config(
    page_title="Getting Started",
    # house icon
    page_icon="🏡",
    initial_sidebar_state='expanded'
)
c1, c2, c3 = st.columns([1,3,1])
with c2:
    st.image("annotation_tool/img/socratic_debugging.png", use_column_width=True)
    st.caption("Image credit: DALL-E 2 by [OpenAI](https://openai.com) and Stable Diffusion 2.0 by [StabilityAI](https://stability.ai/)")

st.write("# Welcome to the Socratic Debugging Project!")

st.markdown(
    """
    The aim of this project is to develop AI agents that help novice programmers debug their code through Socratic dialogue. Our first goal is to create a dataset of Socratic dialogues that can be used to train and evaluate such AI agents.

    ## What is Socratic Dialogue? 🧔

    [Socratic dialogue](https://www.testgeek.com/blog/the-socratic-method-great-teachers-ask-the-right-questions/) is named after the ancient Greek philosopher Socrates, who is known for using a method of questioning in which an expert guides a novice towards answering a question or solving a problem on their own.
    
    ## Contributing :heart:

    We welcome annotations from people with good Python programming skills who are interested in helping create a dataset of Socratic dialogues for learning to code. In each dialogue, an Instructor helps a Student fix buggy implementations of simple computational problems. If you are interested in contributing, first familiarize yourself with the annotation guidelines that you can find [here](https://docs.google.com/document/d/1xKXtLlzdonR2mV8QdwT2lCLobjvsc8dH2dj4pygLpI0/edit?usp=sharing), then follow the annotation process outlined below.

    ## Annotation Process Overview :clipboard:

    1. Start by browsing bugs in the `Browse Bugs` page. To access that page, you can either navigate using the sidebar to the left or by clicking on the "Next >" button. For each bug you will see the programming exercise, the bug, a bug description, and one or more bug fixes.
    2. Once you find a bug that you would like to annotate, click on the **"Annotate"** button that will take you to the annotation tool.
    3. Annotate a complete Socratic dialogue for that bug. Then write up to 3 conversational threads based on that dialogue. 
    4. When you are done, click on the **"Save & Export Data"** button to save your annotations into local text files. Submit the text files through the `Review & Submit` page that you can find in the sidebar to the left.

    
    ## Contact :mailbox_with_mail:
    If you have any questions or run into any issues, please contact Erfan Al-Hossami at ealhossa@uncc.edu.

    ## Ready? :rocket:
    Click on the "Next >" button to start browsing bugs.
    """
, unsafe_allow_html=True)

c1, c2 = st.columns([4,1])

with c2:
    if st.button("Next >"):
        nav_page("Browse_Bugs")