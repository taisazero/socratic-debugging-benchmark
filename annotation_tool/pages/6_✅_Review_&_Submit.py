import streamlit as st

st.set_page_config(
    page_title="Submit",
    # check mark
    page_icon="✅"
)


st.write("# Review & Submit :ballot_box_with_check:")
st.write(f"""
## Review Your Dialogue :eyes:
Please review your annotations carefully. If you are satisfied with your annotation, you can submit it. If you would like to make changes, you can directly edit the text file that you downloaded in the previous step. You can also use the annotation tool to make changes to your annotation.

## Common Pitfalls :warning:
1. Misindented Python code.
2. Alternative utterances are paraphrased of the main utterance.
3. The main Socratic utterance provides stronger guidance to the user than its alternatives.

For more information visit the [annotation guidelines](https://docs.google.com/document/d/1xKXtLlzdonR2mV8QdwT2lCLobjvsc8dH2dj4pygLpI0/edit?usp=sharing).

## Check List :ballot_box_with_check:
1. Make sure that you have downloaded all text files using the annotation tool. The files should be named as follows:
    - `*_socratic_dialogue.txt`: The main conversation between the student and the instructor.
    - `*_conversational_thread_1.txt`: The first conversational thread from the main conversation.
    - `*_conversational_thread_2.txt`: The second conversational thread from the main conversation.
    - `*_conversational_thread_3.txt`: The third conversational thread from the main conversation.
2. Make sure that you have reviewed the text files carefully and edit the text files directly.
    - Avoid the common pitfalls listed on page 11 in the [annotation guidelines](https://docs.google.com/document/d/1xKXtLlzdonR2mV8QdwT2lCLobjvsc8dH2dj4pygLpI0/edit?usp=sharing)
    - Make sure the dialogue is coherent and the responses are appropriate.
    - Ensure that there are no typos or grammatical errors.
""", unsafe_allow_html=True)

st.write("# Submit :ballot_box_with_check:")
st.write("""
Now that you have downloaded the text file using the annotation tool and reviewed it carefully, you are ready to submit!

## How to submit :question:
 Thank you for your contribution! Please fill out the Google Form below to submit your annotation. Note that each form submission will be reviewed by a member of our team before being added to the dataset.
"""
, unsafe_allow_html=True)
# st.write("Below is information that you need to fill out the form.")
# st.write("**Bug ID**: " + str(st.session_state["bug_id"]))
# st.write("**Problem ID**: " + str(int(st.session_state["problem_id"])))
st.write (
"""
Google Form: [link](https://docs.google.com/forms/d/e/1FAIpQLScXWf3UkhOxnBL2UuVW5IJExoqx-lR5LiUeYiupDDvI7XHGKA/viewform)

<iframe src="https://docs.google.com/forms/d/e/1FAIpQLScXWf3UkhOxnBL2UuVW5IJExoqx-lR5LiUeYiupDDvI7XHGKA/viewform?embedded=true" width="750" height="600" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>

## Contact :telephone_receiver:
If you have any questions or run into any issues, please contact [Erfan Al-Hossami](mailto:ealhossa@uncc.edu)
"""
, unsafe_allow_html=True)

         