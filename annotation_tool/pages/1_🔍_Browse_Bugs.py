# streamlit_app.py

import streamlit as st
import pandas as pd
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
import sys
import subprocess
from utils import create_ide, nav_page


st.set_page_config(
    page_title="Browse Bugs",
    # magnifying icon
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


if 'ind' not in st.session_state:
    st.session_state.ind = 0

if 'buggy_code' not in st.session_state:
    st.session_state.buggy_code = ''

if 'problem_id' not in st.session_state:
    st.session_state.problem_id = None

if 'bug_id' not in st.session_state:
    st.session_state.bug_id = None

if 'unit_tests' not in st.session_state:
    st.session_state.unit_tests = None

if 'problem_instruction' not in st.session_state:
    st.session_state.problem_instruction = None

if 'problem_title' not in st.session_state:
    st.session_state.problem_title = None

if 'bug_description' not in st.session_state:
    st.session_state.bug_description = None

if 'bug_fixes' not in st.session_state:
    st.session_state.bug_fixes = []
if 'problem_solutions' not in st.session_state:
    st.session_state.problem_solutions = []

# Load data from local Excel file
bug_repo = "annotation_tool\\bug_repo.xlsx"

# Read the data from Excel sheets into Pandas DataFrames
problems_df = pd.read_excel(bug_repo)

bug_df = pd.read_excel(bug_repo, sheet_name='Bug List')



# Create a slider to select a bug

ind = st.selectbox('Select a bug', [int(b) for b in bug_df['bug_id'].values], st.session_state.ind ) # ('Bug Scroller', 1, len(bug_df), st.session_state.ind)
bug = bug_df[bug_df['bug_id'] == ind].iloc[0]
problem_id = bug['problem_id']
problem = problems_df[problems_df['id'] == problem_id].iloc[0]

st.session_state.problem_title = problem['title']
st.session_state.problem_solutions = [problem['solution_1'], problem['solution_2'], problem['solution_3']]
st.session_state.unit_tests = problem['unit_tests']
st.session_state.bug_description = bug['bug_description']
st.session_state.bug_fixes = [fix for fix in [bug['bug_fix_1'], bug['bug_fix_2'], bug['bug_fix_3']] if pd.isnull(fix) == False]
st.session_state.problem_id = problem_id
st.session_state.bug_id = bug['bug_id']
st.session_state.buggy_code = bug['buggy_code'].replace('```py\n', '').replace('\n```', '')
st.session_state.bug_df = bug_df
st.session_state.problems_df = problems_df
st.session_state.problem_instruction = problem['problem_instruction']

# Display the problem and bug information
st.write(f"# {problem['title']}")
st.write(f"#### Problem ID: {int(problem_id)}")
st.write(f"#### Bug ID: {int(bug['bug_id'])}")
st.write("## Problem Description")
st.write(problem['problem_instruction'])


with st.expander('Problem Data'):
    st.write("## Problem Solution 1")
    create_ide(problem, 'solution_1')

    if pd.isnull(problem['solution_2']) == False:
        st.write("## Problem Solution 2")
        create_ide(problem, 'solution_2')
    if pd.isnull(problem['solution_3']) == False:
        st.write("## Problem Solution 3")

        create_ide(problem, 'solution_3')


    # write unit tests surrounded by ```py`
    if pd.isnull(problem['unit_tests']) == False:
        st.write("## Unit Tests")
        st.write("```py\n" + problem['unit_tests'] + "\n```")
st.write("## Buggy Code")

create_ide(problem, 'buggy_code', buggy_code=bug)
st.write("## Bug Description")
st.write(bug['bug_description'])
st.write("## Bug Fix 1")
st.write(bug['bug_fix_1'])
if pd.isnull(bug['bug_fix_2']) == False:
    st.write("## Bug Fix 2")
    st.write(bug['bug_fix_2'])
if pd.isnull(bug['bug_fix_3']) == False:
    st.write("## Bug Fix 3")
    st.write(bug['bug_fix_3'])



# get the bug_id of the next row in the bug_df
# current row
current_row_index = max(0, bug_df[bug_df['bug_id'] == ind].index[0] - 1)
next_bug = int(current_row_index + 1)
prev_bug = int(current_row_index - 1)
if prev_bug < 1:
    prev_bug = len(bug_df) - 1
if next_bug > len(bug_df):
    next_bug = 0
col1, col2, col3 = st.columns([3,3,1])
with col1:
    if st.button(f"< Go to Previous Bug"):
        st.session_state.ind = prev_bug
        st.experimental_rerun()
with col2:
    if st.button(f"Annotate!"):
        nav_page('Socratic_Dialogue')

        
with col3:
    if st.button(f"Go to Next Bug >"):
        st.session_state.ind = next_bug
        st.experimental_rerun()