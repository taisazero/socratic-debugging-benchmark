import streamlit as st
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
from streamlit_chat import message
import sys
import subprocess
import regex as re
import streamlit.components.v1 as components
from copy import deepcopy
from utils import *
import time
import os
import pandas as pd


PAGE_NUMBER = 3
NEXT_PAGE_NAME = 'Review_&_Submit'
PREV_PAGE_NAME = 'Conversational_Thread_2'
PAGE_TITLE = 'Annotation Phase: Conversational Thread 3'
EXPORT_FILE_NAME = '_conversational_thread_3.txt'
NEXT_BUTTON = 'Review & Submit >'
PREV_BUTTON = '< Socratic Thread 2'

def save_page(dialogue_text, show_format, page_num):
    st.session_state.saved_page_content[page_num] = {
                                                   'exported_text': dialogue_text,
                                                   'display_text': show_format,
                                                   'dialogue_history': deepcopy(list(st.session_state.dialogue_history)),
                                                   'code_states': deepcopy(list(st.session_state.code_states)),
                                                   'program_output': st.session_state.program_output,
                                                   'code_content': st.session_state.code_content,    
                                                   'page_memory': deepcopy(dict(st.session_state.page_memory)),
                                                   'student_profile': deepcopy(str(st.session_state.student_profile)),
                                                }
def load_page(page_num):
    if st.session_state.saved_page_content[page_num] != '':
        st.session_state.dialogue_history = st.session_state.saved_page_content[page_num]['dialogue_history']
        st.session_state.code_states = st.session_state.saved_page_content[page_num]['code_states']
        st.session_state.program_output = st.session_state.saved_page_content[page_num]['program_output']
        st.session_state.code_content = st.session_state.saved_page_content[page_num]['code_content']
        st.session_state.displayed_dialogue_text = st.session_state.saved_page_content[page_num]['display_text']
        st.session_state.dialogue_text = st.session_state.saved_page_content[page_num]['exported_text']
        st.session_state.page_memory = st.session_state.saved_page_content[page_num]['page_memory']
        st.session_state.speaker_id = 0 if st.session_state.dialogue_history[-1]['speaker'] == 'Assistant' else 1
        st.session_state.student_profile = st.session_state.saved_page_content[page_num]['student_profile']
        st.session_state.is_exported = True
    else:
        return

st.set_page_config (
                        page_title= "Socratic Debugging Annotation Tool", 
                        layout="wide",
                        # handwriting icon
                        page_icon = '📝',
                        initial_sidebar_state="collapsed",
                    )


def update_chat_interface (msg, speaker):
   if msg != '':
        if speaker == 'alt':
            # get last speaker from dialogue history
            speaker = st.session_state.dialogue_history[-1]['speaker']
            st.session_state.dialogue_history.append({'txt': msg, 'speaker': speaker, "type": 'alt'})
        else:
            st.session_state.dialogue_history.append({'txt': msg, 'speaker': speaker, "type": 'main'})
        

def initialize_session_state():
    if 'imported_flag' not in st.session_state:
        st.session_state.imported_flag = False
        
    if 'dialogue_history' not in st.session_state:
        st.session_state.dialogue_history = []

    if 'saved_page_content' not in st.session_state:
        st.session_state.saved_page_content = [''] * 4

    if 'code_content' not in st.session_state:
        st.session_state.code_content = ''

    if  st.session_state.code_content == '' and 'buggy_code' in st.session_state:
        if st.session_state.saved_page_content[0] != '':
            st.session_state.code_content = st.session_state.saved_page_content[0]['code_content']
        else:
            st.session_state.code_content = st.session_state.buggy_code.replace('```py\n', '').replace('\n```', '')
     
    if 'program_output' not in st.session_state:
        st.session_state.program_output = ''

    if 'code_states' not in st.session_state:
        st.session_state.code_states = []
  
    if 'speaker_id' not in st.session_state:
        st.session_state.speaker_id = 0

    if 'update_chat' not in st.session_state:
        st.session_state.update_chat = False
    
    if 'update_state' not in st.session_state:
        st.session_state.update_state = False
    
    if 'dialogue_text' not in st.session_state:
        st.session_state.dialogue_text = ''

    if 'displayed_dialogue_text' not in st.session_state:
        st.session_state.displayed_dialogue_text = ''

    if 'is_exported' not in st.session_state:
        st.session_state.is_exported = False

    if 'problem_id' not in st.session_state:
        st.session_state.problem_id = 0
        st.session_state.bug_id = 0
        st.session_state.bug_description = ''
        st.session_state.problem_title = ''
        st.session_state.problem_instruction = ''
        st.session_state.bug_fixes = []
        st.session_state.unit_tests = ''
        st.session_state.buggy_code = ''
        st.error('Please select a bug from the Browse Bugs page to begin annotating.')

    if 'page_memory' not in st.session_state:
        st.session_state.page_memory = {
                                        0:[], 1:[], 2:[], 3:[]
                                       }
    if 'load_page_from_exported_memory' not in st.session_state:
        st.session_state.load_page_from_exported_memory = False
    # page_memory is used for storing "displayed_dialogue_text" for each page.
    if st.session_state.load_page_from_exported_memory:
        load_page(PAGE_NUMBER)
        st.session_state.load_page_from_exported_memory = False

    

def text_area_changed():
    st.session_state.update_state = True
    st.session_state.displayed_dialogue_text = st.session_state.text_area
   

def render_bug_info():
    if 'bug_description' not in st.session_state:
        return
    
    with st.expander('Exercise Info', expanded=False):       
        st.write(f"## {st.session_state.problem_title}")
        st.write(f"## Problem Description")
        st.write(st.session_state.problem_instruction)

    with st.expander('Selected Bug Info', expanded=False):
        st.write(f"## Bug Description")
        st.write(st.session_state.bug_description)
        st.write(f"## Bug Fixes")
        for fix in st.session_state.bug_fixes:
            st.write(f"* {fix}")



def render_instructions():
    with st.expander('Instructions', expanded=False):
        st.write(f"## Instructions")    
        st.write("""
                The task in this page is to annotate a dialogue between the student and the instructor, where the student is trying to fix a bug in that you can see in the IDE. The dialogue starts with a student utterance. You can type the utterance in the input box labeled 'Utterance'. Once you complete typing the utterance click on the button labeled '➕ Main' to add the utterance to the dialogue. If you want to add an alternative utterance, you can type the utterance in the input box labeled 'Utterance' and click on the button labeled '➕ Alternative'. The speakers are automatically cycled. When the student changes the code, edit the code accordingly in the IDE window and click on the button labeled '➕ Code'. This will add the code to the dialogue. When you are done with the dialogue, you can click on the button labeled 'Save & Export Data' to export the dialogue to a `.txt` file, then you can click on the button labeled 'Next' to go to the next page. If you have an exported text file you would want to import into the dialogue, you can click on the button labeled 'Import Data' and select the file and then click on 'Complete Import'. The dialogue will be imported and you can continue annotating.
    """)
        st.write(f"## Additional Annotation Rules")
        st.write("""
                    1. Ensure that all utterances sound natural; that they are clear and to the point; that they are factually and grammatically correct.
                    2. Short code snippets or variable names mentioned in the utterances are surrounded by back quotes, e.g. `var_name`.
                    3. Provide as many alternative utterances as possible, that explore a diverse set of behaviors. These alternative Socratic utterances should be distinct semantically, and not mere paraphrases of the main utterance.
                        * It is especially important to be comprehensive when providing alternatives for the Instructor turn, which are expected to cover as much as possible the entire range of  Socratic guidance.
                        * For the Student, alternative utterances may give different or conflicting answers to an Instructor question, reflecting different levels of understanding. Students may give correct or incorrect answers. Students may also introduce new bugs when fixing the original bug.
                    4. The main utterance provided for the Instructor at each turn should be at the highest Socratic level (most hands-off) that the user is expected to be able to use successfully to make progress towards fixing the bug.
                        * For  example, the utterance “How many times does the for loop iterate?” is at a higher Socratic level than the utterance “What purpose do you think `range` serves in the for loop?”
                        * Note that the highest level of Socratic advice for one student may not be the same as the highest level for another student with different abilities. Therefore, before starting each dialogue annotation, it is useful to set a student profile in terms of his coding and cognitive abilities. This will help determine the optimal (highest) Socratic level that this particular student can follow successfully at each turn.
                        * Note that this is only an expectation. To make the dialogues realistic, it is important that sometimes the Student does not make the expected progress using the Instructor advice, as is the case for example in turn 1. In such cases, the Instructor may need to resort to a lower level of Socratic advice at the next turn.
""")

def render_student_profile():
    if 'student_profile' not in st.session_state:
        st.session_state.student_profile = ''
    
    form = st.form(key='student_profile_form', clear_on_submit=True)
    profile = form.text_input('(Optional) Describe the student in a few sentences', value='', key='st_profile', help= "Describe the student and their learning preferences in a few sentences. For example, 'The student likes trial and error and is not very good at reading documentation'. This is optional.")
    if form.form_submit_button ('Submit') and profile != st.session_state.student_profile:
        st.write('## Student Description')
        st.write(profile)
        st.session_state.student_profile = profile
        st.success('Student description submitted successfully')

def import_from_text_file(txt_filename, dialogue_text):
    # get problem id and bug id from the file name:
    # example: 1_2_blah.txt -> problem_id = 1, bug_id = 2
    problem_id, bug_id = txt_filename.split('_')[:2]
    # convert problem_id and bug_id to int
    problem_id = int(problem_id)
    bug_id = int(bug_id)
    # read problem title and bug description from the spreadsheet
    bug = st.session_state.bug_df[st.session_state.bug_df['bug_id'] == bug_id].iloc[0]
    problem = st.session_state.problems_df[st.session_state.problems_df['id'] == problem_id].iloc[0]

    st.session_state.problem_title = problem['title']
    st.session_state.problem_solutions = [problem['solution_1'], problem['solution_2'], problem['solution_3']]
    st.session_state.unit_tests = problem['unit_tests']
    st.session_state.bug_description = bug['bug_description']
    # include bug fixes only if not None
    st.session_state.bug_fixes = [fix for fix in [bug['bug_fix_1'], bug['bug_fix_2'], bug['bug_fix_3']] if pd.isnull(fix) == False]
    st.session_state.problem_id = problem_id
    st.session_state.bug_id = bug['bug_id']
    st.session_state.buggy_code = bug['buggy_code']
    st.session_state.problem_instruction = problem['problem_instruction']
    st.session_state.unit_tests = problem['unit_tests']
    # remove all text before <dialogue> and after </dialogue> including line breaks
    dialogue_text = extract_tag_content(dialogue_text, 'dialogue')
    dialogue_text = dialogue_text.strip()
    dialogue_text = dialogue_text.replace('<dialogue>\n', '')
    dialogue_text = dialogue_text.replace('\n</dialogue>', '')
    dialogue_history, code_states = parse_dialogue_text(dialogue_text)
    if len(dialogue_history) > 0:
        st.session_state.dialogue_history = dialogue_history
        st.session_state.code_states = code_states
        st.session_state.speaker_id = 1 if dialogue_history[-1]['speaker'] == 'User' else 0

    show_format = create_interface_text(st.session_state)
    st.session_state.code_content = st.session_state.buggy_code.replace('```py\n', '').replace('\n```', '')
    save_page(dialogue_text, show_format, PAGE_NUMBER)

    
def run_interface_ide():
    
    c1, c2 = st.columns([1, 1])
    with c2:
        chat_container = st.container()
        speaker = st.selectbox('Next Speaker:', ['Student', 'Teacher'], index = st.session_state.speaker_id, disabled=True)
        speaker = 'User' if speaker == 'Student' else 'Assistant'
        with st.form("my_form", clear_on_submit=True):
            user_utterance = st.text_area('Utterance:', key='u_input', help= "Enter the student or instructor utterance here.", )
            b1, b2, b3 = st.columns([1, 1, 1])
            chat_submit  = b1.form_submit_button('➕ Main', help= "Add a main utterance to the dialogue.")
            alt_submit = b3.form_submit_button('➕ Alternative', disabled= st.session_state.dialogue_history == [], help= "Add an alternative utterance for the last utterance.")
            undo_button = b2.form_submit_button('Undo', help= "Undo the last utterance.")
        
        if undo_button:
            if len(st.session_state.page_memory[PAGE_NUMBER]) > 0:
                session_state_dict = st.session_state.page_memory[PAGE_NUMBER].pop()
                # load values from previous session state
                for key in ['dialogue_history', 'displayed_dialogue_text', 'speaker_id', 'code_states', 'program_output', 'speaker_id']:
                    st.session_state[key] = session_state_dict[key]
                st.experimental_rerun()

        vetting_container = st.container()
       

        if chat_submit or st.session_state.update_chat:
                # store previous session state
                st.session_state.page_memory[PAGE_NUMBER].append(deepcopy(dict((key,value) for key, value in dict(st.session_state).items() if key in ['dialogue_history', 'displayed_dialogue_text', 'speaker_id', 'code_states', 'program_output', 'speaker_id'])))

                # trim page memory to 5
                st.session_state.page_memory[PAGE_NUMBER] = st.session_state.page_memory[PAGE_NUMBER][-5:]
                update_chat_interface(user_utterance.strip(), speaker)
                if speaker == 'User' and user_utterance != '':
                    st.session_state.speaker_id = 1
                elif speaker == 'Assistant' and user_utterance != '':
                    st.session_state.speaker_id = 0
                st.session_state.update_chat = False
                st.session_state.update_state = False
                st.session_state.is_exported = False
                st.experimental_rerun()
      
        if alt_submit:
            # store previous session state
            st.session_state.page_memory[PAGE_NUMBER].append(deepcopy(dict((key,value) for key, value in dict(st.session_state).items() if key in ['dialogue_history', 'displayed_dialogue_text', 'speaker_id', 'code_states', 'program_output', 'speaker_id'])))
            # trim page memory to 10
            st.session_state.page_memory[PAGE_NUMBER] = st.session_state.page_memory[PAGE_NUMBER][-5:]
            update_chat_interface(user_utterance.strip(), 'alt')
            st.session_state.is_exported = False
            # st.experimental_rerun()
   

        if st.session_state.update_state and chat_submit == False and alt_submit == False: 
            st.session_state.dialogue_history, st.session_state.code_states = parse_visible_dialogue_text(st.session_state.displayed_dialogue_text)
            st.session_state.dialogue_text = create_dialogue_text(st.session_state)
            if len(st.session_state.dialogue_history) > 0:
                    # loop through dialogue history in reverse order to find the last speaker User or Assistant
                    for i in range(len(st.session_state.dialogue_history) -1, -1, -1):
                        if st.session_state.dialogue_history[i]['speaker'] == 'User':
                            st.session_state.speaker_id = 1
                            break
                        elif st.session_state.dialogue_history[i]['speaker'] == 'Assistant':
                            st.session_state.speaker_id = 0
                            break
            if len(st.session_state.page_memory[PAGE_NUMBER]) > 0 and st.session_state.page_memory[PAGE_NUMBER][-1]['displayed_dialogue_text'] != st.session_state.displayed_dialogue_text: 
                st.session_state.is_exported = False
                # store new session state
                st.session_state.page_memory[PAGE_NUMBER].append(deepcopy(dict((key,value) for key, value in dict(st.session_state).items() if key in ['dialogue_history', 'displayed_dialogue_text', 'speaker_id', 'code_states', 'program_output', 'speaker_id'])))
                # trim page memory to 10
                st.session_state.page_memory[PAGE_NUMBER] = st.session_state.page_memory[PAGE_NUMBER][-5:]
            # st.experimental_rerun()
            st.session_state.update_state = False

        if st.session_state['dialogue_history']:
           
            new_text = create_interface_text(st.session_state)
            # set height of the dialogue text area
            height = 50 + 20 * new_text.count('\n') # 450
            st.session_state.displayed_dialogue_text = chat_container.text_area('Chat History', 
                                                        value=new_text, height=height, max_chars=None,
                                                        key='text_area', help=None, on_change=text_area_changed,
                                                        args=None, kwargs=None).strip()
            
        dialogue_text = create_dialogue_text(st.session_state)
        exported_text = '<problem>\n' + st.session_state.problem_instruction + '\n</problem>' + '\n' + '<bug_code>\n' + add_line_nums(st.session_state.buggy_code.replace('```py\n', '').replace('\n```', '')).strip() + '\n</bug_code>' + '\n<bug_desc>\n' + st.session_state.bug_description + '\n</bug_desc>' + '\n' + '<bug_fixes>\n' + '\n'.join(list(st.session_state.bug_fixes)) + '\n</bug_fixes>' + '\n' + '<unit_tests>\n' + st.session_state.unit_tests + '\n</unit_tests>' + '\n' + '<stu_desc>\n' + st.session_state.student_profile + '\n</stu_desc>' + '\n<dialogue>\n' + dialogue_text.strip() + '\n</dialogue>'
        show_format  = create_interface_text(st.session_state)
        vetting_container.info('Double check the dialogue below and ensure all code lines are properly indented before downloading.')
        with vetting_container.expander("Expand me for Import & Export", expanded=False):
            
            vetting_c1, vetting_c2 = st.columns([1,1])
            dl = vetting_c1.download_button(label="Save & Export Data", data=exported_text, file_name=f"{int(st.session_state.problem_id)}_{int(st.session_state.bug_id)}_{st.session_state.problem_title.replace(' ', '_').lower()}{EXPORT_FILE_NAME}", mime="text/plain")
            if dl:
                save_page(dialogue_text, show_format, PAGE_NUMBER)
                st.session_state.is_exported = True
                st.success('Annotation Downloaded!')
                st.info('Please note that the format of the dialogue in the text file is different from the format in the chat interface.', icon="ℹ️")
    
            with vetting_c2.form("upload_form", clear_on_submit=True):
                imported_text_file = st.file_uploader("Import Dialogue", type=['txt'], key='import_file', help="Upload a text file containing exported data from this tool. Ensure that the filename is unchanged. It should be in the format: <problem_id>_<bug_id>_<problem_title>_<suffix>.txt")
                if st.form_submit_button("Complete Import"):
                    if imported_text_file is not None and imported_text_file.name is not None:
                        # get the base filename from imported_text_file which an UploadedFile object
                        base_filename = os.path.basename(imported_text_file.name)
                        imported_text = imported_text_file.read().decode('utf-8') 
                        import_from_text_file(base_filename, imported_text)
                        st.session_state.page_memory[PAGE_NUMBER].append(deepcopy(dict((key,value) for key, value in dict(st.session_state).items() if key in ['dialogue_history', 'displayed_dialogue_text', 'speaker_id', 'code_states', 'program_output', 'speaker_id'])))
                        # trim page memory to 10
                        st.session_state.page_memory[PAGE_NUMBER] = st.session_state.page_memory[PAGE_NUMBER][-5:]
                        st.info("Please click the 'Complete Import' button to complete the import process. If you already clicked the 'Complete Import' button, please disregard this message.")
                        st.session_state.imported_flag = True
                        imported_text_file.close()
                        st.experimental_rerun()
                
    st.sidebar.header("IDE Parameters")
    
    with c1:
        st.header("Code Editor")        
        code_container = st.empty()
        # to force the code container to be refreshed only if code_content is changed
        if st.session_state.imported_flag is True:
            time.sleep(0.5)
            st.session_state.imported_flag = False
        with code_container.container():
            code_content = st.session_state.code_content +'\n\n' + st.session_state.unit_tests if 'unit_tests' in st.session_state else st.session_state.code_content
            
            content = st_ace(
                value =  code_content,
                placeholder='Write your solution here.',
                language=LANGUAGES[121],
                theme=st.sidebar.selectbox("Theme", options=THEMES, index=35),
                keybinding=st.sidebar.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
                font_size=st.sidebar.slider("Font size", 5, 24, 14),
                tab_size= 4,
                show_gutter=True,
                wrap=st.sidebar.checkbox("Wrap enabled", value=True),
                auto_update=True,
                readonly=False,
                min_lines=35,
                key="ide"
            )
        
        
        program_output_container = st.container()
        p_out_c1, p_out_c2 = program_output_container.columns([1, 1])
        with p_out_c2:
            save_code = st.button('➕ Add Code to Chat History')
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        # add a previous button
        prev_button = st.button(PREV_BUTTON)
        if prev_button:
            st.session_state.load_page_from_exported_memory = True
            nav_page(PREV_PAGE_NAME)
        
        if save_code and content :
            # store previous session state
            st.session_state.page_memory[PAGE_NUMBER].append(deepcopy(dict((key,value) for key, value in dict(st.session_state).items() if key in ['dialogue_history', 'displayed_dialogue_text', 'speaker_id', 'code_states', 'program_output', 'speaker_id'])))
            # trim page memory to 10
            st.session_state.page_memory[PAGE_NUMBER] = st.session_state.page_memory[PAGE_NUMBER][-5:]
            printed_content = content
            # remove unit tests from the code content if they exist
            if 'unit_tests' in st.session_state:
                printed_content = content.replace('\n\n' + st.session_state.unit_tests, '')
            st.session_state.code_content = printed_content
            st.session_state.dialogue_history.append({'txt': '<saved_code>', 'speaker': 'User', 'type': 'code'})
            st.session_state.code_states.append(add_line_nums(printed_content))
            
            st.experimental_rerun()
        with p_out_c1:
            if st.button('Run Code & Print Output'):
                exec_process = subprocess.Popen([sys.executable, '-c',  str(content)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                st.session_state.program_output = str(exec_process.stdout.read().decode())
                program_output_container.header('Program Output')
                program_output_container.text(st.session_state.program_output)
                if st.session_state.program_output == '':
                    program_output_container.success('Program passes all unit tests!')





    # make a reset button
    with c2:
        reset_button = st.button('Reset')
        if reset_button:
            st.session_state.dialogue_history = []
            st.session_state.code_states = []
            st.session_state.code_content = ''
            st.session_state.program_output = ''
            st.session_state.speaker_id = 0
            st.session_state.displayed_dialogue_text = ''
            st.session_state.dialogue_text = ''
            # st.session_state.saved_page_content[PAGE_NUMBER] = ''
            st.experimental_rerun()
    with c3:
        next_button = st.button(NEXT_BUTTON)
        if next_button:
            st.session_state.load_page_from_exported_memory = True
            st.session_state.is_exported = False
            nav_page(NEXT_PAGE_NAME)
            

st.title(PAGE_TITLE)
render_student_profile()
render_bug_info()
render_instructions()
initialize_session_state()
run_interface_ide()