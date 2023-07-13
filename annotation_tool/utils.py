
from streamlit.components.v1 import html
import streamlit as st
from streamlit_ace import st_ace, LANGUAGES, THEMES, KEYBINDINGS
import subprocess
import sys
import time
import regex as re


def nav_page(page_name, timeout_secs=3):
    nav_script = """
        <script type="text/javascript">
            function attempt_nav_page(page_name, start_time, timeout_secs) {
                var links = window.parent.document.getElementsByTagName("a");
                for (var i = 0; i < links.length; i++) {
                    if (links[i].href.toLowerCase().endsWith("/" + page_name.toLowerCase())) {
                        links[i].click();
                        return;
                    }
                }
                var elasped = new Date() - start_time;
                if (elasped < timeout_secs * 1000) {
                    setTimeout(attempt_nav_page, 100, page_name, start_time, timeout_secs);
                } else {
                    alert("Unable to navigate to page '" + page_name + "' after " + timeout_secs + " second(s).");
                }
            }
            window.addEventListener("load", function() {
                attempt_nav_page("%s", new Date(), %d);
            });
        </script>
    """ % (page_name, timeout_secs)
    html(nav_script)


def create_ide(problem, col_name, buggy_code=None):
    code_container = st.empty()
    time.sleep(0.75)
    with code_container.container():
        code = problem[col_name].replace('```py\n', '').replace('\n```', '') if buggy_code is None else buggy_code[col_name].replace('```py\n', '').replace('\n```', '')
        ace_content = st_ace(
            value =   code + "\n\n" + problem['unit_tests'],
            language=LANGUAGES[121],
            theme=THEMES[35],
            keybinding=KEYBINDINGS[3],
            font_size=14,
            tab_size= 4,
            show_gutter=True,
            wrap=True,
            auto_update=False,
            readonly=False,
            min_lines=5,
            key=f"{col_name}_{st.session_state.ind}",
        )

        if st.button("Compile and Run", key=f"run_{col_name}"):
            with st.spinner("Running..."):
                exec_process = subprocess.Popen([sys.executable, '-c',  str(ace_content)  ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                st.text(str(exec_process.stdout.read().decode()))
            st.success("Done!")


def convert_speaker(speaker_name, convert_dict={ "User": "Student",
                                                 "Assistant": "Instructor",
                                                 "alt" : "alt"
                                                 }):
    return convert_dict[speaker_name]

"""
Extracts the text between the specified tag in the input string.
Removes all text before the opening tag and after the closing tag.
"""
def extract_tag_content(input_str, tag_name):
    regex = fr'^.*?<{tag_name}>|</{tag_name}>.*$'
    target_str = re.sub(regex, "", input_str, flags=re.DOTALL)
    return target_str


def create_interface_text(st_session_state):
    # chatbox_container.empty()
    # with chatbox_container.container():
    dialogue_text = ''
    code_state_index = 0
    for i in range(len(st_session_state.dialogue_history)):
        utterance = st_session_state.dialogue_history[i]['txt']
        if utterance != '<saved_code>':
            if st_session_state.dialogue_history[i]['type'] == 'alt':
                # check if previous utterance is not an alt utterance
                if i > 0 and st_session_state.dialogue_history[i-1]['type'] != 'alt':
                    dialogue_text += 'Alternatives:\n'
                dialogue_text += '\t• ' + st_session_state.dialogue_history[i]['txt'] + '\n'
            else:
                if dialogue_text.endswith('\n\n') == False and len(dialogue_text.strip()) > 1:
                    dialogue_text += '\n'
                speaker = convert_speaker(st_session_state.dialogue_history[i]['speaker'])
                dialogue_text += speaker + ': ' + st_session_state.dialogue_history[i]['txt'] + '\n\n'
        else:
            if code_state_index < len(st.session_state.code_states):
                # check if code state is not empty
                if st_session_state.code_states[code_state_index] != '':
                    if dialogue_text.endswith('\n\n') == False:
                        dialogue_text += '\n'
                    dialogue_text += 'Student Code:\n' + st_session_state.code_states[code_state_index]
                    if not st_session_state.code_states[code_state_index].endswith('\n'):
                        dialogue_text += '\n\n'
            code_state_index += 1

    return dialogue_text

def create_dialogue_text(st_session_state):
    dialogue_text = ''
    code_state_index = 0
    for i in range(len(st_session_state.dialogue_history)):
        utterance = st_session_state.dialogue_history[i]['txt']
        if utterance != '<saved_code>':
            if st_session_state.dialogue_history[i]['type'] == 'alt':
                dialogue_text += '\t<alt>' + st_session_state.dialogue_history[i]['txt'] + '\n'
            else:
                dialogue_text += st_session_state.dialogue_history[i]['speaker'] + ': ' + st_session_state.dialogue_history[i]['txt'] + '\n'
        else:
            if code_state_index < len(st.session_state.code_states):
                # check if code state is not empty
                if st_session_state.code_states[code_state_index] != '':
                    dialogue_text += '<code>\n' + st_session_state.code_states[code_state_index] + "\n</code>" + '\n'
                
            code_state_index += 1
    return dialogue_text

# TODO: Supprt Multi-line utterances
def parse_visible_dialogue_text(dialogue_text):
    dialogue_history = []
    code_states = []
    code_state = ''
    for line in dialogue_text.splitlines():
        if line.startswith('\t•'):
            if code_state != '':
                dialogue_history.append({'txt': '<saved_code>', 'speaker': 'User', 'type': 'code'})
                code_states.append(code_state)
                code_state = ''
            dialogue_history.append({'txt': line.replace('\t• ', ''), 'speaker': dialogue_history[-1]['speaker'], "type": 'alt'})
        elif line.startswith('Student Code:'):
            code_state = ''
        elif line.startswith('Student:'):
            if code_state != '':
                dialogue_history.append({'txt': '<saved_code>', 'speaker': 'User', 'type': 'code'})
                code_states.append(code_state)
                code_state = ''
            dialogue_history.append({'txt': line.replace('Student: ', ''), 'speaker': 'User', "type": 'main'})
            
        elif line.startswith('Instructor:'):
            if code_state != '':
                dialogue_history.append({'txt': '<saved_code>', 'speaker': 'User', 'type': 'code'})
                code_states.append(code_state)
                code_state = ''
            dialogue_history.append({'txt': line.replace('Instructor: ', ''), 'speaker': 'Assistant', "type": 'main'})
        elif line.startswith('Alternatives:'):
            continue
        
        elif line != '':
            code_state += line + '\n'
        else:
            continue
    
    if code_state != '':
        code_states.append(code_state)
        dialogue_history.append({'txt': '<saved_code>', 'speaker': 'User', 'type': 'code'})
        code_state = ''

    return dialogue_history, code_states

def parse_dialogue_text(dialogue_text):
    dialogue_history = []
    code_states = []
    code_state = ''

    for line in dialogue_text.splitlines():
        if line.startswith('\t<alt>'):
            dialogue_history.append({'txt': line.replace('\t<alt>', ''), 'speaker': dialogue_history[-1]['speaker'], "type": 'alt'})
        elif line.startswith('</code>') and code_state != '':  
            code_states.append(code_state)
            code_state = ''
            dialogue_history.append({'txt': '<saved_code>', 'speaker': 'User', 'type': 'code'})
        elif line.startswith('User:'):
            dialogue_history.append({'txt': line.replace('User: ', ''), 'speaker': 'User', "type": 'main'})
        elif line.startswith('Assistant:'):
            dialogue_history.append({'txt': line.replace('Assistant: ', ''), 'speaker': 'Assistant', "type": 'main'})
        
        # check if line is empty
        elif line == '':
            continue
        elif line.startswith('<code>') or line.startswith('</code>'):
            continue
        else:
            code_state += line + '\n'
    
    return dialogue_history, code_states
"""Remove line numbers from the code.
    Example:
    1.  def foo():
    2.      print('hello')
    3.  foo()
    becomes:
    def foo():
        print('hello')
    foo()

""" 
def remove_line_numbers (txt):
    lines = txt.splitlines()
    lines = [line.split('. ', 1)[-1] for line in lines]
    return '\n'.join(lines).strip()
   
def add_line_nums(code, markdown=False):
    #TODO: standardize indentation
    code = code.split('\n')
    # code = [f"{i+1}.{line}"  for i, line in enumerate(code) if i + 1 >= 10 else f"{i+1}. {line}"]
    # rewrite code above into a for loop
    for i in range(len(code)):
        if i + 1 >= 10:
            code[i] = f"{i+1}.{code[i]}"
        else:
            code[i] = f"{i+1}.{code[i]}"
    code[0] = code[0][:2] + " " + code[0][2:]
    code = '\n'.join(code)
    if markdown:
        return "```py\n" + code + "\n```"
    else:
        return code