from data_utils.data_reader import XMLDataReader
import pytest

def test_parse_dialogue():
    data_reader = XMLDataReader('reviewing_dialogues')
    
    dialogue_text = """
User: I am trying to run the code but it is not working.
    <alt> Nothing is working.
Assistant: What is the error message?
    <alt> Any error message?
    <alt> What does the terminal say?
User: I figured it out. I was missing a comma.
<code>
import pandas as pd
if True:
    df = pd.read_csv('data.csv' , sep=',')
</code>
    <alt> Ah! I think it is because the file name is wrong.
<code>
import pandas as pd
if True:
    df = pd.read_csv('my_data.csv')
</code>
Assistant: Great! It is working now.
    <alt> Awesome! Good job!
User: Thank you for your help.
    <alt> Thanks!
"""

    expected_output = [
        {
            'user_text': 'I am trying to run the code but it is not working.',
            'code_text': '',
            'cumulative_code_text': '',
            'assistant_text': 'What is the error message?',
            'user_alternatives': [
                {
                    'alt_text': 'Nothing is working.',
                    'alt_code': '',
                }
            ],
            'assistant_alternatives': ['Any error message?', 'What does the terminal say?'],
        },
        {
            'user_text': 'I figured it out. I was missing a comma.',
            'code_text': "import pandas as pd\nif True:\n    df = pd.read_csv('data.csv' , sep=',')",
            'cumulative_code_text': "import pandas as pd\nif True:\n    df = pd.read_csv('data.csv' , sep=',')",
            'assistant_text': 'Great! It is working now.',
            'user_alternatives': [
                {
                    'alt_text': 'Ah! I think it is because the file name is wrong.',
                    'alt_code': "import pandas as pd\nif True:\n    df = pd.read_csv('my_data.csv')",
                }
            ],
            'assistant_alternatives': ['Awesome! Good job!'],
        },
        {
            'user_text': 'Thank you for your help.',
            'code_text': "",
            'cumulative_code_text': "import pandas as pd\nif True:\n    df = pd.read_csv('data.csv' , sep=',')",
            'assistant_text': '',
            'user_alternatives': [
                {
                    'alt_text': 'Thanks!',
                    'alt_code': '',
                }
            ],
            'assistant_alternatives': [],
        }
    ]

    assert expected_output == data_reader._parse_dialogue(dialogue_text)