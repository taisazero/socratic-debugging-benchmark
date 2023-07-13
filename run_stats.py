import unittest
from data_utils.data_reader import XMLDataReader
import statistics
import os

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, 'final_dataset')
data_reader = XMLDataReader(path)
dialogue_texts = data_reader.get_dialogues() # list of strings
print(len(dialogue_texts))
dialogue_stats = data_reader.compute_dialogue_stats(dialogue_texts)
print(dialogue_stats)


class TestDialogueStatistics(unittest.TestCase):
    # set maxDiff to None to see full diff
    maxDiff = None


    def test_single_dialogue(self):
        dialogues = [
            "\nUser: Hey, I really need help. This function is not returning the correct number.\nAssistant: I'll be happy to help you. What happens when you pass 5 as `count` to the function `fibonacci(5)`?\n\t<alt>Happy to help. Can you give me an example of your function returning an incorrect number?\n\t<alt>Sure. Can you list the first 5 numbers in the Fibonacci sequence?\n\t<alt>Can you describe what your code does line by line?\nUser: When I run the function with `n` as 5 the output is 8 instead of 5.\nAssistant: I see. What do you think might be the cause of this?\n\t<alt>Can you describe your for loop on line 11?\n\t<alt>Can you walk me through the steps your function takes to generate the Fibonacci sequence?\n\t<alt>Can you identify any issues with the loop that generates the Fibonacci sequence?\n\t<alt>Which part of your code do you suspect of causing this issue?\nUser: I guess maybe the problem is that `a` is set to 0 not 1? I am not really sure.\n\t<alt>I am not sure. Can you help?\n\t<alt>I think the cause of this is probably the for loop since that's the only place where I compute `b`, the value that I return.\n<code>\n1. def foo(n):\n2.  return n\n\n</code>\n\n\t<alt>I am not sure. Can you help?"
        ] 
        result = data_reader.compute_dialogue_stats(dialogues)
        expected = {
            'total_turns': 2,
            'total_utterances': 15,
            'descriptive_stats': {
                'assistant_main': {'total': 2, 'mean': 2, 'median': 2, 'stdev': 0},
                'assistant_alt': {'total': 7, 'mean': 7, 'median': 7, 'stdev': 0},
                 'assistant_total': {'total': 9, 'mean': 9, 'median': 9, 'stdev': 0},
                'user_main': {'total': 3, 'mean': 3, 'median': 3, 'stdev': 0},
                'user_alt': {'total': 3, 'mean': 3, 'median': 3, 'stdev': 0},
                'user_total': {'total': 6, 'mean': 6, 'median': 6, 'stdev': 0},
            },
        }
        self.assertEqual(result, expected)

    def test_multiple_dialogues(self):
        dialogues = [
            "\nUser: Question 1.\nAssistant: Answer 1.\n\t<alt>Alternative answer 1.\nUser: Question 2.\nAssistant: Answer 2.\n\t<alt>Alternative answer 2.\n",
            "\nUser: Question 3.\nAssistant: Answer 3.\n\t<alt>Alternative answer 3.\n\t<alt>Alternative answer 4.\nUser: Question 4.\nAssistant: Answer 4.\n",
        ]
        result = data_reader.compute_dialogue_stats(dialogues)
        expected = {
            'total_turns': 4,
            'total_utterances': 12,
            'descriptive_stats': {
                'assistant_main': {'total': 4, 'mean': 2, 'median': 2, 'stdev': statistics.stdev([2, 2]) if len([2,2]) > 1 else 0},
                'assistant_alt': {'total': 4, 'mean': 2, 'median': 2, 'stdev': statistics.stdev([2, 2]) if len([2 , 2]) > 1 else 0},
                'assistant_total': {'total': 8, 'mean': 4, 'median': 4, 'stdev': statistics.stdev([4, 4]) if len([4, 4]) > 1 else 0},
                'user_main': {'total': 4, 'mean': 2, 'median': 2, 'stdev': statistics.stdev([2, 2]) if len([2, 2]) > 1 else 0},
                'user_alt': {'total': 0, 'mean': 0, 'median': 0, 'stdev': statistics.stdev([0, 0]) if len([0, 0]) > 1 else 0},
                'user_total': {'total': 4, 'mean': 2, 'median': 2, 'stdev': statistics.stdev([2, 2]) if len([2, 2]) > 1 else 0},
            },
        }
        self.assertEqual(result, expected)

    def test_empty_dialogue(self):
        dialogues = [""]
        result = data_reader.compute_dialogue_stats(dialogues)
        expected = {
            'total_turns': 0,
            'total_utterances': 0,
            'descriptive_stats': {
                'assistant_main': {'total': 0, 'mean': 0, 'median': 0, 'stdev': 0},
                'assistant_alt': {'total': 0, 'mean': 0, 'median': 0, 'stdev': 0},
                'assistant_total': {'total': 0, 'mean': 0, 'median': 0, 'stdev': 0},
                'user_main': {'total': 0, 'mean': 0, 'median': 0, 'stdev': 0},
                'user_alt': {'total': 0, 'mean': 0, 'median': 0, 'stdev': 0},
                'user_total': {'total': 0, 'mean': 0, 'median': 0, 'stdev': 0},
            },
        }
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()