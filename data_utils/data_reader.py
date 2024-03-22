"""
This module contains the DataReader class, which is used to read data from a
CSV file with the following columns:
    * Language model instruction (rephrase the first sentence in the example under Instruction)
    * Programming problem instructions in markdown (i.e. between ``` ```)	
    * Context Code: Enter the [faulty] code that the user is working with or starting with.	
    * Describe the bug, its behavior, the fix (if applicable) and how the solution should behave in first person. Use this example for reference: I notice two issues with the user's code snippet. The first issue is the misuse of the equality operator where the user wrote `lst = []`, which assigns an empty list to 'lst', instead of writing `lst ==[]` which tests if 'lst' is empty. The second issue is giving the two arguments for the 'replace' function in the wrong order in `lst.insert(x, len(lst))`, which results in inserting the length of the list `lst` at index `x`, instead of writing `lst.insert(len(lst), x)`, which results in inserting `x` at the end of `lst`.
    * Dialog: Enter the dialogue utterances prepended with the `User:` and `Assistant:` Remember to enter <thought_block>thought here</thought_block>, before assistant utterances, and the Updated code: after each Socratic interaction where the user changes something in their code and record an updated code snippet. Recall, that Socratic questions facilitate the discovery of one of the following processes: Remember (recall information): e.g. What are the file opening modes?  Demonstrate understanding by explaining an idea or a concept: e.g. Can you explain how you are looping through the array? Apply their understanding: e.g. What do you think will happen if you run your code with the input “hello”? Analyze (examine code, concepts, make connections): e.g. What do you notice after running your code? Evaluate: What do you think might be missing here? Create (invites the programmer to do something or make an action plan): Given that information, what do you think you should do next?  You can use a web-app here for this block for your convenience: https://socratic-debugging.streamlit.app/
    * Resulting Code: Enter the user's code snippet in the markdown after the dialogue is completed
    * Comments: Add any comments or a data source you used to adapt the programming problem from

The DataReader class is used to read the data from the CSV file and return a dataframe with the data with simpler column names.
Dataframe columns:
    * instruction: Language model instruction (rephrase the first sentence in the example under Instruction)
    * problem: Programming problem instructions in markdown (i.e. between ``` ```)
    * faulty_code: Context Code: Enter the [faulty] code that the user is working with or starting with.
    * bug_description: Describe the bug, its behavior, the fix (if applicable) and how the solution should behave in first person. Use this example for reference: I notice two issues with the user's code snippet. The first issue is the misuse of the equality operator where the user wrote `lst = []`, which assigns an empty list to 'lst', instead of writing `lst ==[]` which tests if 'lst' is empty. The second issue is giving the two arguments for the 'replace' function in the wrong order in `lst.insert(x, len(lst))`, which results in inserting the length of the list `lst` at index `x`, instead of writing `lst.insert(len(lst), x)`, which results in inserting `x` at the end of `lst`.
    * dialogue: Enter the dialogue utterances prepended with the `User:` and `Assistant:` Remember to enter <thought_block>thought here</thought_block>, before assistant utterances, and the Updated code: after each Socratic interaction where the user changes something in their code and record an updated code snippet. Recall, that Socratic questions facilitate the discovery of one of the following processes: Remember (recall information): e.g. What are the file opening modes?  Demonstrate understanding by explaining an idea or a concept: e.g. Can you explain how you are looping through the array? Apply their understanding: e.g. What do you think will happen if you run your code with the input “hello”? Analyze (examine code, concepts, make connections): e.g. What do you notice after running your code? Evaluate: What do you think might be missing here? Create (invites the programmer to do something or make an action plan): Given that information, what do you think you should do next?  You can use a web-app here for this block for your convenience: https://socratic-debugging.streamlit.app/
    * resulting_code: Resulting Code: Enter the user's code snippet in the markdown after the dialogue is completed
    * comments: Add any comments or a data source you used to adapt the programming problem from
    """

import pandas as pd
import regex as re
import statistics
from unidecode import unidecode
from datasets import Dataset
import json
import itertools
import random

random.seed(42)


class ExcelDataReader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.data.columns = [
            "worker_id",
            "task_id",
            "task_response_id",
            "instruction",
            "problem",
            "faulty_code",
            "bug_description",
            "dialogue",
            "resulting_code",
            "comments",
        ]
        self.data = self.data.dropna()
        self.data = self.data.reset_index(drop=True)
        self.data = self.data[self.data["dialogue"].str.contains("User:")]
        self.data = self.data.reset_index(drop=True)

    """
    This method processes the dialogue column in the dataset into a list of User-Assistant turns.
   
    Returns:
        turns (list): A list of User-Assistant turns with the following format:
            [
                {   
                    context: "Programming problem: <problem>\n
                              Faulty Code: <faulty_code>\n
                              Bug Description: <bug_description>\n
                              <instruction>\n
                              <dialogue_until_current_turn>\n",
                    assistant_utterance: "Asisstant: <assistant_utterance>"
                    user_utterance: "User: <user_utterance>"    
                },
                ...
            ]

    """

    def process_dialogues_into_turns(self):
        turns = []

        for dialogue_ind, dialogue in enumerate(self.data["dialogue"]):
            dialogue = dialogue.split("Assistant: ")
            # fix problem formatting
            problem = self.data["problem"][dialogue_ind].strip()
            # remove markdown formatting
            problem = re.sub(r"```", "", problem)
            display_problem = problem
            # if problem is not in markdown format
            # Note that the problem has line breaks
            # so we need our regex to match any character and line breaks
            if not re.search(r"```.*```", problem, re.DOTALL):
                # add markdown formatting
                display_problem = "```\n" + problem + "\n```"

            # remove any extra line breaks at the end of the problem before the ```
            display_problem = re.sub(r"\s+```", "\n```", problem)

            # fix faulty code formatting
            faulty_code = self.data["faulty_code"][dialogue_ind]
            # if faulty code is not in markdown format
            if not re.search(r"```.*```", faulty_code, re.DOTALL):
                # add markdown formatting
                faulty_code = "```" + faulty_code + "```"
            # if faulty code has a language identifier but is after a space

            if re.search(r"```\s+py", faulty_code):
                faulty_code = faulty_code.replace("``` py", "```py")

            # if faulty code does not have a language identifier
            elif not re.search(r"```.*\s", faulty_code):
                # add python language identifier
                faulty_code = faulty_code.replace("```", "```py")

            # remove any extra line breaks at the end of the faulty code before the ```
            faulty_code = re.sub(r"\s+```", "\n```", faulty_code)

            # initialize the rolling context
            rolling_context = (
                "Programming problem: "
                + problem
                + "\n"
                + "Code:\n"
                + faulty_code
                + "\n"
                + "Bug Description: "
                + self.data["bug_description"][dialogue_ind].strip()
                + "\n"
                + self.data["instruction"][dialogue_ind].strip()
                + "\n"
            )

            display_instruction = self.data["instruction"][dialogue_ind].strip() + "\n"
            # initialize the display context
            display_context = (
                "Programming problem: \n"
                + display_problem
                + "  \n"
                + "Faulty Code: \n"
                + faulty_code
                + "  \n"
                + "Bug Description: "
                + self.data["bug_description"][dialogue_ind].strip()
                + "  \n"
                + self.data["instruction"][dialogue_ind].strip()
                + "  \n"
            )
            # dialgoue starts with user or assistant utterance

            for turn_ind, turn in enumerate(dialogue):
                # identify the User utterance and distinguish it
                # from the Assistant utterance
                if "User: " in turn:
                    user_utterance_index = turn.index("User: ")
                    if dialogue[0].startswith("User: "):
                        if (
                            turn_ind + 1 < len(dialogue)
                            and "User: " in dialogue[turn_ind + 1]
                        ):
                            assistant_utterance_end = dialogue[turn_ind + 1].index(
                                "User: "
                            )
                            assistant_utterance = (
                                "Assistant: "
                                + dialogue[turn_ind + 1][
                                    :assistant_utterance_end
                                ].strip()
                            )
                        elif (
                            turn_ind + 1 < len(dialogue)
                            and "User: " not in dialogue[turn_ind + 1]
                        ):
                            assistant_utterance = (
                                "Assistant: " + dialogue[turn_ind + 1].strip()
                            )
                        else:
                            assistant_utterance = ""
                    elif turn[:user_utterance_index] != "":
                        assistant_utterance = (
                            "Assistant: " + turn[:user_utterance_index].strip()
                        )
                    else:
                        assistant_utterance = ""
                    user_utterance = turn[user_utterance_index:].strip()
                    user_utterance, thoughts, display_assistant_utterance = (
                        self.parse_thoughts(
                            user_utterance, assistant_utterance, return_utterance=True
                        )
                    )
                    # add the turn to the list of turns
                    turns.append(
                        {
                            "dialogue_ind": dialogue_ind,
                            "turn_ind": turn_ind,
                            "context": rolling_context,
                            "display_context": display_context,
                            "display_instruction": display_instruction,
                            "assistant_utterance": assistant_utterance,
                            "display_assistant_utterance": display_assistant_utterance,
                            "user_utterance": user_utterance,
                            "assistant_thoughts": thoughts,
                            "code_state": "",
                            "resulting_code": self.data["resulting_code"][dialogue_ind],
                            "user_initiated": dialogue[0].startswith("User: "),
                        }
                    )
                    # If the dialogue starts with an assistant utterance
                    # we need to add the assistant utterance to the context first
                    # # add the context to the current turn
                    if dialogue[0].startswith("User: "):
                        rolling_context = (
                            rolling_context
                            + "\n"
                            + user_utterance
                            + "\n"
                            + assistant_utterance
                        )
                        display_context = (
                            display_context
                            + "  \n"
                            + user_utterance
                            + "  \n"
                            + display_assistant_utterance
                        )

                    else:
                        rolling_context = (
                            rolling_context
                            + "\n"
                            + assistant_utterance
                            + "\n"
                            + user_utterance
                        )
                        display_context = (
                            display_context
                            + "  \n"
                            + display_assistant_utterance
                            + "  \n"
                            + user_utterance
                        )

        # sort by dialogue index and turn index
        turns = sorted(turns, key=lambda k: (k["dialogue_ind"], k["turn_ind"]))

        return turns

    def filter_turns(self, turns, filter_by, filter_value):
        filtered_turns = []
        for turn in turns:
            if turn[filter_by] == filter_value:
                filtered_turns.append(turn)
        return filtered_turns

    """
    This method filters out duplicate turns from the list of turns.
    A turn is considered duplicate if the context, assistant utterance and user utterance are the same.
    """

    def filter_unique_turns(self, turns):
        filtered_turns = set()

        for turn in turns:
            turn_text = (
                turn["context"]
                + "\n"
                + turn["assistant_utterance"]
                + "\n"
                + turn["user_utterance"]
            )
            filtered_turns.add(turn_text)

        filtered_turns = list(filtered_turns)

        for turn in turns:
            turn_text = (
                turn["context"]
                + "\n"
                + turn["assistant_utterance"]
                + "\n"
                + turn["user_utterance"]
            )
            if turn_text in filtered_turns:
                turns.remove(turn)

        return turns

    """
    Parse the assistant utterance to extract the thoughts of the assistants marked by
    the <thought_block> </thought_block> tags.
    """

    def parse_thoughts(
        self, user_utterance, assistant_utterance, return_utterance=False
    ):
        if user_utterance == "":
            return ""
        # extract the thoughts

        thoughts = re.findall(r"<thought_block>(.*?)</thought_block>", user_utterance)
        thoughts = "".join(thoughts)
        # remove the thoughts from the user utterance and add them to the assistant utterance
        user_utterance = re.sub(
            r"<thought_block>.*?</thought_block>", "", user_utterance
        )
        assistant_utterance = (
            "<thought_block>"
            + thoughts.strip()
            + "</thought_block>"
            + "  \n"
            + assistant_utterance
        )
        if return_utterance:
            return user_utterance, thoughts, assistant_utterance
        else:
            return user_utterance, thoughts


"""
Extracts the text between the specified tag in the input string.
Removes all text before the opening tag and after the closing tag.
"""


def extract_tag_content(input_str, tag_name):
    regex = rf"^.*?<{tag_name}>|</{tag_name}>.*$"
    target_str = re.sub(regex, "", input_str, flags=re.DOTALL)
    return target_str


"""
This module contains the XMLDataReader class, which is used to read data from an
XML file with the following tags:
    <dialogue>
    <problem>
    <bug_code>
    <bug_desc>
    <bug_fixes>
    <unit_tests>
    <stu_desc>

It returns a list of DataFrames with the following columns:
    dialogue: parsed dialogue text into a list of utterances
    problem: problem description
    bug_code: buggy code
    bug_desc: bug description
    bug_fixes: bug fixes
    unit_tests: unit tests
    stu_desc: student description

"""
import os
import regex as re


class XMLDataReader:
    def __init__(
        self,
        folder_path,
        tag_names=[
            "dialogue",
            "problem",
            "bug_code",
            "bug_desc",
            "bug_fixes",
            "unit_tests",
            "stu_desc",
        ],
    ):
        self.file_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".txt")
        ]
        self.tag_names = tag_names
        self.data = self.read_data()

    def read_data(self):
        data = []
        for self.file_path in self.file_paths:
            file_txt = ""
            with open(self.file_path, "r") as f:
                try:
                    file_txt = f.read()
                except UnicodeDecodeError:
                    print("UnicodeDecodeError: ", self.file_path)
                    with open(self.file_path, "r", encoding="utf-8") as f:
                        file_txt = f.read()
                        file_txt = unidecode(file_txt)

                dic = {
                    #'problem_id': self.file_path.split('/')[-1].split('.')[0].split('_')[0],
                    #'bug_id': self.file_path.split('/')[-1].split('.')[0].split('_')[1]
                    # Make path splitting dynamic for Windows and Linux
                    "problem_id": self.file_path.split(os.sep)[-1]
                    .split(".")[0]
                    .split("_")[0],
                    "bug_id": self.file_path.split(os.sep)[-1]
                    .split(".")[0]
                    .split("_")[1],
                }
                for tag_name in self.tag_names:
                    dic[tag_name] = extract_tag_content(file_txt, tag_name).strip()

                data.append(dic)
        return data

    def get_data(self):
        return self.data

    def get_dialogues(self):
        return [d["dialogue"] for d in self.data]

    """
    This method returns a list of dictionaries with the following keys:
        context_text: the context text
        problem: problem description
        bug_code: buggy code
        bug_desc: bug description
        bug_fixes: bug fixes
        unit_tests: unit tests
        stu_desc: student description

    Returns:
        list: list of dictionaries with the keys:
            dialogue: parsed dialogue text into a list of utterances
            context: a dictionary with the following keys:
                context_text: the context text
                problem: problem description
                bug_code: buggy code
                bug_desc: bug description
                bug_fixes: bug fixes
                unit_tests: unit tests
                stu_desc: student description
    """

    def get_parsed_dialogues(
        self, dialogue_tag_name="dialogue", exclude_tags=["stu_desc"]
    ):
        returned_data = []
        for dic in self.data:
            context_dict = {}
            context_dict["context_text"] = ""
            for tag in self.tag_names + ["problem_id", "bug_id"]:
                if tag != dialogue_tag_name and tag not in exclude_tags:
                    context_dict[tag] = dic[tag]
                    # exclude the problem_id and bug_id from the context text
                    if tag not in ["problem_id", "bug_id"]:
                        context_dict["context_text"] += (
                            "<"
                            + tag
                            + ">"
                            + "\n"
                            + dic[tag]
                            + "\n"
                            + "</"
                            + tag
                            + ">"
                            + "\n"
                        )
            parsed_dialogue = self._parse_dialogue(dic[dialogue_tag_name])
            returned_data.append({"dialogue": parsed_dialogue, "context": context_dict})

        return returned_data

    """
        Converts parsed dialogues into a Hugging Face Dataset object.

        This method processes parsed dialogues and creates a Hugging Face Dataset object
        with fields for context, dialogue context, assistant response, and assistant alternatives.

        Returns:
            datasets.Dataset: A Hugging Face Dataset object containing formatted dialogue data.

        Example:
            data_reader = XMLDataReader("path_to_dataset_folder")
            dialogue_dataset = data_reader.export_to_hf_dataset_t5()
            print(dialogue_dataset)
    """

    def export_to_hf_dataset_t5(
        self,
        user_id="User",
        assistant_id="Assistant",
        instruction="Respond to the user with a bulleted list of all possible distinct Socratic utterances that guide the user to discover and fix the bug.",
    ):
        dataset_dict = {
            "problem_id": [],
            "bug_id": [],
            "context": [],
            "dialogue_context": [],
            "assistant_response": [],
            "assistant_alternatives": [],
            "lm_input": [],
            "lm_target": [],
        }

        parsed_dialogues = self.get_parsed_dialogues()

        for parsed_dialogue_object in parsed_dialogues:
            dialogue = parsed_dialogue_object["dialogue"]
            context_dict = parsed_dialogue_object["context"]
            context_text = context_dict["context_text"]

            for i, turn in enumerate(dialogue):
                dialogue_context = ""

                # Prepare context and dialogue context
                for j in range(i + 1):
                    prev_turn = dialogue[j]
                    dialogue_context += f"{user_id}: {prev_turn['user_text']}\n"
                    if prev_turn["code_text"] != "":
                        dialogue_context += (
                            f"<code>\n{prev_turn['code_text']}\n</code>\n"
                        )
                    if j < i:
                        dialogue_context += (
                            f"{assistant_id}: {prev_turn['assistant_text']}\n"
                        )

                # Add to dataset
                dataset_dict["problem_id"].append(context_dict["problem_id"])
                dataset_dict["bug_id"].append(context_dict["bug_id"])
                dataset_dict["context"].append(context_text)
                dataset_dict["dialogue_context"].append(dialogue_context.strip())
                dataset_dict["assistant_response"].append(turn["assistant_text"])
                dataset_dict["assistant_alternatives"].append(
                    turn["assistant_alternatives"]
                )
                dataset_dict["lm_input"].append(
                    context_text + "\n" + dialogue_context.strip() + "\n" + instruction
                )
                # numeric string list of assistant_text and assistant_alternatives
                dataset_dict["lm_target"].append(
                    "\n* ".join(
                        [
                            t
                            for t in ["* " + turn["assistant_text"]]
                            + turn["assistant_alternatives"]
                            if t != ""
                        ]
                    )
                )

        return Dataset.from_dict(dataset_dict)

    """
        Converts parsed dialogues into JSONL format for fine-tuning OpenAI models.
        The format is as follows:
        ```
        {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
        {"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
        ...
        ```
        Each sample is a JSON object with a single key "messages" whose value is a list of JSON objects with keys "role" and "content". Role is 'system', 'user', or 'assistant'. Content is the utterance.
        Parameters:
            user_id (str): The user id to use in the JSON object. Defaults to 'User'.
            assistant_id (str): The assistant id to use in the JSON object. Defaults to 'Assistant'.
            disable_permutation (bool): If True, do not create permutations of the assistant utterances. Defaults to False.
            export_path (str): The path to write the JSONL file to. If not specified, the JSONL file is not written.
            configuation (str): The configuration to use. Defaults to 'gen_all'. Other options are 'gen_given_alts' and 'indp_gen' (see below).
        Returns:
            list: A list of JSON objects in the format above.
            if export_path is specified, the list is written to a JSONL file at the specified path.
            list: A list of dictionaries with the keys:
            problem_id: The problem id.
            bug_id: The bug id.
            dialogue_turn: The dialogue turn number.
            These describe the samples in order of the JSONL file.

        
        There are three ways to formulate the task:
            1. (indp_gen) Generate a valid Socratic utterance independently at each assistant turn. Here the LLM output is a valid Socratic utterance at each assistant turn. The LLM input is the steering prompt, the context, dialogue context. The LLM target is the valid Socratic utterance at each assistant turn which is composed of each unique utterance from the assistant response and the assistant alternatives.
            
            2. (gen_given_alts) Generate one valid Socratic utterance given other alternative Socratic utterances. Here the LLM output is a valid Socratic utterance at each assistant turn. The LLM input is the steering prompt, the context, dialogue context, and either no instructor utterances or a set of alternative utterances. The set of alternative utterances is composed of each unique set of utterances from the assistant response and the assistant alternatives. The LLM target is the valid Socratic utterance at each assistant turn which is composed of each unique utterance from the assistant response and the assistant alternatives. When all the alternative utterances are provided as input the LLM target becomes an empty string. In this configuration, we add 5 permutations of the alternative utterances to the input to encourage the model to generate the same output regardless of the order of the alternative utterances.

            3. (gen_all) Generate all valid Socratic utterances. Here the LLM output is a list of all valid Socratic utterances at each assistant turn. The LLM input is the steering prompt, the context, dialogue context. The LLM target is the list of all valid Socratic utterances at each assistant turn which are composed of the assistant response and the assistant alternatives.

        Example:
            data_reader = XMLDataReader("path_to_dataset_folder")
            dialogue_dataset = data_reader.export_to_jsonl()
            print(dialogue_dataset)
        
    """

    def export_to_jsonl(
        self,
        user_id="User",
        assistant_id="Assistant",
        export_path=None,
        configuration="gen_all",
        disable_permutation=False,
        steering_prompt="You are a tutor conversing with a student that always responds in the Socratic style. You *never* give the student the answer, but always try to ask just the right question to help them learn to think for themselves. You should always tune your question to the interest and knowledge of the student, breaking down the problem into simpler parts until it's at just the right level for them. Socratic utterances are utterances that guide the student and do not give them the solution directly nor make the root cause of the bug immediately obvious.",
    ):

        parsed_dialogues = self.get_parsed_dialogues()

        # create a string of jsonls
        jsonl_string = ""
        ids = []
        for parsed_dialogue_object in parsed_dialogues:
            dialogue = parsed_dialogue_object["dialogue"]
            context_dict = parsed_dialogue_object["context"]
            context_text = context_dict["context_text"]
            bug_id = context_dict["bug_id"]
            problem_id = context_dict["problem_id"]

            for i, turn in enumerate(dialogue):
                dialogue_context = "Conversation so far:\n"
                # Prepare context and dialogue context
                for j in range(i + 1):
                    prev_turn = dialogue[j]
                    dialogue_context += f"{user_id}: {prev_turn['user_text']}\n"
                    if prev_turn["code_text"] != "":
                        dialogue_context += (
                            f"<code>\n{prev_turn['code_text']}\n</code>\n"
                        )
                    if j < i:
                        dialogue_context += (
                            f"{assistant_id}: {prev_turn['assistant_text']}\n"
                        )

                instruction = ""
                if configuration == "gen_all":
                    dialogue_dict = {"messages": []}

                    system_message_dict = {"role": "system", "content": steering_prompt}

                    dialogue_dict["messages"].append(system_message_dict)

                    instruction = "Respond to the student with all possible distinct Socratic utterances that guide the student to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases."
                    # Add to dataset
                    lm_input_dict = {
                        "role": "user",
                        "content": context_text
                        + "\n"
                        + dialogue_context.strip()
                        + "\n"
                        + instruction,
                    }
                    lm_target_dict = {
                        "role": "assistant",
                        "content": "\n* ".join(
                            [
                                t
                                for t in ["* " + turn["assistant_text"]]
                                + turn["assistant_alternatives"]
                                if t != ""
                            ]
                        ),
                    }
                    dialogue_dict["messages"].append(lm_input_dict)
                    dialogue_dict["messages"].append(lm_target_dict)
                    ids.append(
                        {"problem_id": problem_id, "bug_id": bug_id, "dialogue_turn": i}
                    )
                    jsonl_string += json.dumps(dialogue_dict) + "\n"
                # configuration for gen_given_alts
                elif configuration == "gen_given_alts":

                    instruction_first = f"Respond to the student with an appropriate Socratic utterance that guides the student to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation."

                    instruction_follow_up = f"Respond with a Socratic utterance distinct from the utterances in the utterance bank above.\nYour response to the student should be an appropriate Socratic utterance that guides the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`, which describes the buggy code between `<bug_code>` and `</bug_code>`. A Socratic utterance is distinct if it refers to a separate part of the student implementation than all other utterances or if it refers to the same part of the student implementation as another utterance, but gives a significantly different amount of assistance to the student. A Socratic utterance is not distinct if it repeats the question from another utterance, if it rephrases another utterance, or if a student would respond to the question in the same way as another utterance. Respond with `None` if you cannot think of another clearly distinct Socratic utterance to guide the student."
                    # loop through assistant utterances for the current turn.
                    assistant_utterances = [
                        t
                        for t in [turn["assistant_text"]]
                        + turn["assistant_alternatives"]
                        if t != ""
                    ]

                    # create 5 permutations of the assistant utterances
                    # to encourage the model to generate the same output regardless of the order of the alternative utterances.
                    # if there are less than 5 utterances, then we do not need to create permutations
                    if len(assistant_utterances) < 5 or not disable_permutation:
                        permutations = [assistant_utterances]
                    else:
                        permutations = list(
                            itertools.permutations(
                                assistant_utterances, len(assistant_utterances)
                            )
                        )
                        # select 5 random permutations
                        permutations = list(random.sample(permutations, 5))

                    for permutation in permutations:
                        dialogue_dict = {"messages": []}

                        system_message_dict = {
                            "role": "system",
                            "content": steering_prompt,
                        }

                        dialogue_dict["messages"].append(system_message_dict)
                        # create a list of used assistant utterances
                        used_assistant_utterances = []
                        for assistant_utterance in list(permutation) + ["None"]:
                            if used_assistant_utterances == []:
                                lm_input_dict = {
                                    "role": "user",
                                    "content": context_text
                                    + "\n"
                                    + dialogue_context.strip()
                                    + "\n"
                                    + instruction_first,
                                }
                                lm_target_dict = {
                                    "role": "assistant",
                                    "content": assistant_utterance,
                                }
                                # add the lm input to the dialogue dict
                                dialogue_dict["messages"].append(lm_input_dict)
                                # add the lm target to the dialogue dict
                                dialogue_dict["messages"].append(lm_target_dict)
                                # add the assistant utterance to the list of used assistant utterances
                                used_assistant_utterances.append(assistant_utterance)
                            # if the assistant utterance has not been used before
                            elif assistant_utterance not in used_assistant_utterances:

                                # create the lm input
                                lm_input_dict = {
                                    "role": "user",
                                    "content": context_text
                                    + "\n"
                                    + dialogue_context.strip()
                                    + "\n"
                                    + "Utterance Bank:\n* "
                                    + "\n* ".join(used_assistant_utterances)
                                    + "\n"
                                    + instruction_follow_up,
                                }
                                # create the lm target
                                lm_target_dict = {
                                    "role": "assistant",
                                    "content": assistant_utterance,
                                }
                                # add the lm input to the dialogue dict
                                dialogue_dict["messages"].append(lm_input_dict)
                                # add the lm target to the dialogue dict
                                dialogue_dict["messages"].append(lm_target_dict)
                                # add the assistant utterance to the list of used assistant utterances
                                used_assistant_utterances.append(assistant_utterance)
                            else:
                                continue
                        ids.append(
                            {
                                "problem_id": problem_id,
                                "bug_id": bug_id,
                                "dialogue_turn": i,
                            }
                        )
                        # each permutation is a dialogue sample.
                        jsonl_string += json.dumps(dialogue_dict) + "\n"

                # configuration for indp_gen
                elif configuration == "indp_gen":
                    instruction = f"Respond to the student with an appropriate Socratic utterance that guides the student to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation."
                    # loop through assistant utterances for the current turn.
                    assistant_utterances = [
                        t
                        for t in [turn["assistant_text"]]
                        + turn["assistant_alternatives"]
                        if t != ""
                    ]
                    lm_input_dict = {
                        "role": "user",
                        "content": context_text
                        + "\n"
                        + dialogue_context.strip()
                        + "\n"
                        + instruction,
                    }

                    for assistant_utterance in assistant_utterances:
                        dialogue_dict = {"messages": []}

                        system_message_dict = {
                            "role": "system",
                            "content": steering_prompt,
                        }

                        dialogue_dict["messages"].append(system_message_dict)

                        lm_target_dict = {
                            "role": "assistant",
                            "content": assistant_utterance,
                        }
                        # add the lm input to the dialogue dict
                        dialogue_dict["messages"].append(lm_input_dict)
                        # add the lm target to the dialogue dict
                        dialogue_dict["messages"].append(lm_target_dict)
                        # each assistant utterance is a dialogue sample.
                        ids.append(
                            {
                                "problem_id": problem_id,
                                "bug_id": bug_id,
                                "dialogue_turn": i,
                            }
                        )
                        jsonl_string += json.dumps(dialogue_dict) + "\n"
                else:
                    raise ValueError(
                        "Invalid configuration. Valid configurations are: gen_all, gen_given_alts, indp_gen"
                    )

        jsonl_string = jsonl_string.strip()
        if export_path:
            with open(export_path, "w") as outfile:
                outfile.write(jsonl_string)
        # convert jsonl_string to a list of json objects
        jsonl_string = jsonl_string.split("\n")
        jsonl_objs = [json.loads(j) for j in jsonl_string]
        return jsonl_objs, ids

    def get_data_frame(self):
        # self.data is a list of dictionaries
        # convert it to a dictionary of lists to create a DataFrame
        df = {}
        for key in self.data[0].keys():
            df[key] = []
        for dic in self.data:
            for key in dic.keys():
                df[key].append(dic[key])
        return pd.DataFrame(df)

    def compute_dialogue_stats(self, dialogues_list):
        stats = {
            "assistant_main": [],
            "assistant_alt": [],
            "assistant_total": [],
            "user_main": [],
            "user_alt": [],
            "user_total": [],
        }

        for dialogue in dialogues_list:
            user_main_count = len(re.findall(r"User:", dialogue))
            assistant_main_count = len(re.findall(r"Assistant:", dialogue))

            user_alt_count = 0
            assistant_alt_count = 0
            is_user_alt = False
            for line in dialogue.split("\n"):
                if line.startswith("User:"):

                    is_user_alt = True
                elif line.startswith("Assistant:"):
                    is_user_alt = False
                if is_user_alt:
                    user_alt_count += len(re.findall(r"\t<alt>", line))
                else:
                    assistant_alt_count += len(re.findall(r"\t<alt>", line))

            stats["assistant_main"].append(assistant_main_count)
            stats["assistant_alt"].append(assistant_alt_count)
            stats["user_main"].append(user_main_count)
            stats["user_alt"].append(user_alt_count)
            stats["assistant_total"].append(assistant_main_count + assistant_alt_count)
            stats["user_total"].append(user_main_count + user_alt_count)

        descriptive_stats = {}
        for key, values in stats.items():
            descriptive_stats[key] = {
                "total": sum(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            }
        # if sum(stats['assistant_main']) + sum(stats['user_main']) / 2 is a whole number:
        if sum(stats["assistant_main"]) + sum(stats["user_main"]) % 2 == 0:
            total_turns = sum(stats["assistant_main"]) + sum(stats["user_main"]) / 2
        else:
            # there is an extra utterance in the dialogue
            total_turns = min(sum(stats["assistant_main"]), sum(stats["user_main"]))
        return {
            "total_turns": total_turns,
            "total_utterances": sum(stats["assistant_main"])
            + sum(stats["user_main"])
            + sum(stats["assistant_alt"])
            + sum(stats["user_alt"]),
            "descriptive_stats": descriptive_stats,
        }

    # When parsing dialogues if the code state is right after an <alt> tag, then it belong to the alt utterance not the main thread.

    """
    Function: parse_dialogue
    Parses the dialogue text into a list of utterances and code states (if any) and returns a list of dictionaries with the following keys:
        * user_text: the text of the user utterance.
        * code_text: the text of the code state. 
        * cumulative_code_text: the text of the code state. If the utterance is not a code state, then the text will be the cumulative code state if any.
        * assistant_text: the text of the assistant utterance.
        * user_alternatives: a list of dictionaries with the following keys:
            - alt_text: the text of the alternative utterance.
            - alt_code: the text of the alternative code state. If the alternative utterance is not associated with a code state, then the text will be empty.
        * assistant_alternatives: a list of alternative utterances for the assistant utterance.
    Args:
        dialogue_text: the text of the dialogue. In the format of the dialogues in the dataset.
        e.g. 
                User: I am trying to run the code but it is not working.
                    <alt> Nothing is working.
                Assistant: What is the error message?
                    <alt> Any error message?
                    <alt> What does the terminal say?
                User: I figured it out. I was missing a comma.
                <code>
                import pandas as pd
                df = pd.read_csv('data.csv' , sep=',')
                </code>
                    <alt> Ah! I think it is because the file name is wrong.
                <code>
                import pandas as pd
                df = pd.read_csv('my_data.csv' , sep=',')
                </code>
                Assistant: Great! It is working now.
                    <alt> Awesome! Good job!
                User: Thank you for your help.
                    <alt> Thanks!
            
    Returns:
       [
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
                'assistant_alternatives': ['Any error message?', 'What does the terminal say?'],
                
            },
            {
                'user_text': 'I figured it out. I was missing a comma.',
                'code_text': 'import pandas as pd\ndf = pd.read_csv(\'data.csv\' , sep=\',\')',
                'cumulative_code_text': 'import pandas as pd\ndf = pd.read_csv(\'data.csv\' , sep=\',\')',
                'assistant_text': 'Great! It is working now.',
                'user_alternatives': {
                    'alt_text': 'Ah! I think it is because the file name is wrong.',
                    'alt_code': 'import pandas as pd\ndf = pd.read_csv(\'my_data.csv\' , sep=\',\')'
                }
                'assistant_alternatives': ['Awesome! Good job!'],
            }
            {
                'user_text': 'Thank you for your help.',
                'code_text': '',
                'cumulative_code_text': 'import pandas as pd\ndf = pd.read_csv(\'data.csv\' , sep=\',\')',
                'assistant_text': '',
                'user_alternatives': {
                    'alt_text': 'Thanks!',
                    'alt_code': '',
                }
                'assistant_alternatives': [],
            }

       ]
    """

    def _parse_dialogue(self, dialogue_text):
        dialogue_data = []
        code_text = ""
        user_text = ""
        assistant_text = ""
        user_alternatives = []
        assistant_alternatives = []
        skip_next_code_block = False
        active_utternace = "user"
        lines = dialogue_text.split("\n")
        # strip lines where the first word is 'User
        for l in lines:
            if (
                l.strip().startswith("User:")
                or l.strip().startswith("Assistant:")
                or l.strip().startswith("<alt>")
                or l.strip().startswith("<code>")
                or l.strip().startswith("</code>")
            ):
                # replace the line with the stripped line
                lines[lines.index(l)] = l.strip()

        for i, line in enumerate(lines):
            if line.startswith("User:"):
                user_text = line[6:].strip()
                active_utternace = "user"
            elif line.startswith("Assistant:"):
                assistant_text = line[11:].strip()
                active_utternace = "assistant"
            elif line.startswith("<code>"):

                if skip_next_code_block:
                    skip_next_code_block = False
                    continue
                code_text = ""
                # check if the code state is associated with an alternative utterance
                while i + 1 < len(lines) and not lines[i + 1].startswith("</code>"):
                    i += 1
                    code_text += "\n" + lines[i]
                code_text = code_text.strip()
            elif line.startswith("<alt>"):
                alt_text = line.strip()[5:].strip()
                if i + 1 < len(lines) and lines[i + 1].startswith("<code>"):
                    i += 1
                    alt_code_text = ""
                    while i + 1 < len(lines) and not lines[i + 1].startswith("</code>"):
                        i += 1
                        alt_code_text += "\n" + lines[i]

                    alt_code_text = alt_code_text.strip()
                    skip_next_code_block = True
                else:
                    alt_code_text = ""
                if active_utternace == "user":
                    user_alternatives.append(
                        {"alt_text": alt_text, "alt_code": alt_code_text}
                    )
                elif active_utternace == "assistant":
                    assistant_alternatives.append(alt_text)

            if user_text and (assistant_text or i == len(lines) - 1):
                if (
                    i < len(lines) - 1
                    and (
                        lines[i + 1].startswith("Assistant:")
                        or lines[i + 1].startswith("User:")
                    )
                    or i == len(lines) - 1
                ):

                    dialogue_data.append(
                        {
                            "user_text": user_text,
                            "code_text": code_text,
                            "cumulative_code_text": (
                                code_text
                                if code_text != ""
                                else (
                                    dialogue_data[-1]["cumulative_code_text"]
                                    if len(dialogue_data) > 0
                                    else ""
                                )
                            ),
                            "assistant_text": assistant_text,
                            "user_alternatives": user_alternatives,
                            "assistant_alternatives": assistant_alternatives,
                        }
                    )

                    user_text = ""
                    assistant_text = ""
                    user_alternatives = []
                    assistant_alternatives = []
                    code_text = ""

        return dialogue_data
