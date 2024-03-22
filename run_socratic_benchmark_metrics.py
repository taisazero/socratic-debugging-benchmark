from metrics.metric_computer import MetricComputer
from data_utils.data_reader import XMLDataReader
from inference.gpt_inference import ChatGPTModel
from data_utils.data_reader import extract_tag_content
import argparse
from tqdm import tqdm
import os
import regex as re


def debug():
    # metric = MetricComputer(export_to_excel=True, export_path="results.xlsx")

    # predictions = ["hello there, I am badeed", "general kenobi kappacino frank"]
    # references = [
    #     ["hello there, I am badeed", "hi here! you like cats?"],
    #     ["general kenobi kappacino frank", "general grievous badeed neice"]
    # ]

    # scores = metric.compute_overall_score(predictions, references)
    # print(scores)
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
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "reviewing_dialogues")
    path = os.path.join(path, "final_dataset")
    data_reader = XMLDataReader(path)
    print(data_reader._parse_dialogue(dialogue_text))


def main(use_chat_prompt, generation_mode, num_responses, dataset_path, eval_method):
    path = os.path.dirname(os.path.abspath(__file__))
    # split dataset_path by / and replace with os.path.join
    dataset_path = os.path.join(*dataset_path.split("/"))
    path = os.path.join(path, dataset_path)
    data_reader = XMLDataReader(path)
    if generation_mode == "sample" and num_responses > 1:
        n = num_responses
    else:
        n = 1

    parsed_dialogues = data_reader.get_parsed_dialogues()
    api_key = open(".streamlit/oai_key.txt", "r").read().strip()
    steering_prompt = ""
    if generation_mode != "sample":
        generation_args = {
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_p": 0.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "n": n,  # number of responses to return,
            "stream": False,
        }
    else:
        generation_args = {
            "max_tokens": 1024,
            "temperature": 0.75,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "n": n,  # number of responses to return,
            "stream": False,
        }

    if generation_mode == "multiple":

        steering_prompt = "You are a tutor that always responds in the Socratic style. You *never* give the student the answer, but always try to ask just the right question to help them learn to think for themselves. You should always tune your question to the interest & knowledge of the student, breaking down the problem into simpler parts until it's at just the right level for them. Socratic utterances are utterances that guide the user and do not give them the solution directly. In each of your responses, provide a comprehensive list of Socratic responses that you can give to the user to help them solve the problem on their own, based on the conversation so far."

    elif generation_mode == "cot":
        steering_prompt = "You are a reflective and experienced tutor. You always introspect and think about all the reasons causing the user to make their mistake. When asked to respond to the user you always respond in the Socratic style. You *never* give the student the answer, but always try to ask just the right question to help them learn to think for themselves. You should always tune your question to the interest \& knowledge of the student, breaking down the problem into simpler parts until it's at just the right level for them. Socratic utterances guide the user and do not give them the solution directly. You are as comprehensive as possible when listing reasons. You are also as comprehensive as possible when listing Socratic utterances guiding the user. Your responses should be in-line with the instruction you are given."
    else:
        steering_prompt = "You are a tutor that always responds in the Socratic style. You *never* give the student the answer, but always try to ask just the right question to help them learn to think for themselves. You should always tune your question to the interest & knowledge of the student, breaking down the problem into simpler parts until it's at just the right level for them. Socratic utterances are utterances that guide the user and do not give them the solution directly."

    chatgpt = ChatGPTModel(
        api_key=api_key,
        generation_args=generation_args,
        steering_prompt=steering_prompt,
    )
    gpt4 = ChatGPTModel(
        model_name="gpt-4",
        api_key=api_key,
        generation_args=generation_args,
        steering_prompt=steering_prompt,
    )

    # make sure the results folder exists
    if not os.path.exists("results/gpt4"):
        os.makedirs("results/gpt4")
    if not os.path.exists("results/chatgpt"):
        os.makedirs("results/chatgpt")

    if use_chat_prompt:
        chatgpt_path = (
            f"results/chatgpt/chat_prompt_{generation_mode}_{n}_responses.xlsx"
        )
        gpt4_path = f"results/gpt4/chat_prompt_{generation_mode}_{n}_responses.xlsx"

    else:
        if generation_mode == "multiple" or generation_mode == "cot":
            chatgpt_path = f"results/chatgpt/comprehensive_prompt_{generation_mode}_{n}_responses.xlsx"
            gpt4_path = f"results/gpt4/comprehensive_prompt_{generation_mode}_{n}_responses.xlsx"
        else:
            chatgpt_path = f"results/chatgpt/instruction_prompt_{generation_mode}_{n}_responses.xlsx"
            gpt4_path = (
                f"results/gpt4/instruction_prompt_{generation_mode}_{n}_responses.xlsx"
            )

    chatgpt_metrics = MetricComputer(export_to_excel=True, export_path=chatgpt_path)
    gpt4_metrics = MetricComputer(export_to_excel=True, export_path=gpt4_path)
    chatgpt_responses = []
    gpt4_responses = []
    dataset_references = []
    prompts = []
    for parsed_dialogue_object in tqdm(parsed_dialogues, desc="Processing dialogues"):
        dialogue = parsed_dialogue_object["dialogue"]
        context_dict = parsed_dialogue_object["context"]
        context_text = context_dict["context_text"]

        if generation_mode == "multiple":

            instruction = "Respond to the user with all possible distinct Socratic utterances that guide the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases.\n1."

        elif generation_mode == "cot":

            instruction = {
                "first_introspection": "After looking at the buggy code, the bug description, and the bug fix, what are all the possible reasons or misconceptions that led the user to make this mistake? Do NOT list Socratic questions.",
                "introspection": 'Given the dialogue so far, what are all the possible reasons or misconceptions if any that the user still has that impede them from fixing the bug? Do NOT list Socratic questions. If the bug is already fixed, say "There are no remaining misconceptions or reasons that impede the user from fixing the bug, as they have already identified and corrected their code."',
                "respond": "Utilizing the dialogue and reasons or misconceptions so far, respond to the user with all possible distinct Socratic utterances that guide the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases.\n1.",
            }

        else:
            instruction = "Respond to the user with a Socratic utterance that guides the user to discover and fix the bug described between `<bug_desc>` and `</bug_desc>`. Student code is written between `<code>` and `</code>` throughout the conversation. Assume that the student has run the test cases."

        for i, turn in enumerate(dialogue):
            prompt = context_text + "\n\n"
            # dialogue is composed of a list of dictionaries, each dictionary is a turn
            # each turn has the following keys:
            # user_text: the text the user said
            # code_text: the code if any the user has written in this turn
            # cumulative_code_text: the most current code if any the user has written up to this point
            # assistant_text: the text the assistant said
            # user_alternatives: [{alt_text: the alternatives the user could have said, alt_code: any code the user could have written}]
            # assistant_alternatives: the alternatives the assistant could have said.

            # break if the turn has no assistant text
            if i < len(dialogue) and dialogue[i]["assistant_text"] == "":
                break

            if use_chat_prompt:
                if generation_mode == "multiple":
                    raise ValueError(
                        "Multiple generation mode is not supported with chat prompt"
                    )
                elif generation_mode == "cot":
                    raise ValueError(
                        "COT generation mode is not supported with chat prompt"
                    )
                else:
                    # Valid chat prompt generation mode
                    # loop through all previous turns up to this point and add them to a messages list of tuples containing (speaker, text).
                    # then use `generate_turn(messages, user_identifier='User', system_identifier='Assistant')'`
                    messages = []
                    # instantiate messages with the context text and the first turn
                    messages.append(
                        ("User", context_text + "\n\n" + dialogue[0]["user_text"])
                    )

                    for j in range(1, i):
                        prev_turn = dialogue[j]

                        if prev_turn["code_text"] != "":
                            messages.append(
                                (
                                    "User",
                                    prev_turn["user_text"]
                                    + "\n<code>\n"
                                    + prev_turn["code_text"]
                                    + "\n</code>\n",
                                )
                            )
                        else:
                            messages.append(("User", prev_turn["code_text"]))
                        messages.append(("Assistant", prev_turn["assistant_text"]))
                        if prev_turn["code_text"] != "":
                            messages.append(("Assistant", prev_turn["code_text"]))

                        prompt += f"User: {prev_turn['user_text']}\n"
                        prompt += f"Assistant: {prev_turn['assistant_text']}\n"
                        if prev_turn["code_text"] != "":
                            prompt += f"<code>\n{prev_turn['code_text']}\n</code>\n"

                    # add the current turn to the messages list
                    if turn["code_text"] != "" and i > 0:
                        messages.append(
                            (
                                "User",
                                turn["user_text"]
                                + "\n<code>\n"
                                + turn["code_text"]
                                + "\n</code>\n",
                            )
                        )
                    elif i > 0:
                        messages.append(("User", turn["user_text"]))

                    # finish the prompt with the current turn
                    prompt += f"User: {turn['user_text']}\n"
                    if turn["code_text"] != "":
                        prompt += f"<code>\n{turn['code_text']}\n</code>\n"

                    chatgpt_response = chatgpt.generate_turn(
                        messages, user_identifier="User", system_identifier="Assistant"
                    )
                    gpt4_response = gpt4.generate_turn(
                        messages, user_identifier="User", system_identifier="Assistant"
                    )
                    prompts.append(prompt)

            else:
                # if in instruction prompt mode, add the instruction to the prompt
                # loop through all previous turns up to this point and add them to the prompt
                if i == 0 and generation_mode == "cot":
                    introspecton_prompt = (
                        prompt + f"{instruction['first_introspection']}\n"
                    )
                    # generate responses
                    introspection_chatgpt = chatgpt.generate(introspecton_prompt)
                    introspection_gpt4 = gpt4.generate(introspecton_prompt)

                for j in range(i):
                    if generation_mode == "cot" and j == 0:
                        prompt += "\nDialogue:\n"

                    prev_turn = dialogue[j]
                    prompt += f"User: {prev_turn['user_text']}\n"
                    prompt += f"Assistant: {prev_turn['assistant_text']}\n"
                    if prev_turn["code_text"] != "":
                        prompt += f"<code>\n{prev_turn['code_text']}\n</code>\n"

                prompt += f"User: {turn['user_text']}\n"
                if turn["code_text"] != "":
                    prompt += f"<code>\n{turn['code_text']}\n</code>\n"

                if generation_mode == "cot":
                    if 1 > i and generation_mode == "cot":
                        introspecton_prompt = (
                            prompt + f"{instruction['introspection']}\n"
                        )
                        # generate responses
                        introspection_chatgpt = chatgpt.generate(introspecton_prompt)
                        introspection_gpt4 = gpt4.generate(introspecton_prompt)

                    gpt4_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n{introspection_gpt4}\n{instruction['respond']}"
                    )
                    chatgpt_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n{introspection_chatgpt}\n{instruction['respond']}"
                    )
                    save_prompt = (
                        prompt
                        + f"\nReasons or Misconceptions so far:\n\tchatgpt:\n{introspection_chatgpt}\n\tgpt4:\n{introspection_gpt4}\n{instruction['respond']}"
                    )
                    # generate responses
                    chatgpt_response = chatgpt.generate(chatgpt_prompt)
                    gpt4_response = gpt4.generate(gpt4_prompt)
                    prompts.append(save_prompt)

                else:
                    prompt += f"\n{instruction}"

                    # generate responses
                    chatgpt_response = chatgpt.generate(prompt)
                    gpt4_response = gpt4.generate(prompt)
                    prompts.append(prompt)

            references = [turn["assistant_text"]] + turn["assistant_alternatives"]
            dataset_references.append(references)

            if generation_mode == "multiple" or generation_mode == "cot":
                # parse the itemized list string response into a list of responses
                if not chatgpt_response.startswith(
                    "1."
                ) and not chatgpt_response.startswith("1. "):
                    chatgpt_response = (
                        "1." + chatgpt_response
                        if chatgpt_response.startswith(" ")
                        else "1. " + chatgpt_response
                    )
                if not gpt4_response.startswith("1.") and not gpt4_response.startswith(
                    "1. "
                ):
                    gpt4_response = (
                        "1." + gpt4_response
                        if gpt4_response.startswith(" ")
                        else "1. " + gpt4_response
                    )
                # clean responses from any code blocks
                if "<code>" in chatgpt_response:
                    remove_code = extract_tag_content(chatgpt_response, "code")
                    chatgpt_response = chatgpt_response.replace(
                        f"<code>\n{remove_code}\n</code>", ""
                    )
                if "<code>" in gpt4_response:
                    remove_code = extract_tag_content(gpt4_response, "code")
                    gpt4_response = gpt4_response.replace(
                        f"<code>\n{remove_code}\n</code>", ""
                    )
                # create a list of responses from the itemized list string response while removing the item numbers using regex
                chatgpt_response = re.findall(r"\d+\.\s*(.*)", chatgpt_response)
                gpt4_response = re.findall(r"\d+\.\s*(.*)", gpt4_response)

            # Note: if generation mode is multiple or sample with n > 1, then we have multiple responses per sample.
            chatgpt_responses.append(chatgpt_response)
            gpt4_responses.append(gpt4_response)

    # compute metrics for all generation modes.
    if eval_method != "thoroughness":
        chat_gpt_scores = chatgpt_metrics.compute(
            chatgpt_responses, dataset_references, contexts=prompts
        )
        gpt4_scores = gpt4_metrics.compute(
            gpt4_responses, dataset_references, contexts=prompts
        )
    else:
        chat_gpt_scores = chatgpt_metrics.compute_thoroughness(
            chatgpt_responses, dataset_references, contexts=prompts
        )
        gpt4_scores = gpt4_metrics.compute_thoroughness(
            gpt4_responses, dataset_references, contexts=prompts
        )
    print("ChatGPT scores:")
    print(chat_gpt_scores)
    print("GPT-4 scores:")
    print(gpt4_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run debug mode")
    parser.add_argument(
        "--do_chat_prompt",
        action="store_true",
        help="Run chat prompt mode where the prompt is the chat history between the user and chatgpt. Defaults to an instructional prompt 'Respond to the user with a Socratic utterance that guides the user to discover and fix the bug described in `<bug_desc>`. Student code is denoted by `<code>` throughout the conversation. Utterances that have the same meaning but different words are considered duplicates. Assume that the student has run the test cases.'",
    )
    # Add argument for generation mode. Either 'single', 'multiple', 'cot', or 'sample'
    parser.add_argument(
        "--generation_mode",
        type=str,
        default="single",
        help="Generate a single response or multiple responses using instruction or sample k responses. Options are 'single', 'multiple', 'cot', or 'sample'. Defaults to 'single'. 'multiple' is incompatible with '--do_chat_prompt'. If 'sample', the number of responses to sample is specified by the --num_responses argument.",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Number of responses to sample if --generation_mode is 'sample'. Defaults to 5.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="reviewing_dialogues/eval_full_v2",
        help="Path to the dataset to evaluate. Defaults to 'reviewing_dialogues/eval_full_v2'.",
    )
    # add argument for evaluation method. Either 'overall' or 'thoroughness'
    parser.add_argument(
        "--evaluation_method",
        type=str,
        default="thoroughness",
        help="Evaluation method. Either 'overall' or 'thoroughness'. Defaults to 'thoroughness'. Thoroughness uses the bipartite matching algorithm to find the best match between each reference and the response. Overall uses scoring of the best matching (reference, pred) pair.",
    )
    args = parser.parse_args()
    if args.debug:
        debug()
    else:
        main(
            args.do_chat_prompt,
            args.generation_mode,
            args.num_responses,
            args.dataset_path,
            args.evaluation_method,
        )
        # debug()


# run with:
# instruction mode with a single response
# python run_socratic_benchmark_metrics.py (done on dekstop)
# single response without instruction mode
# python run_socratic_benchmark_metrics.py --do_chat_prompt (ready to test)
# multiple responses (all possible socratic utterances) with instruction mode
# python run_socratic_benchmark_metrics.py --generation_mode multiple (ready to test)
# sample 5 responses with instruction mode
# python run_socratic_benchmark_metrics.py --generation_mode sample --num_responses 5 (ready to test)
# sample 10 responses with instruction mode
# python run_socratic_benchmark_metrics.py --generation_mode sample --num_responses 10
# sample 5 responses without instruction mode
# python run_socratic_benchmark_metrics.py --do_chat_prompt --generation_mode sample --num_responses 5
# sample 10 responses without instruction mode
# python run_socratic_benchmark_metrics.py --do_chat_prompt --generation_mode sample --num_responses 10
