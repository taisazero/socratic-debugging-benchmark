from metrics.metric_computer import MetricComputer
from data_utils.data_reader import XMLDataReader
from inference.gpt_inference import ChatGPTModel
from data_utils.data_reader import extract_tag_content
import argparse
from tqdm import tqdm
import os 
import regex as re
import ast
import pandas as pd

def main (file_path_dict, save_path_dict):
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'reviewing_dialogues')
    path = os.path.join(path, 'final_dataset')
    data_reader = XMLDataReader(path)
    chatgpt_path = file_path_dict['chatgpt']
    gpt4_path = file_path_dict['gpt4']
    chatgpt_save_path = save_path_dict['chatgpt']
    gpt4_save_path = save_path_dict['gpt4']
    

    # open the excel file from file_path
    # the columns are: context	prediction	reference	reference_list	prediction_list	metric	score
    # we want to get the context, prediction_list, reference_list.
    for file_path, save_path in zip([chatgpt_path, gpt4_path], [chatgpt_save_path, gpt4_save_path]):
        # create a metric computer
        metric_computer = MetricComputer(export_to_excel=True, export_path=save_path)
        # read the excel file
        df = pd.read_excel(file_path)
        # get the columns we want
        df = df[['context', 'prediction_list', 'reference_list']]
        # convert the prediction_list and reference_list to list
        df['prediction_list'] = df['prediction_list'].apply(lambda x: ast.literal_eval(x))
        df['reference_list'] = df['reference_list'].apply(lambda x: ast.literal_eval(x))
        print('Computing thoroughness for ', file_path)
        scores = metric_computer.compute_thoroughness(df['prediction_list'].to_list(), df['reference_list'].to_list(), df['context'].to_list())

        print(f'{file_path} Scores:')
        print('--------------------------------')
        # print a pretty dictionary of f1, precision, recall only
        for key, value in scores.items():
            if key == 'true_positives' or key == 'false_positives' or key == 'false_negatives':
                continue
            print(f'{key}: {value}')
            print('--------------------------------')

        print(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatgpt_file_path", type=str, default="results/chatgpt/comprehensive_prompt_multiple_1_responses.xlsx")
    parser.add_argument("--gpt4_file_path", type=str, default="results/gpt4/comprehensive_prompt_multiple_1_responses.xlsx")
    parser.add_argument("--chatgpt_save_path", type=str, default="results/chatgpt/comprehensive_prompt_multiple_1_responses_thoroughness.xlsx")
    parser.add_argument("--gpt4_save_path", type=str, default="results/gpt4/comprehensive_prompt_multiple_1_responses_thoroughness.xlsx")
    args = parser.parse_args()

    file_path_dict = {'chatgpt': args.chatgpt_file_path, 'gpt4': args.gpt4_file_path}
    save_path_dict = {'chatgpt': args.chatgpt_save_path, 'gpt4': args.gpt4_save_path}
    main(file_path_dict, save_path_dict)


# # Path: run_thoroughness_metrics.py
# python run_thoroughness_metrics.py --file_path results/chatgpt/comprehensive_prompt_multiple_1_responses.xlsx --save_path results/chatgpt/comprehensive_prompt_multiple_1_responses_thoroughness.xlsx
# python run_thoroughness_metrics.py --file_path results/gpt4/comprehensive_prompt_multiple_1_responses.xlsx --save_path results/gpt4/comprehensive_prompt_multiple_1_responses_thoroughness.xlsx
