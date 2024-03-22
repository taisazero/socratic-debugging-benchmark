from metrics.metric_computer import MetricComputer
from data_utils.data_reader import XMLDataReader
from inference.gpt_inference import ChatGPTModel
from data_utils.data_reader import extract_tag_content
import argparse
from tqdm import tqdm
import os 
import regex as re
from trainers.finetuner import Seq2SeqFineTuner
from utils.huggingface_path_handler import get_huggingface_path, get_huggingface_dataset_path



def debug(dataset_path, test_data_path):
    # load data
    train_data_reader = XMLDataReader(dataset_path)
    test_data_reader = XMLDataReader(test_data_path)
    train_data = train_data_reader.export_to_hf_dataset_t5()
    test_data = test_data_reader.export_to_hf_dataset_t5()

    print(test_data)
    print(test_data["dialogue_context"])
    print(test_data["assistant_response"])
    print(test_data["assistant_alternatives"])
    print(test_data["context"])
def finetune(dataset_path, test_data_path, evaluation_method, model_name, batch_size, epochs, learning_rate):
    # load data
    train_data_reader = XMLDataReader(dataset_path)
    test_data_reader = XMLDataReader(test_data_path)
    train_dataset = train_data_reader.export_to_hf_dataset_t5(instruction='')
    val_dataset = test_data_reader.export_to_hf_dataset_t5(instruction='')
    output_dir = f"trained_models/{model_name.split('/')[-1]}_finetuned"
    # make it 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_name = get_huggingface_path(model_name)
    fine_tuner = Seq2SeqFineTuner(model_name, train_dataset, val_dataset, export_path=f"trained_models/{model_name.split('/')[-1]}_finetuned/results.xlsx")
    fine_tuner.fine_tune(output_dir, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, input_field="lm_input", target_field="lm_target")
    # make a text file that documents the hyperparameters used
    with open(f"{model_name.split('/')[-1]}_finetuned/hyperparameters.txt", 'w') as f:
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"evaluation_method: {evaluation_method}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run debug mode")
    parser.add_argument("--dataset_path", type=str, default= "socratic_benchmark_v2/train", help="Path to the dataset to evaluate. Defaults to 'socratic_benchmark_v2/train'.")
    parser.add_argument("--testset_path", type=str, default= "socratic_benchmark_v2/testset", help="Path to the testset to evaluate. Defaults to 'socratic_benchmark_v2/testset'.")
    # add argument for evaluation method. Either 'overall' or 'thoroughness'
    parser.add_argument("--evaluation_method", type=str, default= "thoroughness", help="Evaluation method. Either 'overall' or 'thoroughness'. Defaults to 'thoroughness'. Thoroughness uses the bipartite matching algorithm to find the best match between each reference and the response. Overall uses scoring of the best matching (reference, pred) pair.")
    parser.add_argument("--model_name", type=str, default= "google/flan-t5-small", help="Model name to finetune. Defaults to 'google/flan-t5-small'.")
    parser.add_argument("--batch_size", type=int, default= 2, help="Batch size for finetuning. Defaults to 2.")
    parser.add_argument("--epochs", type=int, default= 10, help="Number of epochs for finetuning. Defaults to 10.")
    parser.add_argument("--learning_rate", type=float, default= 1e-5, help="Learning rate for finetuning. Defaults to 1e-5.")

    args = parser.parse_args()
    if args.debug:
        debug(args.dataset_path, args.testset_path)
    else:
        finetune(args.dataset_path, args.testset_path, args.evaluation_method, args.model_name, args.batch_size, args.epochs, args.learning_rate)
        # debug()
        

# run with:
# python run_finetuning.py --model_name google/flan-t5-small --batch_size 2 --epochs 20 --learning_rate 1e-5
# python run_finetuning.py --model_name google/flan--base --batch_size 2 --epochs 20 --learning_rate 1e-5
# python run_finetuning.py --model_name google/flan-t5-xl --batch_size 2 --epochs 20 --learning_rate 1e-5
