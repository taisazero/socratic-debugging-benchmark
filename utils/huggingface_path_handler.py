import os

def get_huggingface_path(model_name):
    path = os.environ['HF_MODELS']+f'/{model_name}' if 'HF_MODELS' in  os.environ.keys() else model_name
    return path

def get_huggingface_dataset_path(dataset_name):
    path = os.environ['HF_DATASETS']+f'/{dataset_name}' if 'HF_DATASETS' in  os.environ.keys() else dataset_name
    return path