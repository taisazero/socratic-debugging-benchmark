
# unit tests
import pytest
from inference.gpt_inference import ChatGPTModel
from inference.gpt_inference import GPT3Model

def test_chatgpt_generation():
    generation_args = {
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "n": 1, # number of responses to return,
        "stream": False,
    }
    oai_key = open('.streamlit/oai_key.txt', 'r').read()
    model = ChatGPTModel(generation_args=generation_args, api_key=oai_key)
    prompt = "Hello, how are you?"
    response = model.generate(prompt)
    assert response is not None
    assert len(response) > 0
    print(response)

def test_gpt4_generation():
    generation_args = {
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 0.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "n": 1, # number of responses to return,
        "stream": False,
    }
    oai_key = open('.streamlit/oai_key.txt', 'r').read()
    model = ChatGPTModel(model_name = 'gpt-4', generation_args=generation_args, api_key=oai_key)
    prompt = "Hello, how are you?"
    response = model.generate(prompt)
    assert response is not None
    assert len(response) > 0
    print(response)

if __name__ == "__main__":
    test_chatgpt_generation()
    test_gpt4_generation()