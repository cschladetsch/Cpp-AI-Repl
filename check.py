import sys
import os
import pkg_resources
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket

# Define cache directories
BASE_CACHE_DIR = os.path.expanduser('~/.cache')
CPP_ANALYZER_CACHE_DIR = os.path.join(BASE_CACHE_DIR, 'cpp_analyzer')
MODEL_CACHE_DIR = os.path.join(CPP_ANALYZER_CACHE_DIR, 'models')
HUGGINGFACE_CACHE_DIR = os.path.join(BASE_CACHE_DIR, 'huggingface')

# Create cache directories
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(HUGGINGFACE_CACHE_DIR, exist_ok=True)

# Set environment variables for caching
os.environ['TRANSFORMERS_CACHE'] = HUGGINGFACE_CACHE_DIR
os.environ['HF_HOME'] = HUGGINGFACE_CACHE_DIR

def check_internet_connection():
    try:
        # Attempt to resolve a well-known domain
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        pass
    return False

def check_environment():
    print(f"Python version: {sys.version}")
    
    packages = ['transformers', 'torch']
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"{package} version: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"{package} is not installed")
    
    print(f"\nCache Directories:")
    print(f"BASE_CACHE_DIR: {BASE_CACHE_DIR}")
    print(f"CPP_ANALYZER_CACHE_DIR: {CPP_ANALYZER_CACHE_DIR}")
    print(f"MODEL_CACHE_DIR: {MODEL_CACHE_DIR}")
    print(f"HUGGINGFACE_CACHE_DIR: {HUGGINGFACE_CACHE_DIR}")

    internet_connection = check_internet_connection()
    print(f"\nInternet Connection: {'Available' if internet_connection else 'Not Available'}")

def test_model_loading(model_name):
    print(f"\nAttempting to load model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, cache_dir=MODEL_CACHE_DIR)
        print(f"Successfully loaded {model_name}")
        
        # Test tokenizer and model
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=20)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model response to '{input_text}':")
        print(response)
    except Exception as e:
        print(f"Failed to load {model_name}: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    print("Environment Information:")
    check_environment()
    
    if check_internet_connection():
        model_to_test = "microsoft/phi-3.5-mini-instruct"
        test_model_loading(model_to_test)
    else:
        print("\nUnable to load model due to no internet connection. Please check your network and try again.")
