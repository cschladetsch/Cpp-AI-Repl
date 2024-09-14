import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, Style
import warnings
import pickle

class PhiModelHandler:
    AVAILABLE_MODELS = {
        # Hugging Face format models
        "3.5-mini": "microsoft/Phi-3.5-mini-instruct",
        "3.5-moe": "microsoft/Phi-3.5-MoE-instruct",
        "3.5-vision": "microsoft/Phi-3.5-vision-instruct",
        "3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
        "3-mini-128k": "microsoft/Phi-3-mini-128k-instruct",
        "3-small-8k": "microsoft/Phi-3-small-8k-instruct",
        "3-small-128k": "microsoft/Phi-3-small-128k-instruct",
        "3-medium-4k": "microsoft/Phi-3-medium-4k-instruct",
        "3-medium-128k": "microsoft/Phi-3-medium-128k-instruct",
        "3-vision-128k": "microsoft/Phi-3-vision-128k-instruct",
        
        # ONNX format models
        "3.5-mini-onnx": "microsoft/Phi-3.5-mini-instruct-onnx",
        "3-mini-4k-onnx": "microsoft/Phi-3-mini-4k-instruct-onnx",
        "3-mini-4k-onnx-web": "microsoft/Phi-3-mini-4k-instruct-onnx-web",
        "3-mini-128k-onnx": "microsoft/Phi-3-mini-128k-instruct-onnx",
        "3-small-8k-onnx-cuda": "microsoft/Phi-3-small-8k-instruct-onnx-cuda",
        "3-small-128k-onnx-cuda": "microsoft/Phi-3-small-128k-instruct-onnx-cuda",
        "3-medium-4k-onnx-cpu": "microsoft/Phi-3-medium-4k-instruct-onnx-cpu",
        "3-medium-4k-onnx-cuda": "microsoft/Phi-3-medium-4k-instruct-onnx-cuda",
        "3-medium-4k-onnx-directml": "microsoft/Phi-3-medium-4k-instruct-onnx-directml",
        "3-medium-128k-onnx-cpu": "microsoft/Phi-3-medium-128k-instruct-onnx-cpu",
        "3-medium-128k-onnx-cuda": "microsoft/Phi-3-medium-128k-instruct-onnx-cuda",
        "3-medium-128k-onnx-directml": "microsoft/Phi-3-medium-128k-instruct-onnx-directml",
        "3-vision-128k-onnx-cpu": "microsoft/Phi-3-vision-128k-instruct-onnx-cpu",
        "3-vision-128k-onnx-cuda": "microsoft/Phi-3-vision-128k-instruct-onnx-cuda",
        "3-vision-128k-onnx-directml": "microsoft/Phi-3-vision-128k-instruct-onnx-directml",
        
        # GGUF format model
        "3-mini-4k-gguf": "microsoft/Phi-3-mini-4k-instruct-gguf",
    }

    def __init__(self, model_key="3-medium-4k"):
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model key. Choose from: {', '.join(self.AVAILABLE_MODELS.keys())}")
        
        self.model_key = model_key
        self.model_name = self.AVAILABLE_MODELS[model_key]
        self.cache_dir = os.path.expanduser('~/.cache/cpp_analyzer')
        self.model_cache_dir = os.path.join(self.cache_dir, 'models')
        self.model_cache_file = os.path.join(self.cache_dir, f'phi_{model_key}_model.pkl')
        self.tokenizer_cache_file = os.path.join(self.cache_dir, f'phi_{model_key}_tokenizer.pkl')
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self._load_from_cache():
            print(f"{Fore.GREEN}Phi model '{self.model_key}' loaded from cache.{Style.RESET_ALL}")
            return True

        print(f"{Fore.CYAN}Loading Phi model '{self.model_key}'...{Style.RESET_ALL}")
        try:
            if "onnx" in self.model_key:
                # ONNX model loading logic here
                # You may need to use a different library for ONNX models
                raise NotImplementedError("ONNX model loading not implemented yet")
            elif "gguf" in self.model_key:
                # GGUF model loading logic here
                # You may need to use a different library for GGUF models
                raise NotImplementedError("GGUF model loading not implemented yet")
            else:
                # Hugging Face model loading
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.model_cache_dir)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # Use float16 for better memory efficiency
                    device_map="auto",  # Automatically use GPU if available
                    cache_dir=self.model_cache_dir
                )
            print(f"{Fore.GREEN}Phi model '{self.model_key}' loaded successfully.{Style.RESET_ALL}")
            self._save_to_cache()
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to load Phi model '{self.model_key}': {str(e)}{Style.RESET_ALL}")
            return False

    def _load_from_cache(self):
        if os.path.exists(self.model_cache_file) and os.path.exists(self.tokenizer_cache_file):
            try:
                with open(self.model_cache_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.tokenizer_cache_file, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                return True
            except Exception as e:
                print(f"{Fore.YELLOW}Failed to load model from cache: {str(e)}. Will load from scratch.{Style.RESET_ALL}")
        return False

    def _save_to_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            with open(self.model_cache_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.tokenizer_cache_file, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"{Fore.GREEN}Model and tokenizer cached successfully.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Failed to cache model: {str(e)}{Style.RESET_ALL}")

    def generate_response(self, question, code_info):
        if not self.tokenizer or not self.model:
            return f"{Fore.RED}Model not loaded. Cannot generate response.{Style.RESET_ALL}"

        context = f"""Analyze the following C++ code information:
File: {code_info['file_path']}
Includes: {code_info['includes']}
Namespaces: {code_info['namespaces']}
Classes: {code_info['classes']}
Structs: {code_info['structs']}
Enums: {code_info['enums']}
Global Variables: {code_info['global_variables']}
Functions: {code_info['functions']}
Templates: {code_info['templates']}
Typedefs: {code_info['typedefs']}
Macros: {code_info['macros']}
Local Variables: {code_info['local_variables']}
Member Variables: {code_info['member_variables']}
Constructors: {code_info['constructors']}
Destructors: {code_info['destructors']}
Operator Overloads: {code_info['operator_overloads']}
Friend Functions: {code_info['friend_functions']}
Virtual Functions: {code_info['virtual_functions']}
Pure Virtual Functions: {code_info['pure_virtual_functions']}
Lambda Expressions: {code_info['lambda_expressions']}
Exception Handlers: {code_info['exception_handlers']}
Memory Allocations: {code_info['memory_allocations']}
Static Assertions: {code_info['static_assertions']}
Anomalies: {code_info['anomalies']}
        """
# Assistant: Based on the provided C++ code information, I will directly answer your question:

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
            outputs = self.model.generate(**inputs, max_new_tokens=200, num_return_sequences=1, temperature=0.7)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        answer = response.split("Assistant:")[-1].strip()
        return f"{Fore.YELLOW}{answer}{Style.RESET_ALL}"
