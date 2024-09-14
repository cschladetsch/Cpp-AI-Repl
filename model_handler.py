import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from colorama import Fore, Style
import warnings
import pickle
from spinner import Spinner
import time
import psutil
from tqdm import tqdm

class PhiModelHandler:
    AVAILABLE_MODELS = {
        "3.5-mini": "microsoft/phi-3.5-mini-instruct",
        "3.5-moe": "microsoft/phi-3.5-MoE-instruct",
        "3.5-vision": "microsoft/phi-3.5-vision-instruct",
        "3-mini-4k": "microsoft/phi-3-mini-4k-instruct",
        "3-mini-128k": "microsoft/phi-3-mini-128k-instruct",
        "3-small-8k": "microsoft/phi-3-small-8k-instruct",
        "3-small-128k": "microsoft/phi-3-small-128k-instruct",
        "3-medium-4k": "microsoft/phi-3-medium-4k-instruct",
        "3-medium-128k": "microsoft/phi-3-medium-128k-instruct",
        "3-vision-128k": "microsoft/phi-3-vision-128k-instruct",
    }

    def __init__(self, model_key="3-medium-4k"):
        if model_key not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model key. Choose from: {', '.join(self.AVAILABLE_MODELS.keys())}")
        
        self.model_key = model_key
        self.model_name = self.AVAILABLE_MODELS[model_key]
        self.cache_dir = os.path.expanduser('~/.cache/cpp_analyzer')
        self.model_cache_dir = os.path.join(self.cache_dir, 'models')
        self.model_cache_file = os.path.join(self.cache_dir, f'phi_{model_key}_model.pt')
        self.tokenizer_cache_file = os.path.join(self.cache_dir, f'phi_{model_key}_tokenizer.pkl')
        self.tokenizer = None
        self.model = None
        self.spinner = Spinner()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_from_cache(self):
        if os.path.exists(self.model_cache_file) and os.path.exists(self.tokenizer_cache_file):
            try:
                print(f"{Fore.CYAN}Loading model from cache...{Style.RESET_ALL}")
                state_dict = torch.load(self.model_cache_file, map_location=self.device)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto",
                    cache_dir=self.model_cache_dir
                )
                
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                
                with open(self.tokenizer_cache_file, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                
                return True
            except Exception as e:
                print(f"{Fore.YELLOW}Failed to load model from cache: {str(e)}. Will load from scratch.{Style.RESET_ALL}")
        return False

    def load_model(self):
        if self._load_from_cache():
            print(f"{Fore.GREEN}Phi model '{self.model_key}' loaded from cache.{Style.RESET_ALL}")
            return True

        print(f"{Fore.CYAN}Loading Phi model '{self.model_key}'...{Style.RESET_ALL}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, cache_dir=self.model_cache_dir)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                cache_dir=self.model_cache_dir
            )
            
            self.model.to(self.device)
            
            print(f"{Fore.GREEN}Phi model '{self.model_key}' loaded successfully on {self.device}.{Style.RESET_ALL}")
            self._save_to_cache()
            return True
        except Exception as e:
            print(f"{Fore.RED}Failed to load Phi model '{self.model_key}': {str(e)}{Style.RESET_ALL}")
            return False

    def _save_to_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            print(f"{Fore.CYAN}Saving model to cache...{Style.RESET_ALL}")
            torch.save(self.model.state_dict(), self.model_cache_file)
            with open(self.tokenizer_cache_file, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"{Fore.GREEN}Model state dict and tokenizer cached successfully.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Failed to cache model: {str(e)}. The model will be reloaded in future sessions.{Style.RESET_ALL}")

    def generate_response(self, question, code_info):
        if not self.tokenizer or not self.model:
            return f"{Fore.RED}Model not loaded. Cannot generate response.{Style.RESET_ALL}"
        
        print(f"{Fore.CYAN}Preparing input...{Style.RESET_ALL}")
        start_time = time.time()
        
        try:
            code_info_str = json.dumps(code_info, indent=2, default=str)
        except:
            code_info_str = str(code_info)
        
        context = f"""Based on this C++ code information:
{code_info_str}

Question: {question}
Answer: """

        print(f"{Fore.CYAN}Tokenizing input...{Style.RESET_ALL}")
        inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
        input_tokens = inputs['input_ids'].numel()
        print(f"{Fore.YELLOW}Input tokens: {input_tokens}{Style.RESET_ALL}")

        print(f"{Fore.CYAN}Generating response...{Style.RESET_ALL}")
        self.spinner.start()
        try:
            print(f"{Fore.YELLOW}Using device: {self.device}{Style.RESET_ALL}")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            max_new_tokens = 500
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.3)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except RuntimeError as e:
            if "out of memory" in str(e) and self.device.type == "cuda":
                print(f"{Fore.YELLOW}GPU out of memory. Falling back to CPU.{Style.RESET_ALL}")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.3)
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                print(f"{Fore.RED}Unexpected RuntimeError: {str(e)}{Style.RESET_ALL}")
                return f"{Fore.RED}An unexpected error occurred. Please try again or consider using a smaller model.{Style.RESET_ALL}"
        except Exception as e:
            print(f"{Fore.RED}Error during generation: {str(e)}{Style.RESET_ALL}")
            return f"{Fore.RED}An error occurred while generating the response. Please try again.{Style.RESET_ALL}"
        finally:
            self.spinner.stop()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        output_tokens = len(self.tokenizer.encode(response)) - input_tokens
        print(f"\n{Fore.YELLOW}Output tokens: {output_tokens}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Total processing time: {total_time:.2f} seconds{Style.RESET_ALL}")
        
        # Extract the generated response part (everything after "Answer: ")
        answer_parts = response.split("Answer: ")
        if len(answer_parts) > 1:
            answer = answer_parts[-1].strip()
        else:
            answer = response.split(question)[-1].strip()
        
        if not answer:
            answer = "Unable to generate a response. Please try rephrasing your question."
        
        return f"{Fore.GREEN}{answer}{Style.RESET_ALL}"
