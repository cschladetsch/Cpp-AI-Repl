import os
import torch
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
import time
import logging
from colorama import Fore, Style

class CodeBERTPhiHandler:
    def __init__(self, model_codebert="microsoft/codebert-base", model_phi="microsoft/phi-3.5-mini-instruct", timeout=300, debug=False):
        self.codebert_model_name = model_codebert
        self.phi_model_name = model_phi
        self.timeout = timeout
        self.debug = debug
        self.device = torch.device("cpu")
        self.cache_dir = os.path.expanduser('~/.cache/cpp-analyzer')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def load_codebert_model(self):
        self.logger.info("Loading CodeBERT model...")
        self.codebert_tokenizer = AutoTokenizer.from_pretrained(self.codebert_model_name, cache_dir=self.cache_dir)
        self.codebert_model = AutoModelForMaskedLM.from_pretrained(self.codebert_model_name, cache_dir=self.cache_dir).to(self.device)
        self.logger.info("CodeBERT model loaded successfully")

    def load_phi_model(self):
        self.logger.info("Loading Phi model...")
        self.phi_tokenizer = AutoTokenizer.from_pretrained(self.phi_model_name, cache_dir=self.cache_dir)
        self.phi_model = AutoModelForCausalLM.from_pretrained(self.phi_model_name, cache_dir=self.cache_dir).to(self.device)
        self.logger.info("Phi model loaded successfully")

    def analyze_code_with_codebert(self, code):
        self.logger.debug("Analyzing code with CodeBERT")
        inputs = self.codebert_tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
        return outputs.logits.mean(dim=1)

    def generate_combined_response(self, question, code_embeddings):
        start_time = time.time()
        self.logger.debug(f"Generating response for question: {question}")

        self.logger.debug("Generating query embeddings")
        query_inputs = self.codebert_tokenizer(question, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            query_outputs = self.codebert_model(**query_inputs)
        query_embeddings = query_outputs.logits.mean(dim=1)

        self.logger.debug("Combining embeddings")
        combined_embeddings = torch.cat((query_embeddings, code_embeddings), dim=1)
        combined_context = ' '.join(map(str, combined_embeddings.cpu().numpy().flatten().tolist()))

        self.logger.debug("Preparing input for Phi model")
        phi_input = f"CodeBERT context: {combined_context}\nQuestion: {question}\nAnswer:"
        
        inputs = self.phi_tokenizer(phi_input, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        self.logger.debug("Generating response with Phi model")
        with torch.no_grad():
            outputs = self.phi_model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
        
        self.logger.debug("Decoding response")
        response = self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Answer:")[-1].strip()
        
        end_time = time.time()
        self.logger.debug(f"Response generation completed in {end_time - start_time:.2f} seconds")
        
        return response
