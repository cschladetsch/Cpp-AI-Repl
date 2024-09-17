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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        try:
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                self.phi_model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except RuntimeError as e:
            self.logger.warning(f"Failed to load Phi model on GPU: {e}")
            self.logger.info("Attempting to load Phi model on CPU...")
            self.phi_model = AutoModelForCausalLM.from_pretrained(
                self.phi_model_name,
                cache_dir=self.cache_dir,
                device_map="cpu"
            )
        self.logger.info("Phi model loaded successfully")

    def analyze_code_with_codebert(self, code):
        self.logger.debug("Analyzing code with CodeBERT")
        inputs = self.codebert_tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.codebert_model(**inputs)
        return outputs.logits.mean(dim=1)

    def generate_combined_response(self, question, code_embeddings, code_content, static_analysis):
        start_time = time.time()
        self.logger.debug(f"Generating response for question: {question}")

        try:
            self.logger.debug("Generating query embeddings")
            query_inputs = self.codebert_tokenizer(question, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                query_outputs = self.codebert_model(**query_inputs)
            query_embeddings = query_outputs.logits.mean(dim=1)

            self.logger.debug("Combining embeddings")
            combined_embeddings = torch.cat((query_embeddings, code_embeddings), dim=1)

            embedding_summary = f"Embedding shape: {combined_embeddings.shape}, Mean: {combined_embeddings.mean().item():.2f}, Max: {combined_embeddings.max().item():.2f}"
            code_snippet = code_content[:500] + "..." if len(code_content) > 500 else code_content
            
            static_analysis_summary = "\n".join([f"{key}: {value}" for key, value in static_analysis.items() if key != 'ast_graph'])

            self.logger.debug("Preparing input for Phi model")
            phi_input = f"""Analyze the following C++ code and answer the question:

C++ Code Snippet:
{code_snippet}

Static Analysis Summary:
{static_analysis_summary}

Code Embedding Summary:
{embedding_summary}

Question: {question}

Provide a concise and relevant answer based on the C++ code and analysis provided:
"""
            
            inputs = self.phi_tokenizer(phi_input, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.phi_model.device) for k, v in inputs.items()}
            
            self.logger.debug("Generating response with Phi model")
            self.phi_model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                outputs = self.phi_model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3
                )

            self.logger.debug("Decoding response")
            response = self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Provide a concise and relevant answer based on the C++ code and analysis provided:")[-1].strip()
            
            end_time = time.time()
            self.logger.debug(f"Response generation completed in {end_time - start_time:.2f} seconds")
            
            return response
        except Exception as e:
            self.logger.error(f"Error in generate_combined_response: {str(e)}")
            self.logger.exception("Detailed error information:")
            raise
