import sys
import os
import ctypes
import ctypes.util
import glob
import warnings
import hashlib
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch
from functools import lru_cache
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*window_size.*")

# Set up caching directories
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'cpp_analyzer')
CLANG_CACHE_DIR = os.path.join(CACHE_DIR, 'clang')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')

os.makedirs(CLANG_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Set up caching for Hugging Face models and tokenizers
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
os.environ['HF_HOME'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')

def cache_file(file_path, cache_dir):
    try:
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{os.path.basename(file_path)}_{file_hash}")
        if not os.path.exists(cache_path):
            os.symlink(file_path, cache_path)
        return cache_path
    except Exception as e:
        print(f"Error in cache_file: {e}")
        return file_path  # Return original path if caching fails

def find_libclang():
    possible_names = ['libclang.so', 'libclang-*.so', 'libclang.dylib']
    search_paths = [
        '/usr/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/llvm-*/lib',
        '/usr/local/lib',
        '/usr/local/opt/llvm/lib',
        '/opt/homebrew/opt/llvm/lib'  # For macOS with Homebrew
    ]
    
    for path in search_paths:
        for name in possible_names:
            full_path = os.path.join(path, name)
            matches = glob.glob(full_path)
            for match in matches:
                if os.path.isfile(match):
                    return cache_file(match, CLANG_CACHE_DIR)
    return None

def setup_clang_library():
    try:
        # First, check if LIBCLANG_PATH is set in the environment
        libclang_path = os.environ.get('LIBCLANG_PATH')
        if libclang_path and os.path.exists(libclang_path):
            cached_path = cache_file(libclang_path, CLANG_CACHE_DIR)
            ctypes.CDLL(cached_path)
            print(f"Loaded libclang from LIBCLANG_PATH (cached): {cached_path}")
            return cached_path

        # Try to find libclang using ctypes
        libclang = ctypes.util.find_library("clang")
        if libclang:
            cached_path = cache_file(libclang, CLANG_CACHE_DIR)
            ctypes.CDLL(cached_path)
            print(f"Loaded libclang from system path (cached): {cached_path}")
            return cached_path

        # If ctypes couldn't find it, try our manual search
        libclang_path = find_libclang()
        if libclang_path:
            ctypes.CDLL(libclang_path)
            print(f"Loaded libclang from found path (cached): {libclang_path}")
            return libclang_path

        print("Could not find libclang. You may need to install libclang-dev or set the LIBCLANG_PATH environment variable.")
        return None
    except Exception as e:
        print(f"Error loading libclang: {e}")
        print("You may need to install libclang-dev or set the LIBCLANG_PATH environment variable.")
        return None

def load_clang():
    libclang_path = setup_clang_library()

    if libclang_path is None:
        print("Failed to load libclang. Please ensure it's installed and try setting the LIBCLANG_PATH environment variable.")
        return False

    try:
        import clang.cindex
        clang.cindex.Config.set_library_file(libclang_path)
        return True
    except ImportError:
        print("Failed to import clang.cindex. Please ensure the python-clang package is installed.")
        print("You can install it using: pip install libclang")
        return False
    except Exception as e:
        print(f"Error setting up clang.cindex: {e}")
        return False

# Load Clang at the beginning
if not load_clang():
    sys.exit(1)

# Now we can import clang.cindex
import clang.cindex

@lru_cache(maxsize=100)
def parse_cpp_file(file_path):
    try:
        index = clang.cindex.Index.create()
        return index.parse(file_path)
    except Exception as e:
        print(f"Error parsing C++ file: {e}")
        return None

def build_ast_graph(cursor):
    G = nx.DiGraph()
    
    def add_node_and_edges(node, parent=None):
        node_id = str(node.hash)
        G.add_node(node_id, kind=node.kind.name, spelling=node.spelling)
        if parent:
            G.add_edge(str(parent.hash), node_id)
        for child in node.get_children():
            add_node_and_edges(child, node)
    
    add_node_and_edges(cursor)
    return G

def extract_code_features(G):
    features = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        'num_functions': sum(1 for _, data in G.nodes(data=True) if data['kind'] == 'FUNCTION_DECL'),
        'num_classes': sum(1 for _, data in G.nodes(data=True) if data['kind'] == 'CLASS_DECL'),
        'max_depth': max(nx.shortest_path_length(G, source=list(G.nodes())[0]).values()),
    }
    return list(features.values())

@lru_cache(maxsize=100)
def extract_code_info(file_path):
    translation_unit = parse_cpp_file(file_path)
    if not translation_unit:
        return None
    
    cursor = translation_unit.cursor
    code_info = {
        'allocations': [],
        'int_vars': [],
        'functions': [],
        'classes': [],
        'ast_graph': build_ast_graph(cursor)
    }
    
    def visit_node(node):
        if node.kind == clang.cindex.CursorKind.CALL_EXPR:
            if node.spelling in ['new', 'malloc', 'calloc', 'realloc']:
                code_info['allocations'].append((node.location.file.name, node.location.line, node.spelling))
        elif node.kind == clang.cindex.CursorKind.VAR_DECL:
            if node.type.spelling == 'int':
                code_info['int_vars'].append((node.spelling, node.location.file.name, node.location.line))
        elif node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            code_info['functions'].append((node.spelling, node.location.file.name, node.location.line))
        elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
            code_info['classes'].append((node.spelling, node.location.file.name, node.location.line))
        
        for child in node.get_children():
            visit_node(child)
    
    visit_node(cursor)
    return code_info

def load_model():
    cache_file = os.path.join(MODEL_CACHE_DIR, 'model_cache.pkl')
    if os.path.exists(cache_file):
        print("Attempting to load model from cache...")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Failed to load model from cache: {e}")
            print("Falling back to loading model from Hugging Face cache.")
            os.remove(cache_file)  # Remove the invalid cache file

    try:
        print("Loading Phi-3.5 model from Hugging Face cache...")
        model_name = "microsoft/Phi-3.5-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
        print("Phi-3.5 model loaded successfully.")
        print("Note: For better performance, consider installing the 'flash-attention' package.")
    except Exception as e:
        print(f"Failed to load Phi-3.5 model: {e}")
        print("Falling back to GPT-2 model...")
        model_name = "gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        print("GPT-2 model loaded successfully.")

    # Cache the loaded model and tokenizer
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump((tokenizer, model), f)
        print("Model cached successfully.")
    except Exception as e:
        print(f"Failed to cache model: {e}")

    return tokenizer, model

def detect_code_anomalies(code_info):
    features = extract_code_features(code_info['ast_graph'])
    features = StandardScaler().fit_transform([features])
    
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(features)
    labels = clustering.labels_
    
    if -1 in labels:
        return "Potential code anomaly detected. This code structure is significantly different from typical patterns."
    return "No significant code anomalies detected."

@lru_cache(maxsize=1000)
def generate_response(question, file_path, tokenizer, model):
    code_info = extract_code_info(file_path)
    if not code_info:
        return "Sorry, I couldn't analyze the code due to an error in parsing the C++ file."
    
    anomaly_detection = detect_code_anomalies(code_info)
    
    context = f"""
    As an AI assistant specialized in C++ code analysis, analyze the following code information and answer the question:

    Dynamic Allocations: {code_info['allocations']}
    Integer Variables: {code_info['int_vars']}
    Functions: {code_info['functions']}
    Classes: {code_info['classes']}
    
    Code Structure Analysis: {anomaly_detection}

    Human: {question}

    Assistant: Certainly! I'll analyze the provided C++ code information and answer your question. 
    """

    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("Assistant:")[-1].strip()
    return answer

def repl(file_path, tokenizer, model):
    print("C++ Code Analyzer REPL (Type 'exit' to quit)")
    print("--------------------------------------------------------")
    
    while True:
        question = input("\nAsk a question about the code: ")
        if question.lower() == 'exit':
            break
        
        answer = generate_response(question, file_path, tokenizer, model)
        print(f"\nAnswer: {answer}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <cpp_file_path>")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    print("Loading language model (this may take a moment on first run)...")
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(f"Failed to load the language model: {e}")
        print("Please ensure you have an active internet connection and the required libraries installed.")
        return

    # Test C++ parsing
    test_parse = parse_cpp_file(file_path)
    if test_parse is None:
        print("Failed to parse C++ file. Please check your Clang installation and ensure the file is valid C++ code.")
        return

    print(f"Successfully loaded the model and parsed the C++ file: {file_path}")
    print("You can now start asking questions about the code.")
    
    repl(file_path, tokenizer, model)

if __name__ == "__main__":
    main()
