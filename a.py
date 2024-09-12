import sys
import os
import ctypes
import glob
import hashlib
import pickle
import torch
import networkx as nx
import numpy as np
import warnings

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from functools import lru_cache
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

# Suppress warnings
warnings.filterwarnings("ignore", message=".*flash-attention.*")
warnings.filterwarnings("ignore", message=".*window_size.*")

# Set up caching directories
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'cpp_analyzer')
CLANG_CACHE_DIR = os.path.join(CACHE_DIR, 'clang')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')

os.makedirs(CLANG_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
os.environ['HF_HOME'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')

# Function to cache files
def cache_file(file_path, cache_dir):
    try:
        file_hash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        cache_path = os.path.join(cache_dir, f"{os.path.basename(file_path)}_{file_hash}")
        if not os.path.exists(cache_path):
            os.symlink(file_path, cache_path)
        return cache_path
    except Exception as e:
        print(f"Error in cache_file: {e}")
        return file_path

# Function to load libclang
def setup_clang_library():
    libclang_path = os.environ.get('LIBCLANG_PATH')
    if libclang_path and os.path.exists(libclang_path):
        return cache_file(libclang_path, CLANG_CACHE_DIR)

    libclang = ctypes.util.find_library("clang")
    if libclang:
        return cache_file(libclang, CLANG_CACHE_DIR)

    search_paths = ['/usr/lib', '/usr/local/lib', '/usr/local/opt/llvm/lib', '/opt/homebrew/opt/llvm/lib']
    for path in search_paths:
        matches = glob.glob(os.path.join(path, 'libclang*'))
        if matches:
            return cache_file(matches[0], CLANG_CACHE_DIR)
    return None

def load_clang():
    libclang_path = setup_clang_library()
    if libclang_path is None:
        print("Failed to load libclang.")
        return False

    try:
        import clang.cindex
        clang.cindex.Config.set_library_file(libclang_path)
        return True
    except ImportError:
        print("Install libclang with 'pip install libclang'.")
        return False

if not load_clang():
    sys.exit(1)

import clang.cindex

# Parse the C++ file and build the AST graph
@lru_cache(maxsize=100)
def parse_cpp_file(file_path):
    try:
        index = clang.cindex.Index.create()
        return index.parse(file_path)
    except Exception as e:
        print(f"Error parsing C++ file: {e}")
        return None

@lru_cache(maxsize=100)
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

# Extract features from the AST graph
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

# Detect anomalies using Isolation Forest
def detect_code_anomalies(code_info):
    features = extract_code_features(code_info['ast_graph'])
    features = StandardScaler().fit_transform([features])

    clf = IsolationForest(random_state=0).fit(features)
    labels = clf.predict(features)

    if -1 in labels:
        return "Potential code anomaly detected."
    return "No significant code anomalies detected."

@lru_cache(maxsize=1000)
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
        if node.kind == clang.cindex.CursorKind.CALL_EXPR and node.spelling in ['new', 'malloc']:
            code_info['allocations'].append((node.location.file.name, node.location.line, node.spelling))
        elif node.kind == clang.cindex.CursorKind.VAR_DECL and node.type.spelling == 'int':
            code_info['int_vars'].append((node.spelling, node.location.file.name, node.location.line))
        elif node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            code_info['functions'].append((node.spelling, node.location.file.name, node.location.line))
        elif node.kind == clang.cindex.CursorKind.CLASS_DECL:
            code_info['classes'].append((node.spelling, node.location.file.name, node.location.line))

        for child in node.get_children():
            visit_node(child)

    visit_node(cursor)
    return code_info

# Load the model from Hugging Face with fallback to GPT-2
def load_model():
    cache_file_path = os.path.join(MODEL_CACHE_DIR, 'model_cache.pkl')
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            os.remove(cache_file_path)

    try:
        print("Attempting to load CodeT5 model...")
        model_name = "Salesforce/codet5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("CodeT5 model loaded successfully.")
    except Exception as e:
        print(f"Failed to load CodeT5 model: {e}")
        print("Falling back to GPT-2 model...")
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("GPT-2 model loaded successfully.")

    with open(cache_file_path, 'wb') as f:
        pickle.dump((tokenizer, model), f)
        print("Model cached successfully.")

    return tokenizer, model

# Generate a response using the model
@lru_cache(maxsize=1000)
def generate_response(question, file_path, tokenizer, model):
    code_info = extract_code_info(file_path)
    if not code_info:
        return "Could not analyze the code."

    anomaly_detection = detect_code_anomalies(code_info)

    context = f"""
    Allocations: {code_info['allocations']}
    Integer Variables: {code_info['int_vars']}
    Functions: {code_info['functions']}
    Classes: {code_info['classes']}
    Anomalies: {anomaly_detection}
    Question: {question}
    """

    inputs = tokenizer(context, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

# REPL for interactive analysis
def repl(file_path, tokenizer, model):
    print("C++ Code Analyzer REPL (Type 'exit' to quit)")
    while True:
        question = input("\nAsk a question about the code: ")
        if question.lower() == 'exit':
            break
        answer = generate_response(question, file_path, tokenizer, model)
        print(f"\nAnswer: {answer}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python a.py <cpp_file_path>")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    tokenizer, model = load_model()
    if tokenizer and model:
        repl(file_path, tokenizer, model)

if __name__ == "__main__":
    main()


