import sys
import os
import ctypes
import glob
from functools import lru_cache
import networkx as nx
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

warnings.filterwarnings("ignore", category=UserWarning)

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

def setup_clang_library():
    libclang_path = os.environ.get('LIBCLANG_PATH')
    if libclang_path and os.path.exists(libclang_path):
        return libclang_path

    libclang = ctypes.util.find_library("clang")
    if libclang:
        return libclang

    search_paths = ['/usr/lib', '/usr/local/lib', '/usr/local/opt/llvm/lib', '/opt/homebrew/opt/llvm/lib']
    for path in search_paths:
        matches = glob.glob(os.path.join(path, 'libclang*'))
        if matches:
            return matches[0]
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
        node_id = f"{node.kind}:{node.spelling}:{node.location.line}:{node.location.column}"
        G.add_node(node_id, kind=node.kind.name, spelling=node.spelling)
        if parent:
            G.add_edge(parent, node_id)
        for child in node.get_children():
            add_node_and_edges(child, node_id)

    add_node_and_edges(cursor)
    return G

def extract_code_features(G):
    if G is None:
        return [0, 0, 0, 0, 0, 0]
    features = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'num_functions': sum(1 for _, data in G.nodes(data=True) if data['kind'] == 'FUNCTION_DECL'),
        'num_classes': sum(1 for _, data in G.nodes(data=True) if data['kind'] == 'CLASS_DECL'),
        'max_depth': max(nx.shortest_path_length(G, source=list(G.nodes())[0]).values()) if G.number_of_nodes() > 0 else 0,
    }
    return list(features.values())

def detect_code_anomalies(code_info):
    features = extract_code_features(code_info.get('ast_graph'))
    features = StandardScaler().fit_transform([features])

    clf = IsolationForest(random_state=0).fit(features)
    labels = clf.predict(features)

    if -1 in labels:
        return "Potential code anomaly detected."
    return "No significant code anomalies detected."

@lru_cache(maxsize=1000)
def extract_code_info(file_path):
    try:
        translation_unit = parse_cpp_file(file_path)
        if not translation_unit:
            return None

        cursor = translation_unit.cursor
        code_info = {
            'allocations': [],
            'int_vars': [],
            'functions': [],
            'classes': [],
            'ast_graph': None
        }

        try:
            code_info['ast_graph'] = build_ast_graph(cursor)
        except Exception as e:
            print(f"Error building AST graph: {e}")

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
    except Exception as e:
        print(f"Error extracting code info: {e}")
        return None

def load_phi_model():
    print("Loading Phi-3.5 model...")
    model_name = "microsoft/phi-3.5-mini-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, cache_dir=MODEL_CACHE_DIR)
        print("Phi-3.5 model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load Phi-3.5 model: {str(e)}")
        return None, None

def generate_response(question, code_info, tokenizer, model):
    if not tokenizer or not model:
        return "Model not loaded. Cannot generate response."

    anomaly_detection = detect_code_anomalies(code_info)

    context = f"""
    Analyze the following C++ code information:
    Allocations: {code_info['allocations']}
    Integer Variables: {code_info['int_vars']}
    Functions: {code_info['functions']}
    Classes: {code_info['classes']}
    Anomalies: {anomaly_detection}

    Human: {question}

    Assistant: Based on the provided C++ code information, I can answer your question:
    """

    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(**inputs, max_new_tokens=200, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part after "Assistant:"
    answer = response.split("Assistant:")[-1].strip()
    return answer

def repl(file_path, tokenizer, model):
    code_info = extract_code_info(file_path)
    if not code_info:
        print("Failed to extract code information.")
        return

    print("C++ Code Analyzer REPL (Type 'exit' to quit)")
    while True:
        try:
            question = input("\nAsk a question about the code: ")
            if question.lower() == 'exit':
                break
            answer = generate_response(question, code_info, tokenizer, model)
            print(f"\nAnswer: {answer}")
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit or continue asking questions.")
        except Exception as e:
            print(f"Error generating response: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python a.py <cpp_file_path>")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    tokenizer, model = load_phi_model()
    if tokenizer and model:
        repl(file_path, tokenizer, model)
    else:
        print("Failed to load the Phi-3.5 model. Exiting.")

if __name__ == "__main__":
    main()
