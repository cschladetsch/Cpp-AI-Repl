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
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define cache directories
BASE_CACHE_DIR = os.path.expanduser('~/.cache')
CPP_ANALYZER_CACHE_DIR = os.path.join(BASE_CACHE_DIR, 'cpp_analyzer')
MODEL_CACHE_DIR = os.path.join(CPP_ANALYZER_CACHE_DIR, 'models')

# Create cache directories if they do not exist
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

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
        print(f"{Fore.RED}Failed to load libclang.{Style.RESET_ALL}")
        return False

    try:
        import clang.cindex
        clang.cindex.Config.set_library_file(libclang_path)
        return True
    except ImportError:
        print(f"{Fore.YELLOW}Install libclang with 'pip install libclang'.{Style.RESET_ALL}")
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
        print(f"{Fore.RED}Error parsing C++ file: {e}{Style.RESET_ALL}")
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
        return f"{Fore.RED}Potential code anomaly detected.{Style.RESET_ALL}"
    return f"{Fore.GREEN}No significant code anomalies detected.{Style.RESET_ALL}"

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
            print(f"{Fore.RED}Error building AST graph: {e}{Style.RESET_ALL}")

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
        print(f"{Fore.RED}Error extracting code info: {e}{Style.RESET_ALL}")
        return None

def load_phi_model():
    print(f"{Fore.CYAN}Loading Phi-3.5 model...{Style.RESET_ALL}")
    model_name = "microsoft/phi-3.5-mini-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32, cache_dir=MODEL_CACHE_DIR)
        print(f"{Fore.GREEN}Phi-3.5 model loaded successfully.{Style.RESET_ALL}")
        return tokenizer, model
    except Exception as e:
        print(f"{Fore.RED}Failed to load Phi-3.5 model: {str(e)}{Style.RESET_ALL}")
        return None, None

def generate_response(question, code_info, tokenizer, model):
    if not tokenizer or not model:
        return f"{Fore.RED}Model not loaded. Cannot generate response.{Style.RESET_ALL}"

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
    return f"{Fore.YELLOW}{answer}{Style.RESET_ALL}"

def repl(file_path, tokenizer, model):
    code_info = extract_code_info(file_path)
    if not code_info:
        print(f"{Fore.RED}Failed to extract code information.{Style.RESET_ALL}")
        return

    print(f"{Fore.CYAN}C++ Code Analyzer REPL (Type 'exit' to quit){Style.RESET_ALL}")
    while True:
        try:
            question = input(f"\n{Fore.GREEN}Ask a question about the code: {Style.RESET_ALL}")
            if question.lower() == 'exit':
                break
            answer = generate_response(question, code_info, tokenizer, model)
            print(f"\nAnswer: {answer}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Interrupted. Type 'exit' to quit or continue asking questions.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {e}{Style.RESET_ALL}")

def main():
    if len(sys.argv) < 2:
        print(f"{Fore.RED}Usage: python a.py <cpp_file_path>{Style.RESET_ALL}")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"{Fore.RED}Error: File '{file_path}' does not exist.{Style.RESET_ALL}")
        return

    tokenizer, model = load_phi_model()
    if tokenizer and model:
        repl(file_path, tokenizer, model)
    else:
        print(f"{Fore.RED}Failed to load the Phi-3.5 model. Exiting.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

