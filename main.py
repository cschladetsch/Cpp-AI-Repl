import argparse
import logging
from model_handler import CodeBERTPhiHandler
from colorama import init, Fore, Style
from tqdm import tqdm
import time
import threading
import queue as queue_module

# Initialize colorama
init(autoreset=True)

# Try to import alive_progress, use a fallback if not available
try:
    from alive_progress import alive_bar
    use_alive_progress = True
except ImportError:
    use_alive_progress = False
    print(f"{Fore.YELLOW}Note: For enhanced progress bars, install 'alive-progress' using 'pip install alive-progress'{Style.RESET_ALL}")

def setup_logging(debug=False, log_file=None):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="C++ Code Analyzer with CodeBERT and Phi models")
    parser.add_argument("cpp_file", help="Path to the C++ file to analyze")
    parser.add_argument("--codebert", default="microsoft/codebert-base", 
                        help="Specify the CodeBERT model to use (default: microsoft/codebert-base)")
    parser.add_argument("--phi", default="microsoft/phi-3.5-mini-instruct", 
                        help="Specify the Phi model to use (default: microsoft/phi-3.5-mini-instruct)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds for model loading and analysis (default: 300)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-file", help="Specify a file to write logs to")
    return parser.parse_args()

def generate_response_with_timeout(model_handler, question, code_embeddings, timeout):
    def target(q):
        try:
            result = model_handler.generate_combined_response(question, code_embeddings)
            q.put(result)
        except Exception as e:
            q.put(e)

    q = queue_module.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, "Response generation timed out"
    
    try:
        result = q.get_nowait()
        if isinstance(result, Exception):
            return None, f"Error during response generation: {str(result)}"
        return result, None
    except queue_module.Empty:
        return None, "Unknown error occurred during response generation"

def main():
    args = parse_arguments()
    setup_logging(args.debug, args.log_file)
    logger = logging.getLogger(__name__)

    print(f"{Fore.CYAN}C++ Code Analyzer{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}File to analyze:{Style.RESET_ALL} {args.cpp_file}")
    print(f"{Fore.YELLOW}CodeBERT model:{Style.RESET_ALL} {args.codebert}")
    print(f"{Fore.YELLOW}Phi model:{Style.RESET_ALL} {args.phi}")
    print(f"{Fore.YELLOW}Timeout:{Style.RESET_ALL} {args.timeout} seconds")
    print(f"{Fore.YELLOW}Debug mode:{Style.RESET_ALL} {'Enabled' if args.debug else 'Disabled'}")
    
    try:
        with open(args.cpp_file, 'r') as f:
            code_content = f.read()
        print(f"{Fore.GREEN}File read successfully{Style.RESET_ALL}")
    except FileNotFoundError:
        print(f"{Fore.RED}Error: File not found: {args.cpp_file}{Style.RESET_ALL}")
        return
    except Exception as e:
        print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
        return

    model_handler = CodeBERTPhiHandler(
        model_codebert=args.codebert,
        model_phi=args.phi,
        timeout=args.timeout,
        debug=args.debug
    )
    
    print(f"{Fore.YELLOW}Loading models...{Style.RESET_ALL}")
    if use_alive_progress:
        with alive_bar(2, title='Loading models', bar='classic', spinner='classic') as bar:
            model_handler.load_codebert_model()
            bar()
            model_handler.load_phi_model()
            bar()
    else:
        print("Loading CodeBERT model...")
        model_handler.load_codebert_model()
        print("Loading Phi model...")
        model_handler.load_phi_model()
    print(f"{Fore.GREEN}Models loaded successfully{Style.RESET_ALL}")

    print(f"{Fore.YELLOW}Analyzing code with CodeBERT...{Style.RESET_ALL}")
    for _ in tqdm(range(100), desc="Analyzing code", ncols=70):
        time.sleep(0.05)  # Simulating analysis time
    code_embeddings = model_handler.analyze_code_with_codebert(code_content)
    print(f"{Fore.GREEN}Code analysis complete{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}Analysis complete. Entering REPL. Type 'exit' to quit.{Style.RESET_ALL}")
    
    while True:
        try:
            question = input(f"\n{Fore.BLUE}cwc> {Style.RESET_ALL}")
            if question.lower() == 'exit':
                break

            print(f"{Fore.YELLOW}Generating response...{Style.RESET_ALL}")
            if use_alive_progress:
                with alive_bar(title='Generating response', bar='classic', spinner='classic') as bar:
                    response, error = generate_response_with_timeout(model_handler, question, code_embeddings, args.timeout)
                    bar()
            else:
                response, error = generate_response_with_timeout(model_handler, question, code_embeddings, args.timeout)

            if error:
                print(f"\n{Fore.RED}Error: {error}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL} {response}")
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Keyboard interrupt received. Exiting...{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}Error processing query: {e}{Style.RESET_ALL}")
            logger.exception("Detailed error information:")

    print(f"{Fore.CYAN}Exiting C++ Analyzer. Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
