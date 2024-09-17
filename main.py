import argparse
import logging
from model_handler import CodeBERTPhiHandler
from code_analyzer import CodeAnalyzer
from colorama import init, Fore, Style
from tqdm import tqdm
import time
import threading
import queue as queue_module
import concurrent.futures
import os

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
    parser.add_argument("cpp_file", help="Path to the C++ file or directory to analyze")
    parser.add_argument("--codebert", default="microsoft/codebert-base", 
                        help="Specify the CodeBERT model to use (default: microsoft/codebert-base)")
    parser.add_argument("--phi", default="microsoft/phi-3.5-mini-instruct", 
                        help="Specify the Phi model to use (default: microsoft/phi-3.5-mini-instruct)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds for model loading and analysis (default: 300)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-file", help="Specify a file to write logs to")
    parser.add_argument("--output", help="Specify a file to write analysis results")
    return parser.parse_args()

def generate_response_with_timeout(model_handler, question, code_embeddings, timeout):
    def target(q):
        try:
            result = model_handler.generate_combined_response(question, code_embeddings)
            q.put(result)
        except Exception as e:
            q.put(f"Error: {str(e)}")

    q = queue_module.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, "Response generation timed out"
    
    try:
        result = q.get_nowait()
        if isinstance(result, str) and result.startswith("Error:"):
            return None, result
        return result, None
    except queue_module.Empty:
        return None, "Unknown error occurred during response generation"
def analyze_file(file_path, code_analyzer, model_handler):
    try:
        with open(file_path, 'r') as f:
            code_content = f.read()
        print(f"{Fore.GREEN}Analyzing file: {file_path}{Style.RESET_ALL}")
        
        # Static analysis
        static_analysis = code_analyzer.analyze_file(file_path)
        
        # ML-based analysis
        code_embeddings = model_handler.analyze_code_with_codebert(code_content)
        
        return {
            'file_path': file_path,
            'static_analysis': static_analysis,
            'code_embeddings': code_embeddings,
            'code_content': code_content  # Store the code content
        }
    except Exception as e:
        print(f"{Fore.RED}Error analyzing file {file_path}: {e}{Style.RESET_ALL}")
        return None

def generate_response_with_timeout(model_handler, question, code_embeddings, code_content, timeout):
    def target(q):
        try:
            result = model_handler.generate_combined_response(question, code_embeddings, code_content)
            q.put(result)
        except Exception as e:
            q.put(f"Error: {str(e)}")

    q = queue_module.Queue()
    thread = threading.Thread(target=target, args=(q,))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, "Response generation timed out"
    
    try:
        result = q.get_nowait()
        if isinstance(result, str) and result.startswith("Error:"):
            return None, result
        return result, None
    except queue_module.Empty:
        return None, "Unknown error occurred during response generation"

def main():
    args = parse_arguments()
    setup_logging(args.debug, args.log_file)
    logger = logging.getLogger(__name__)

    print(f"{Fore.CYAN}C++ Code Analyzer{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}File/Directory to analyze:{Style.RESET_ALL} {args.cpp_file}")
    print(f"{Fore.YELLOW}CodeBERT model:{Style.RESET_ALL} {args.codebert}")
    print(f"{Fore.YELLOW}Phi model:{Style.RESET_ALL} {args.phi}")
    print(f"{Fore.YELLOW}Timeout:{Style.RESET_ALL} {args.timeout} seconds")
    print(f"{Fore.YELLOW}Debug mode:{Style.RESET_ALL} {'Enabled' if args.debug else 'Disabled'}")

    model_handler = CodeBERTPhiHandler(
        model_codebert=args.codebert,
        model_phi=args.phi,
        timeout=args.timeout,
        debug=args.debug
    )
    
    code_analyzer = CodeAnalyzer()
    
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

    # Determine if the input is a file or directory
    if os.path.isfile(args.cpp_file):
        files_to_analyze = [args.cpp_file]
    elif os.path.isdir(args.cpp_file):
        files_to_analyze = [os.path.join(root, file) 
                            for root, _, files in os.walk(args.cpp_file) 
                            for file in files if file.endswith(('.cpp', '.hpp', '.h'))]
    else:
        print(f"{Fore.RED}Error: {args.cpp_file} is not a valid file or directory{Style.RESET_ALL}")
        return

    print(f"{Fore.YELLOW}Analyzing code...{Style.RESET_ALL}")
    analysis_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(analyze_file, file, code_analyzer, model_handler): file for file in files_to_analyze}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files_to_analyze), desc="Analyzing files", ncols=70):
            result = future.result()
            if result:
                analysis_results.append(result)

    print(f"{Fore.GREEN}Code analysis complete{Style.RESET_ALL}")

    if args.output:
        # Save analysis results to file
        with open(args.output, 'w') as f:
            for result in analysis_results:
                f.write(f"File: {result['file_path']}\n")
                f.write("Static Analysis:\n")
                for key, value in result['static_analysis'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        print(f"{Fore.GREEN}Analysis results saved to {args.output}{Style.RESET_ALL}")

    print(f"\n{Fore.GREEN}Analysis complete. Entering REPL. Type 'exit' to quit.{Style.RESET_ALL}")
    
    while True:
        try:
            question = input(f"\n{Fore.BLUE}cwc> {Style.RESET_ALL}")
            if question.lower() == 'exit':
                break

            print(f"{Fore.YELLOW}Generating response...{Style.RESET_ALL}")
            if use_alive_progress:
                with alive_bar(title='Generating response', bar='classic', spinner='classic') as bar:
                    response, error = generate_response_with_timeout(
                        model_handler, 
                        question, 
                        analysis_results[0]['code_embeddings'],
                        analysis_results[0]['code_content'],
                        args.timeout
                    )
                    bar()
            else:
                response, error = generate_response_with_timeout(
                    model_handler, 
                    question, 
                    analysis_results[0]['code_embeddings'],
                    analysis_results[0]['code_content'],
                    args.timeout
                )

            if error:
                print(f"\n{Fore.RED}Error: {error}{Style.RESET_ALL}")
                logger.error(f"Error generating response: {error}")
            elif response is None:
                print(f"\n{Fore.YELLOW}No response generated. This might be due to an internal error or limitation.{Style.RESET_ALL}")
                logger.warning("No response generated")
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
