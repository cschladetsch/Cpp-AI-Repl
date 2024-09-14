import sys
import os
import argparse
from colorama import init, Fore, Style
from code_analyzer import CodeAnalyzer
from model_handler import PhiModelHandler
from utils import setup_environment

# Initialize colorama for colored output
init(autoreset=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="C++ Code Analyzer with Phi Model")
    parser.add_argument("cpp_file", help="Path to the C++ file to analyze")
    parser.add_argument("-m", "--model", choices=PhiModelHandler.AVAILABLE_MODELS.keys(), 
                        default="small", help="Select the Phi model to use (default: small)")
    return parser.parse_args()

def repl(file_path, model_handler, code_analyzer):
    while True:
        try:
            question = input(f"\n{Fore.GREEN}cwc> {Style.RESET_ALL}")
            if question.lower() == 'exit':
                break
            code_info = code_analyzer.analyze_file(file_path)
            if not code_info:
                print(f"{Fore.RED}Failed to extract code information.{Style.RESET_ALL}")
                continue
            answer = model_handler.generate_response(question, code_info)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please try again or type 'exit' to quit.{Style.RESET_ALL}")

def main():
    args = parse_arguments()

    if not os.path.exists(args.cpp_file):
        print(f"{Fore.RED}Error: File '{args.cpp_file}' does not exist.{Style.RESET_ALL}")
        return

    setup_environment()
    
    model_handler = PhiModelHandler(model_key=args.model)
    code_analyzer = CodeAnalyzer()

    print(f"{Fore.CYAN}Using Phi model: {args.model}{Style.RESET_ALL}")

    if model_handler.load_model():
        repl(args.cpp_file, model_handler, code_analyzer)
    else:
        print(f"{Fore.RED}Failed to load the Phi model. Exiting.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
