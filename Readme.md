# C++ Code Analyzer with Phi Model

This project is a C++ code analysis tool that uses the Phi language model to provide insights and answer questions about C++ code.

## Features

- Analyzes C++ code structure and components
- Uses the Phi language model to answer questions about the analyzed code
- Supports various Phi model versions
- Provides a command-line interface for interaction

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Colorama
- NetworkX
- scikit-learn
- libclang

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cpp-code-analyzer.git
   cd cpp-code-analyzer
   ```

2. Install the required dependencies:
   ```
   pip install torch transformers colorama networkx scikit-learn libclang
   ```

3. Ensure that libclang is properly installed and accessible in your system path.

## Usage

Run the main script with a C++ file as an argument:

```
python main.py path/to/your/cpp/file.cpp
```

You can specify a different Phi model using the `-m` or `--model` flag:

```
python main.py path/to/your/cpp/file.cpp -m 3.5-mini
```

Available models include:
- 3.5-mini
- 3.5-moe
- 3.5-vision
- 3-mini-4k
- (and others as listed in the `AVAILABLE_MODELS` dictionary)

## Interactive Mode

After loading a C++ file, you can interactively ask questions about the code:

```
cwc> What are the main classes in this file?
cwc> List all the methods in the Shape class
cwc> How many pure virtual functions are there in the code?
```

Type 'exit' to quit the interactive mode.

## Project Structure

- `main.py`: Entry point of the application
- `code_analyzer.py`: Contains the `CodeAnalyzer` class for parsing and analyzing C++ code
- `model_handler.py`: Handles loading and interacting with the Phi model
- `utils.py`: Utility functions for environment setup

## Troubleshooting

If you encounter issues with loading the model or analyzing code, ensure that:
- You have a stable internet connection for initial model download
- libclang is properly installed and configured
- You have sufficient disk space for model caching

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- This project uses the Phi model developed by Microsoft
- Thanks to the developers of PyTorch, Transformers, and other libraries used in this project
