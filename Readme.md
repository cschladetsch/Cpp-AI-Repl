# C++ Code Analyzer with Phi Model
[![Code Climate](https://codeclimate.com/github/cschladetsch/Cpp-AI-Repl/badges/gpa.svg)](https://codeclimate.com/github/username/Cpp-AI-Repl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a C++ code analysis tool that uses the Phi language model to provide insights and answer questions about C++ code.

## Features

- Analyzes C++ code structure and components
- Uses the Phi language model to answer questions about the analyzed code
- Supports various Phi model versions
- Provides a command-line interface for interaction

## Example Session

Here's a sample session with `cwc` as of September 2024:

![Session](resources/cwc-1.jpg)

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Colorama
- NetworkX
- scikit-learn
- libclang
- CodeBERT

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/cschladetsch/Cpp-AI-Repl cpp-ai-repl
   cd cpp-ai-repl
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure that libclang is properly installed and accessible in your system path.
    * run `bash setup.py`

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

MIT

## Acknowledgments

- This project uses the Phi model developed by Microsoft
- Thanks to the developers of PyTorch, Transformers, and other libraries used in this project

-------
# CodeBERT vs Phi-3.5 for C++ Code Analysis

## CodeBERT

1. **Purpose**: CodeBERT is specifically designed for programming language understanding tasks.
2. **Architecture**: Based on RoBERTa, a BERT variant optimized for code.
3. **Training**: Pre-trained on large-scale code repositories in multiple programming languages, including C++.
4. **Strengths**:
   - Strong at understanding code structure and semantics
   - Good at tasks like code search, clone detection, and code-to-text generation
5. **Limitations**:
   - Not designed for open-ended text generation
   - Limited context window (typically 512 tokens)

## Phi-3.5

1. **Purpose**: General-purpose language model with instruction-following capabilities.
2. **Architecture**: Based on the Transformer architecture, optimized for efficiency.
3. **Training**: Trained on a broad range of internet text, including some programming-related content.
4. **Strengths**:
   - Capable of generating human-like responses to open-ended questions
   - Can follow complex instructions and generate coherent, contextual responses
   - Larger context window (up to 128k tokens for some versions)
5. **Limitations**:
   - Not specifically optimized for code understanding
   - May sometimes generate plausible but incorrect code or explanations

## Working Together in Your Code

1. **Complementary Roles**:
   - CodeBERT provides a deep understanding of the code structure and semantics.
   - Phi-3.5 generates human-readable responses and explanations based on the code analysis.

2. **Integration Process**:
   - The code is first analyzed using the `CodeAnalyzer` class, which uses Clang to parse the C++ code and extract structural information.
   - This structural information is then processed by CodeBERT to generate embeddings that capture the code's semantic meaning.
   - The CodeBERT embeddings are combined with the user's question and fed into Phi-3.5.
   - Phi-3.5 uses this combined input to generate a response that leverages both the code understanding from CodeBERT and its own language generation capabilities.

3. **Specific Implementation**:
   - In the `generate_response` method of `ModelHandler`:
     - CodeBERT processes the input (which includes the code summary and user question) to generate embeddings.
     - These embeddings are concatenated with the Phi-3.5 input tokens.
     - Phi-3.5 then generates the final response based on this combined input.

This approach allows the system to leverage CodeBERT's specialized code understanding capabilities while utilizing Phi-3.5's more general language understanding and generation abilities to provide informative and contextually relevant responses to user queries about the C++ code.

-------

## C++ Analyzer Code Review

### Project Overview

This project is a C++ code analyzer that combines static analysis with machine learning techniques using CodeBERT and Phi models. It's designed to analyze C++ files, provide insights, and answer questions about the code.
File Structure

* main.py: Entry point of the application
* code_analyzer.py: Contains the CodeAnalyzer class for static analysis
* model_handler.py: Handles the CodeBERT and Phi models
* replace_method.py: Utility script for replacing methods in Python files
* utils.py: Contains utility functions for environment setup

### Key Components

1. Main Application (main.py)
    Handles command-line arguments
    Sets up logging
    Manages the overall flow of the application
    Provides a REPL (Read-Eval-Print Loop) for user interactions

2. Code Analyzer (code_analyzer.py)
    Uses clang for parsing C++ files
    Builds an Abstract Syntax Tree (AST) graph
    Extracts various code features
    Detects potential code anomalies using Isolation Forest

3. Model Handler (model_handler.py)
    Manages CodeBERT and Phi models
    Handles model loading, code analysis, and response generation

4. Replace Method Utility (replace_method.py)
    Standalone script for replacing methods in Python files
    Uses regex for method detection and replacement

5. Utilities (utils.py)
    Sets up the environment
    Manages Clang library setup

### Observations and Suggestions

Modularity: The project is well-structured with clear separation of concerns between different components.
Error Handling: Good use of try-except blocks for error handling throughout the code.
Logging: Comprehensive logging is implemented, which is crucial for debugging and monitoring.
Model Management: The CodeBERTPhiHandler class efficiently manages both CodeBERT and Phi models.
Concurrency: The main script uses concurrent.futures for parallel file analysis, which is good for performance.
User Interface: The use of colorama for colored console output enhances user experience.
Caching: Model caching is implemented to improve loading times on subsequent runs.
Flexibility: The code allows for different CodeBERT and Phi models to be specified via command-line arguments.
Timeout Handling: Timeouts are implemented for model loading and analysis, which is important for handling large files or slow systems.
Code Quality: Overall, the code is well-commented and follows good Python practices.

### Potential Improvements

Configuration: Consider using a configuration file for default settings instead of hardcoding them.
Testing: Add unit tests for critical components to ensure reliability.
Documentation: While the code is well-commented, adding docstrings to classes and methods would improve maintainability.
Error Recovery: Implement more robust error recovery mechanisms, especially in the REPL loop.
Progress Reporting: Consider using tqdm consistently across all long-running operations for better progress visibility.
Code Optimization: The CodeAnalyzer class might benefit from some optimization, especially for large codebases.
Security: Ensure that user inputs are properly sanitized, especially when dealing with file paths.
Extensibility: Consider implementing a plugin system for easy addition of new analysis techniques or models.

Overall, this is a well-structured and feature-rich C++ analyzer with good use of modern Python features and external libraries. The combination of static analysis and machine learning models provides a powerful tool for code analysis and understanding.

