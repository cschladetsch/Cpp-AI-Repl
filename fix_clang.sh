#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for clang version
echo "Checking installed clang version..."
if command_exists clang; then
    clang_version=$(clang --version | head -n 1)
    echo "Clang installed: $clang_version"
else
    echo "Clang is not installed. Installing clang..."
    if command_exists apt-get; then
        sudo apt-get update && sudo apt-get install clang -y
    elif command_exists brew; then
        brew install llvm
    else
        echo "Package manager not found. Please install clang manually."
        exit 1
    fi
fi

# Locate libclang.so
echo "Looking for libclang.so..."
libclang_path=$(find /usr -name "libclang.so*" 2>/dev/null | head -n 1)

if [ -z "$libclang_path" ]; then
    echo "libclang.so not found. Installing libclang..."
    if command_exists apt-get; then
        sudo apt-get install libclang-dev -y
    elif command_exists brew; then
        brew install llvm
    else
        echo "Package manager not found. Please install libclang manually."
        exit 1
    fi
    libclang_path=$(find /usr -name "libclang.so*" 2>/dev/null | head -n 1)
else
    echo "Found libclang at: $libclang_path"
fi

# Set LIBCLANG_PATH environment variable
echo "Setting LIBCLANG_PATH..."
export LIBCLANG_PATH="$libclang_path"
echo "LIBCLANG_PATH set to: $LIBCLANG_PATH"

# Uninstall and reinstall libclang Python bindings
echo "Reinstalling Python clang bindings..."
pip uninstall -y libclang
pip install clang

# Verify that the correct libclang is used in Python
python -c "
import clang.cindex
try:
    print(f'Using libclang from: {clang.cindex.Config.library_file}')
except Exception as e:
    print(f'Error loading libclang: {e}')
"

# Success message
echo "Setup completed. libclang and clang bindings should now be compatible."

