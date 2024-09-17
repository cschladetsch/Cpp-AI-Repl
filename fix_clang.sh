#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Python
echo "Checking Python installation..."
if command_exists python3; then
    python_cmd="python3"
elif command_exists python; then
    python_cmd="python"
else
    echo "Python not found. Please install Python 3 and try again."
    exit 1
fi

# Check for clang version
echo "Checking installed clang version..."
if command_exists clang; then
    clang_version=$(clang --version | grep -oP '(?<=clang version )\d+')
    echo "Clang installed: version $clang_version"
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
    clang_version=$(clang --version | grep -oP '(?<=clang version )\d+')
fi

# Locate libclang.so for the installed version
echo "Looking for libclang.so..."
libclang_path=$(find /usr -name "libclang-$clang_version.so*" 2>/dev/null | head -n 1)

if [ -z "$libclang_path" ]; then
    echo "libclang.so not found. Installing libclang..."
    if command_exists apt-get; then
        sudo apt-get install libclang-$clang_version-dev -y
    elif command_exists brew; then
        brew install llvm
    else
        echo "Package manager not found. Please install libclang manually."
        exit 1
    fi
    libclang_path=$(find /usr -name "libclang-$clang_version.so*" 2>/dev/null | head -n 1)
fi

echo "Found libclang at: $libclang_path"

# Set LIBCLANG_PATH environment variable
echo "Setting LIBCLANG_PATH..."
export LIBCLANG_PATH="$libclang_path"
echo "LIBCLANG_PATH set to: $LIBCLANG_PATH"

# Add the line to .bashrc to make it persistent
echo "export LIBCLANG_PATH=$libclang_path" >> ~/.bashrc

# Uninstall and reinstall libclang Python bindings
echo "Reinstalling Python clang bindings..."
$python_cmd -m pip uninstall -y libclang
$python_cmd -m pip install libclang==$clang_version.*

# Verify that the correct libclang is used in Python
$python_cmd -c "
import clang.cindex
clang.cindex.Config.set_library_file('$libclang_path')
try:
    print(f'Using libclang from: {clang.cindex.Config.library_file}')
    index = clang.cindex.Index.create()
    print('Successfully created Clang Index')
except Exception as e:
    print(f'Error loading libclang: {e}')
"

# Success message
echo "Setup completed. libclang and clang bindings should now be compatible."
echo "Please restart your terminal or run 'source ~/.bashrc' to apply the changes."
