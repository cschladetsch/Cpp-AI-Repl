import os
import ctypes
import glob
from colorama import Fore, Style
import warnings

def setup_environment():
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Define and create cache directories
    base_cache_dir = os.path.expanduser('~/.cache')
    cpp_analyzer_cache_dir = os.path.join(base_cache_dir, 'cpp_analyzer')
    model_cache_dir = os.path.join(cpp_analyzer_cache_dir, 'models')
    os.makedirs(model_cache_dir, exist_ok=True)

    # Setup Clang library
    if not setup_clang_library():
        print(f"{Fore.RED}Failed to setup Clang library. Exiting.{Style.RESET_ALL}")
        exit(1)

def setup_clang_library():
    libclang_path = os.environ.get('LIBCLANG_PATH')
    if libclang_path and os.path.exists(libclang_path):
        return load_clang(libclang_path)

    libclang = ctypes.util.find_library("clang")
    if libclang:
        return load_clang(libclang)

    search_paths = ['/usr/lib', '/usr/local/lib', '/usr/local/opt/llvm/lib', '/opt/homebrew/opt/llvm/lib']
    for path in search_paths:
        matches = glob.glob(os.path.join(path, 'libclang*'))
        if matches:
            return load_clang(matches[0])

    print(f"{Fore.RED}Failed to find libclang.{Style.RESET_ALL}")
    return False

def load_clang(libclang_path):
    try:
        import clang.cindex
        clang.cindex.Config.set_library_file(libclang_path)
        return True
    except ImportError:
        print(f"{Fore.YELLOW}Install libclang with 'pip install libclang'.{Style.RESET_ALL}")
        return False
