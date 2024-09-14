import sys
import time
import threading
from colorama import Fore, Style, init

init(autoreset=True)

class Spinner:
    def __init__(self):
        self.spinner_chars = "|/-\\"
        self.is_active = False
        self.counter = 0
        self.start_time = None

    def spin(self):
        while self.is_active:
            elapsed_time = time.time() - self.start_time
            sys.stdout.write(f"\r{Fore.CYAN}Thinking{Style.RESET_ALL} {self.spinner_chars[self.counter % len(self.spinner_chars)]} ({elapsed_time:.1f}s)")
            sys.stdout.flush()
            time.sleep(0.1)
            self.counter += 1

    def start(self):
        self.is_active = True
        self.start_time = time.time()
        threading.Thread(target=self.spin, daemon=True).start()

    def stop(self):
        self.is_active = False
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear the spinner line
        sys.stdout.flush()
