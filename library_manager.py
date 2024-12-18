import subprocess
import sys

def import_or_install(libraries):
    print('in installer')
    for lib in libraries:
        try:
            __import__(lib)
        except ImportError:
            print(f"'{lib}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])