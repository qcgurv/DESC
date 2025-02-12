import importlib
import subprocess
import sys

def install_and_import(package, import_name=None):
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
    except ImportError:
        if package == 'openbabel':
            print("Installing openbabel-wheel...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openbabel-wheel"])
            try:
                importlib.import_module("openbabel")
            except ImportError:
                print("Failed to import 'openbabel' after installing 'openbabel-wheel'.")
                raise
        else:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            try:
                importlib.import_module(import_name)
            except ImportError:
                print(f"Failed to import '{import_name}' after installation.")
                raise
