# desc/dependency_installer.py
import importlib
import subprocess
import sys

def install_and_import(package, import_name=None):
    import_name = import_name or package
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {package}...")
        if package == 'openbabel':
            print("Open Babel requires manual installation. Please ensure it is installed correctly.")
            print("Refer to: http://openbabel.org/wiki/Get_Open_Babel")
            raise ImportError("Open Babel is not installed or not found by Python.")
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            importlib.import_module(import_name)
