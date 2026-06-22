import os
import sys
import subprocess

def main():
    # repo_root is repositories/kairo-scaffold
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_dir = os.path.join(repo_root, "kernel", "sidecar", ".venv")
    
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment at {venv_dir}...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    
    # Locate pip and python inside virtualenv
    if os.name == "nt":
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_dir, "bin", "python")
        
    print("Upgrading pip...")
    subprocess.check_call([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    
    print("Installing requirements.txt...")
    reqs_path = os.path.join(repo_root, "kernel", "sidecar", "requirements.txt")
    subprocess.check_call([python_path, "-m", "pip", "install", "-r", reqs_path])
    print("Virtualenv setup complete.")

if __name__ == "__main__":
    main()
