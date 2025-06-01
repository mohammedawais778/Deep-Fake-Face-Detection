import subprocess
import sys
import os

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()

def main():
    # Create virtual environment
    print("Creating virtual environment...")
    venv_path = os.path.join(os.getcwd(), "venv38")
    run_command(f"py -3.8 -m venv {venv_path}")
    
    # Activate virtual environment and install packages
    print("Installing required packages...")
    pip_path = os.path.join(venv_path, "Scripts", "pip")
    python_path = os.path.join(venv_path, "Scripts", "python")
    
    # Install packages
    packages = [
        "tensorflow==2.4.1",
        "keras==2.4.3",
        "numpy==1.19.5",
        "opencv-python==4.5.3.56",
        "scikit-learn==0.24.2",
        "Pillow==8.3.1",
        "pandas==1.3.0",
        "tqdm==4.61.2",
        "Flask==2.0.1",
        "Werkzeug==2.0.1",
        "pathlib2==2.3.6",
        "argparse==1.4.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        run_command(f'"{pip_path}" install {package}')
    
    print("\nEnvironment setup complete!")
    print(f"\nTo activate the environment, run:")
    print(f'"{venv_path}\\Scripts\\activate"')

if __name__ == "__main__":
    main() 