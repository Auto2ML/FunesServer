#!/usr/bin/env python3
"""
Funes Course - Module 1: Environment Setup
Automated setup script for the development environment
"""

import os
import subprocess
import sys
import platform
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(50)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*50}{Colors.RESET}\n")

def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def run_command(command, description=""):
    """Run a shell command and handle errors"""
    try:
        if description:
            print(f"Running: {description}")
        
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {command}")
        print_error(f"Error: {e.stderr}")
        return False, e.stderr

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_system_requirements():
    """Check system requirements"""
    print_header("CHECKING SYSTEM REQUIREMENTS")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check OS
    os_name = platform.system()
    print_success(f"Operating System: {os_name}")
    
    # Check for required commands
    required_commands = ['git', 'pip3']
    for cmd in required_commands:
        success, _ = run_command(f"which {cmd}", f"Checking for {cmd}")
        if success:
            print_success(f"{cmd} is available")
        else:
            print_error(f"{cmd} is not available. Please install it first.")
            return False
    
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    print_header("CREATING VIRTUAL ENVIRONMENT")
    
    venv_name = "funes-course-env"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_name):
        print_warning(f"Virtual environment '{venv_name}' already exists")
        return True, venv_name
    
    # Create virtual environment
    success, output = run_command(f"python3 -m venv {venv_name}", 
                                 "Creating virtual environment")
    if not success:
        return False, ""
    
    print_success(f"Virtual environment '{venv_name}' created")
    return True, venv_name

def install_dependencies(venv_name):
    """Install Python dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    # Path to pip in virtual environment
    if platform.system() == "Windows":
        pip_path = f"{venv_name}\\Scripts\\pip"
    else:
        pip_path = f"{venv_name}/bin/pip"
    
    # Install basic dependencies
    dependencies = [
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "psycopg2-binary>=2.9.5",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.6.0",  # For visualization exercises
        "jupyter>=1.0.0"      # For optional notebook exercises
    ]
    
    for dep in dependencies:
        success, _ = run_command(f"{pip_path} install {dep}", 
                               f"Installing {dep}")
        if success:
            print_success(f"{dep} installed")
        else:
            print_error(f"Failed to install {dep}")
            return False
    
    return True

def create_project_structure():
    """Create basic project structure"""
    print_header("CREATING PROJECT STRUCTURE")
    
    directories = [
        "funes-project",
        "funes-project/data",
        "funes-project/src",
        "funes-project/tests",
        "funes-project/docs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_success(f"Created directory: {directory}")
    
    # Create basic files
    files = {
        "funes-project/README.md": "# Funes Project\n\nBuilding Funes from scratch!\n",
        "funes-project/.gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
""",
        "funes-project/src/__init__.py": "# Funes source code",
        "funes-project/tests/__init__.py": "# Funes tests"
    }
    
    for file_path, content in files.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print_success(f"Created file: {file_path}")
    
    return True

def create_activation_script(venv_name):
    """Create a convenient activation script"""
    print_header("CREATING ACTIVATION SCRIPT")
    
    if platform.system() == "Windows":
        script_content = f"""@echo off
echo Activating Funes Course Environment...
call {venv_name}\\Scripts\\activate.bat
echo Environment activated! You can now run the course exercises.
echo.
echo To deactivate, run: deactivate
echo To run exercises, go to: cd exercises
cmd /k
"""
        script_name = "activate_funes.bat"
    else:
        script_content = f"""#!/bin/bash
echo "Activating Funes Course Environment..."
source {venv_name}/bin/activate
echo "Environment activated! You can now run the course exercises."
echo ""
echo "To deactivate, run: deactivate"
echo "To run exercises, go to: cd exercises"
exec bash
"""
        script_name = "activate_funes.sh"
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    if platform.system() != "Windows":
        os.chmod(script_name, 0o755)
    
    print_success(f"Created activation script: {script_name}")
    return script_name

def main():
    """Main setup function"""
    print_header("FUNES COURSE - MODULE 1 SETUP")
    print("This script will set up your development environment for the Funes course.")
    
    # Check system requirements
    if not check_system_requirements():
        print_error("System requirements not met. Please install missing components.")
        return False
    
    # Create virtual environment
    success, venv_name = create_virtual_environment()
    if not success:
        print_error("Failed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies(venv_name):
        print_error("Failed to install dependencies")
        return False
    
    # Create project structure
    if not create_project_structure():
        print_error("Failed to create project structure")
        return False
    
    # Create activation script
    script_name = create_activation_script(venv_name)
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    print_success("Your Funes development environment is ready!")
    print()
    print("Next steps:")
    print(f"1. Activate your environment: ./{script_name}")
    print("2. Run the setup test: python test_setup.py")
    print("3. Start with the exercises: cd exercises")
    print()
    print("Happy coding! 🧠")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
