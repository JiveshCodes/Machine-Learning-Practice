import os
import sys
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path):
    print(f"Executing: {notebook_path} ... ", end="", flush=True)
    
    # Extract directory and filename
    notebook_dir = os.path.dirname(notebook_path)
    notebook_name = os.path.basename(notebook_path)
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure execution preprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    
    try:
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        
        # Save the executed notebook back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print("SUCCESS ✅")
    except Exception as e:
        print("FAILED ❌")
        print(f"Error executing {notebook_name}: {str(e)}")

def main():
    print("="*60)
    print("Machine Learning Practice - Auto Execution Script")
    print("="*60)
    
    # Find all .ipynb files recursively in the current directory (excluding virtual environments)
    notebook_patterns = os.path.join("**", "*.ipynb")
    all_notebooks = glob.glob(notebook_patterns, recursive=True)
    
    # Filter out checkpoints and venv notebooks
    target_notebooks = []
    for nb in all_notebooks:
        if ".ipynb_checkpoints" in nb or "venv" in nb:
            continue
        if "Linear_Regression.ipynb" in nb and "06_Decision_Tree" in nb:
            continue
        target_notebooks.append(nb)
        
    target_notebooks = sorted(target_notebooks)
    
    if not target_notebooks:
        print("No notebooks found to execute!")
        return
        
    print(f"Found {len(target_notebooks)} notebooks to execute.")
    
    try:
        import nbconvert
    except ImportError:
        print("\nERROR: 'nbconvert' package is not installed.")
        print("Please install requirements first using:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    for index, nb_path in enumerate(target_notebooks, 1):
        print(f"[{index}/{len(target_notebooks)}] ", end="")
        run_notebook(nb_path)
        
    print("\n" + "="*60)
    print("All notebooks processed! Open them in Jupyter/VS Code to view outcomes.")
    print("="*60)

if __name__ == "__main__":
    main()
