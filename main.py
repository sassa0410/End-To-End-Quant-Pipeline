import subprocess
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_script(script_path: str):
    print(f"\n\n{'='*60}")
    logging.info(f"Running execution step: {script_path}...")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running {script_path}: Pipeline aborted.")
        sys.exit(1)
        
def main():
    logging.info("Starting End-to-End Quant Pipeline...")
    start_time = time.time()
    
    pipeline_scripts = [
        "src/data_engineering/build_features.py",
        "src/ml_models/train_model.py",
        "src/optimization/portfolio_optimization.py",
        "src/reports/generate_report.py"
    ]
    
    for script in pipeline_scripts:
        run_script(script)
        
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n{'='*60}\n")
    logging.info(f"PIPELINE COMPLETED WITH SUCCESS in {elapsed:.1f} seconds!")
    logging.info("Execution plan and the generated PDF report are available in the 'reports' directory.")
    print(f"\n{'='*60}\n")
    
    
if __name__ == "__main__":
    main()