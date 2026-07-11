import os
import subprocess
import sys

def run_experiments():
    experiments = [
        ("Table 3 (CIFAR-10 AlexNet)", "run_table3.py"),
        ("Table 5 (CIFAR-10 VGG-19)", "run_table5.py"),
        ("Table 6 (CIFAR-10 ResNet-18)", "run_table6.py"),
        ("Table 7 (Domain Adaptation)", "run_table7.py"),
        ("Table 8 (Continual Learning)", "run_table8.py"),
        ("Table 10 (UAPs)", "run_table10.py")
    ]
    
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    device2_log_path = os.path.join(log_dir, "device2_runs.log")
    
    with open(device2_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("="*70 + "\n")
        log_file.write("  DATA IMPRESSIONS - DEVICE 2 RUNS LOG (OPTIMIZED PATH)\n")
        log_file.write("="*70 + "\n\n")
        
    print(f"Device 2 Master log is located at: {device2_log_path}")
    
    for name, script in experiments:
        if not os.path.exists(script):
            print(f"Skipping {name} (script {script} not found)")
            continue
            
        print(f"\n>>> Running {name} ({script})...")
        with open(device2_log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n" + "="*60 + "\n")
            log_file.write(f"  STARTING {name} ({script})\n")
            log_file.write("="*60 + "\n\n")
            log_file.flush()
            
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            with open(device2_log_path, "a", encoding="utf-8") as log_file:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
                    
            process.wait()
            
            status_str = "SUCCESS" if process.returncode == 0 else f"FAILED (code {process.returncode})"
            print(f"Finished {name} with status: {status_str}")
            
            with open(device2_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"\n>>> FINISHED {name} WITH STATUS: {status_str}\n")
                log_file.flush()
                
        except Exception as e:
            print(f"Error running {name}: {str(e)}")
            with open(device2_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"ERROR executing {script}: {str(e)}\n")
                log_file.flush()

if __name__ == "__main__":
    run_experiments()
