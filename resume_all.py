import os
import subprocess
import sys

def run_resume_experiments():
    # Bypassing Table 5, Table 7, and Table 10 since they are fully completed.
    # Running only the optimized Table 3.
    experiments = [
        ("Table 3 (CIFAR-10 AlexNet)", "run_table3.py")
    ]
    
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    master_log_path = os.path.join(log_dir, "overnight_runs.log")
    
    with open(master_log_path, "a", encoding="utf-8") as master_log:
        master_log.write("\n" + "="*70 + "\n")
        master_log.write("  RESUMING PIPELINE FOR TABLE 3 ONLY (OTHERS COMPLETED)\n")
        master_log.write("="*70 + "\n\n")
        master_log.flush()
        
    print(f"Resuming experiments. Log appending to: {master_log_path}")
    
    for name, script in experiments:
        if not os.path.exists(script):
            print(f"Skipping {name} (script {script} not found)")
            continue
            
        print(f"\n>>> Running {name} ({script})...")
        with open(master_log_path, "a", encoding="utf-8") as master_log:
            master_log.write("\n" + "="*60 + "\n")
            master_log.write(f"  STARTING {name} ({script})\n")
            master_log.write("="*60 + "\n\n")
            master_log.flush()
            
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
            
            with open(master_log_path, "a", encoding="utf-8") as master_log:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    master_log.write(line)
                    master_log.flush()
                    
            process.wait()
            
            status_str = "SUCCESS" if process.returncode == 0 else f"FAILED (code {process.returncode})"
            print(f"Finished {name} with status: {status_str}")
            
            with open(master_log_path, "a", encoding="utf-8") as master_log:
                master_log.write(f"\n>>> FINISHED {name} WITH STATUS: {status_str}\n")
                master_log.flush()
                
        except Exception as e:
            print(f"Error running {name}: {str(e)}")
            with open(master_log_path, "a", encoding="utf-8") as master_log:
                master_log.write(f"ERROR executing {script}: {str(e)}\n")
                master_log.flush()

if __name__ == "__main__":
    run_resume_experiments()
