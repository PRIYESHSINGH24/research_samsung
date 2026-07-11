import os
import subprocess
import sys

# ---- 1. PATCH FILES ----
def fix_similarity_matrix(content):
    # Replaces global normalization with row-wise normalization
    old_code_1 = """    sim = F.cosine_similarity(weights.unsqueeze(1), weights.unsqueeze(0), dim=2)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)"""
    new_code_1 = """    sim = F.cosine_similarity(weights.unsqueeze(1), weights.unsqueeze(0), dim=2)
    row_min = sim.min(dim=1, keepdim=True).values
    row_max = sim.max(dim=1, keepdim=True).values
    sim = (sim - row_min) / (row_max - row_min + 1e-8)"""

    old_code_2 = """    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    return ((s - s.min()) / (s.max() - s.min() + 1e-8)).cpu().numpy()"""
    new_code_2 = """    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    row_min = s.min(dim=1, keepdim=True).values
    row_max = s.max(dim=1, keepdim=True).values
    s = (s - row_min) / (row_max - row_min + 1e-8)
    return s.cpu().numpy()"""
    
    old_code_3 = """    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)"""
    new_code_3 = """    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    row_min = s.min(dim=1, keepdim=True).values
    row_max = s.max(dim=1, keepdim=True).values
    s = (s - row_min) / (row_max - row_min + 1e-8)"""

    old_code_4 = """    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    return ((s-s.min())/(s.max()-s.min()+1e-8)).cpu().numpy()"""
    new_code_4 = """    s = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=2)
    row_min = s.min(dim=1, keepdim=True).values
    row_max = s.max(dim=1, keepdim=True).values
    s = (s - row_min) / (row_max - row_min + 1e-8)
    return s.cpu().numpy()"""

    content = content.replace(old_code_1, new_code_1)
    content = content.replace(old_code_2, new_code_2)
    content = content.replace(old_code_3, new_code_3)
    content = content.replace(old_code_4, new_code_4)
    return content

def fix_mnist_clamping(content):
    # run_table1.py and run_table4.py mnist clamp bounds
    old_clamp = "with torch.no_grad():\n            di.clamp_(-2.5, 2.5)"
    new_clamp = "with torch.no_grad():\n            di.clamp_(-0.4242, 2.8215)"
    content = content.replace(old_clamp, new_clamp)
    return content

def fix_fmnist_clamping(content):
    # run_table2.py fmnist clamp bounds
    old_clamp = "with torch.no_grad():\n            di.clamp_(-2.5, 2.5)"
    new_clamp = "with torch.no_grad():\n            di.clamp_(-0.8102, 2.0227)"
    content = content.replace(old_clamp, new_clamp)
    return content

def fix_cifar_clamping(content):
    # run_table3.py / run_table6.py channel-wise clamping
    old_setup = """def generate_di_batch(model, targets, device):\n    B = targets.shape[0]\n    di = torch.randn(B, 3, 32, 32, device=device)\n    di.requires_grad_(True)\n    targets = targets.to(device)\n    optimizer = torch.optim.Adam([di], lr=0.1)\n    \n    for _ in range(1500):\n        optimizer.zero_grad()\n        logits = model(di, temperature=20)\n        pred = F.softmax(logits, dim=1)\n        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))\n        loss.backward()\n        optimizer.step()\n        with torch.no_grad():\n            di.clamp_(-2.5, 2.5)\n    return di.detach()"""
    
    new_setup = """_CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
_CLAMP_MIN = ((0.0 - _CIFAR_MEAN) / _CIFAR_STD)
_CLAMP_MAX = ((1.0 - _CIFAR_MEAN) / _CIFAR_STD)

def generate_di_batch(model, targets, device):
    B = targets.shape[0]
    cmin, cmax = _CLAMP_MIN.to(device), _CLAMP_MAX.to(device)
    di = torch.randn(B, 3, 32, 32, device=device)
    with torch.no_grad():
        di = torch.max(torch.min(di, cmax), cmin)
    di.requires_grad_(True)
    targets = targets.to(device)
    optimizer = torch.optim.Adam([di], lr=0.1)
    
    for _ in range(1500):
        optimizer.zero_grad()
        logits = model(di, temperature=20)
        pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            di.data = torch.max(torch.min(di.data, cmax), cmin)
    return di.detach()"""
    
    content = content.replace(old_setup, new_setup)
    
    # Also for run_table6.py
    old_gen_di = """def gen_di(model, targets, dev):\n    B = targets.shape[0]; di = torch.randn(B,3,32,32,device=dev); di.requires_grad_(True)\n    targets = targets.to(dev); opt = torch.optim.Adam([di], lr=0.001)\n    for _ in range(1500):\n        opt.zero_grad(); logits = model(di, temperature=20); pred = F.softmax(logits, dim=1)\n        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))\n        loss.backward(); opt.step()\n        with torch.no_grad(): di.clamp_(-2.5, 2.5)\n    return di.detach()"""
    
    new_gen_di = """_CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_CIFAR_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
_CLAMP_MIN = ((0.0 - _CIFAR_MEAN) / _CIFAR_STD)
_CLAMP_MAX = ((1.0 - _CIFAR_MEAN) / _CIFAR_STD)

def gen_di(model, targets, dev):
    B = targets.shape[0]
    cmin, cmax = _CLAMP_MIN.to(dev), _CLAMP_MAX.to(dev)
    di = torch.randn(B, 3, 32, 32, device=dev)
    with torch.no_grad():
        di = torch.max(torch.min(di, cmax), cmin)
    di.requires_grad_(True)
    targets = targets.to(dev); opt = torch.optim.Adam([di], lr=0.001)
    for _ in range(1500):
        opt.zero_grad(); logits = model(di, temperature=20); pred = F.softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(targets * torch.log(pred + 1e-8), dim=1))
        loss.backward(); opt.step()
        with torch.no_grad(): di.data = torch.max(torch.min(di.data, cmax), cmin)
    return di.detach()"""
    
    content = content.replace(old_gen_di, new_gen_di)
    return content

def fix_mnist_teacher_load(content):
    old_code = """    # ── 1. Teacher-CE ──
    print("\\n  --- Teacher-CE ---")
    teacher = LeNet5().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    best_teacher = 0
    
    for epoch in range(1, 101):"""
    
    new_code = """    # ── 1. Teacher-CE ──
    print("\\n  --- Teacher-CE ---")
    teacher = LeNet5().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    best_teacher = 0
    
    if os.path.exists(f'{config.CHECKPOINT_DIR}mnist_teacher.pth'):
        teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}mnist_teacher.pth', map_location=config.DEVICE))
        teacher.eval()
        best_teacher = evaluate(teacher, test_loader)
        print(f"  Loaded saved Teacher model. Acc: {best_teacher:.2f}%")
        epochs_range = []
    else:
        epochs_range = range(1, 101)
        
    for epoch in epochs_range:"""
    
    content = content.replace(old_code, new_code)
    return content

def fix_fmnist_teacher_load(content):
    old_code = """    # ── 1. Teacher-CE ──
    print("\\n  --- Teacher-CE ---")
    teacher = LeNet5().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    best_teacher = 0
    
    for epoch in range(1, 101):"""
    
    new_code = """    # ── 1. Teacher-CE ──
    print("\\n  --- Teacher-CE ---")
    teacher = LeNet5().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    best_teacher = 0
    
    if os.path.exists(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth'):
        teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}fmnist_teacher.pth', map_location=config.DEVICE))
        teacher.eval()
        best_teacher = evaluate(teacher, test_loader)
        print(f"  Loaded saved FMNIST Teacher model. Acc: {best_teacher:.2f}%")
        epochs_range = []
    else:
        epochs_range = range(1, 101)
        
    for epoch in epochs_range:"""
    
    content = content.replace(old_code, new_code)
    return content

def fix_cifar_teacher_load(content):
    old_code = """    # ── 1. Teacher-CE (AlexNet) ──
    print("\\n  --- AlexNet Teacher-CE ---")
    teacher = AlexNet().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    best_teacher = 0
    
    for epoch in range(1, 101):"""
    
    new_code = """    # ── 1. Teacher-CE (AlexNet) ──
    print("\\n  --- AlexNet Teacher-CE ---")
    teacher = AlexNet().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    best_teacher = 0
    
    if os.path.exists(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth'):
        teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}cifar_alexnet_teacher.pth', map_location=config.DEVICE))
        teacher.eval()
        best_teacher = evaluate(teacher, test_loader)
        print(f"  Loaded saved CIFAR Teacher model. Acc: {best_teacher:.2f}%")
        epochs_range = []
    else:
        epochs_range = range(1, 101)
        
    for epoch in epochs_range:"""
    
    content = content.replace(old_code, new_code)
    
    # Also for run_table6.py
    old_t_code_6 = """    # 1. Teacher (Paper: lr=0.01, batch=512, 500ep)
    print("\\n  --- ResNet-18 Teacher ---")
    train_loader, test_loader = get_cifar_loaders(512)
    teacher = ResNet18().to(config.DEVICE)
    opt = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[250,375], gamma=0.1)
    best_t = 0
    for ep in range(1, 501):"""
    
    new_t_code_6 = """    # 1. Teacher (Paper: lr=0.01, batch=512, 500ep)
    print("\\n  --- ResNet-18 Teacher ---")
    train_loader, test_loader = get_cifar_loaders(512)
    teacher = ResNet18().to(config.DEVICE)
    opt = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[250,375], gamma=0.1)
    best_t = 0
    
    if os.path.exists(f'{config.CHECKPOINT_DIR}resnet18_cifar_teacher.pth'):
        teacher.load_state_dict(torch.load(f'{config.CHECKPOINT_DIR}resnet18_cifar_teacher.pth', map_location=config.DEVICE))
        teacher.eval()
        best_t = evaluate(teacher, test_loader)
        print(f"  Loaded saved ResNet-18 Teacher model. Acc: {best_t:.2f}%")
        epochs_range = []
    else:
        epochs_range = range(1, 501)
        
    for ep in epochs_range:"""
    
    content = content.replace(old_t_code_6, new_t_code_6)
    return content

def fix_run_table5_lr(content):
    # Fix the learning rate of VGG-19 DI generation to 0.1 to avoid Standard Deviation explosion
    content = content.replace("optimizer = torch.optim.Adam([di], lr=10.0)", "optimizer = torch.optim.Adam([di], lr=0.1)")
    content = content.replace("lr=10.0", "lr=0.1")
    return content

def apply_fixes():
    print("Patching files with row-wise similarity and correct clamping...")
    files = [
        "run_table1.py",
        "run_table2.py",
        "run_table3.py",
        "run_table4.py",
        "run_table5.py",
        "run_table6.py",
        "run_table7.py",
        "run_table8.py",
        "run_table10.py"
    ]
    
    for f in files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
        with open(f, "r") as file_in:
            content = file_in.read()
            
        content = fix_similarity_matrix(content)
        
        if f == "run_table1.py":
            content = fix_mnist_clamping(content)
            content = fix_mnist_teacher_load(content)
        elif f == "run_table2.py":
            content = fix_fmnist_clamping(content)
            content = fix_fmnist_teacher_load(content)
        elif f == "run_table3.py":
            content = fix_cifar_clamping(content)
            content = fix_cifar_teacher_load(content)
        elif f == "run_table4.py":
            content = fix_mnist_clamping(content)
        elif f == "run_table5.py":
            content = fix_run_table5_lr(content)
        elif f == "run_table6.py":
            content = fix_cifar_clamping(content)
            content = fix_cifar_teacher_load(content)
            
        with open(f, "w") as file_out:
            file_out.write(content)
    print("All files patched successfully!")

# ---- 2. RUN EXPERIMENTS SEQUENTIALLY ----
def run_experiments():
    experiments = [
        ("Table 1 (MNIST)", "run_table1.py"),
        ("Table 2 (Fashion-MNIST)", "run_table2.py"),
        ("Table 3 (CIFAR-10 AlexNet)", "run_table3.py"),
        ("Table 4 (Adversarial Robustness)", "run_table4.py"),
        ("Table 5 (CIFAR-10 VGG-19)", "run_table5.py"),
        ("Table 6 (CIFAR-10 ResNet-18)", "run_table6.py"),
        ("Table 7 (Domain Adaptation)", "run_table7.py"),
        ("Table 8 (Continual Learning)", "run_table8.py"),
        ("Table 10 (UAPs)", "run_table10.py")
    ]
    
    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    master_log_path = os.path.join(log_dir, "overnight_runs.log")
    
    with open(master_log_path, "w", encoding="utf-8") as master_log:
        master_log.write("="*70 + "\n")
        master_log.write("  DATA IMPRESSIONS OVERNIGHT EXPERIMENTS MASTER LOG\n")
        master_log.write("="*70 + "\n\n")
        
    print(f"Master log is located at: {master_log_path}")
    
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
            # Set PYTHONPATH to current directory so modules load correctly
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["PYTHONUNBUFFERED"] = "1"
            
            # Execute script and redirect stdout/stderr in real-time
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
    apply_fixes()
    run_experiments()
