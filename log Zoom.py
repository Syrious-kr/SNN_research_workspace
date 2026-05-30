import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

log_file_path = "log/output_t4_ep100.log"
target_epochs = 90
target_resolution = 16
baselines = ['Ideal_FP32', 'Linear_Quant', 'HfO2_Physics', 'Noise_Aware']

ablation_data = defaultdict(lambda: defaultdict(list))

log_pattern = re.compile(r"\[(.*?) \| Seed (\d+)\] Epoch (\d+)/\d+ - Acc: ([\d\.]+)")

with open(log_file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = log_pattern.search(line)
        if match:
            mode = match.group(1)
            epoch = int(match.group(3)) - 1 
            acc = float(match.group(4))
            
            if epoch < target_epochs:
                ablation_data[mode][epoch].append(acc)

plt.figure(figsize=(12, 7))

for mode in baselines:
    if mode not in ablation_data:
        continue
        

    epochs_range = sorted(list(ablation_data[mode].keys()))
    
    if not epochs_range:
        continue
        
    means = [np.mean(ablation_data[mode][e]) for e in epochs_range]
    stds = [np.std(ablation_data[mode][e]) for e in epochs_range]

    plt.plot(epochs_range, means, label=mode, linewidth=2)
    plt.fill_between(epochs_range, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

plt.title(f'Ablation Study Zoom-in (First {target_epochs} Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()

save_name = f'zoomed_ablation_{target_epochs}ep.png'
plt.show()
print(f"그래프 추출 완료: {save_name} 파일이 생성되었습니다.")