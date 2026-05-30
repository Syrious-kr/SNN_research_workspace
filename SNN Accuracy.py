import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, layer, functional, surrogate

class HfO2_Device:
    NL_LTP = 2.4

# 해상도와 레이어 스케일이 독립 제어되는 HAT 레이어
class RRAM_Linear_HAT(nn.Linear):
    def __init__(self, in_features, out_features, res_steps=100):
        super().__init__(in_features, out_features, bias=False)
        self.res_steps = res_steps
        
        nn.init.uniform_(self.weight, -0.8, 0.8)
        self.dynamic_scale = np.sqrt(2.0 / in_features) * 4.5

        import math
        self.norm_factor = 1.0 - math.exp(-HfO2_Device.NL_LTP)

    def apply_physics(self, w):
        w_clipped = torch.clamp(w, -1.0, 1.0)
        
        w_pos = torch.clamp(w_clipped, min=0)
        w_neg = torch.clamp(-w_clipped, min=0)
        
        w_pos_q = w_pos + (torch.round(w_pos * self.res_steps) / self.res_steps - w_pos).detach()
        w_neg_q = w_neg + (torch.round(w_neg * self.res_steps) / self.res_steps - w_neg).detach()
        
        w_pos_phys = (1.0 - torch.exp(-HfO2_Device.NL_LTP * w_pos_q)) / self.norm_factor
        w_neg_phys = (1.0 - torch.exp(-HfO2_Device.NL_LTP * w_neg_q)) / self.norm_factor
        
        return (w_pos_phys - w_neg_phys) * self.dynamic_scale

    def forward(self, x):
        p_weight = self.apply_physics(self.weight)
        return nn.functional.linear(x, p_weight)

# SNN 모델 정의
class HfO2_Quant_SNN(nn.Module):
    def __init__(self, res_steps):
        super().__init__()
        self.model = nn.Sequential(
            layer.Flatten(),
            RRAM_Linear_HAT(784, 128, res_steps=res_steps),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True),
            RRAM_Linear_HAT(128, 10, res_steps=res_steps),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True)
        )
    def forward(self, x): return self.model(x)

num_operator = 30

# 해상도별 정량 실험 제어 함수
def run_resolution_analysis(res_list):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=128)

    all_results = {}

    for res in res_list:
        print(f"\n--- 분석 시작: Resolution {res} Steps ---")
        model = HfO2_Quant_SNN(res_steps=res).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        for epoch in range(num_operator):
            model.train()
            for img, label in train_loader:
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()
                
                out_fr = 0
                for t in range(8): out_fr += model(img)
                loss = criterion(out_fr / 8, label)
                loss.backward()
                
                optimizer.step()
                functional.reset_net(model)
            
            # 평가
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for img, label in test_loader:
                    img, label = img.to(device), label.to(device)
                    out_fr = 0
                    for t in range(8): out_fr += model(img)
                    correct += (out_fr.argmax(1) == label).sum().item()
                    total += label.size(0)
                    functional.reset_net(model)
            
            acc = correct / total
            history.append(acc)
            print(f"Epoch {epoch+1}: 정확도 = {acc:.4f}")
        
        all_results[res] = history
    return all_results

# 최종 실험 제어 (8, 16, 64, 256 전 영역 완전 통제 비교)
resolutions = [8, 16, 64, 256]
analysis_data = run_resolution_analysis(resolutions)

plt.figure(figsize=(10, 6))
for res, history in analysis_data.items():
    plt.plot(range(1, num_operator+1), history, marker='o', label=f'Resolution: {res} steps')
plt.title('Quantitative Impact of RRAM Resolution on SNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()