import torch, math, random, numpy as np, matplotlib.pyplot as plt
import torch.nn as nn; from spikingjelly.activation_based import neuron, layer, functional, surrogate
from torchvision import datasets, transforms; from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class HfO2_Device: NL_LTP = 2.4; NOISE_SIGMA = 0.05
class RRAM_Linear_Universal(nn.Linear):
    def __init__(self, in_features, out_features, res_steps=100, mode='HfO2_Physics'):
        super().__init__(in_features, out_features, bias=False)
        self.res_steps = res_steps; self.mode = mode
        nn.init.uniform_(self.weight, -0.8, 0.8)
        self.dynamic_scale = np.sqrt(2.0 / in_features) * 4.5
        self.norm_factor = 1.0 - math.exp(-HfO2_Device.NL_LTP)
    def apply_physics(self, w):
        if self.mode == 'Ideal_FP32': return w * self.dynamic_scale
        w_clipped = torch.clamp(w, -1.0, 1.0)
        w_pos, w_neg = torch.clamp(w_clipped, min=0), torch.clamp(-w_clipped, min=0)
        w_pos_q = w_pos + (torch.round(w_pos * self.res_steps) / self.res_steps - w_pos).detach()
        w_neg_q = w_neg + (torch.round(w_neg * self.res_steps) / self.res_steps - w_neg).detach()
        if self.mode == 'Linear_Quant': return (w_pos_q - w_neg_q) * self.dynamic_scale
        w_pos_phys = (1.0 - torch.exp(-HfO2_Device.NL_LTP * w_pos_q)) / self.norm_factor
        w_neg_phys = (1.0 - torch.exp(-HfO2_Device.NL_LTP * w_neg_q)) / self.norm_factor
        if self.mode == 'Noise_Aware' and self.training:
            w_pos_phys = torch.clamp(w_pos_phys + torch.randn_like(w_pos_phys) * HfO2_Device.NOISE_SIGMA, 0.0, 1.0)
            w_neg_phys = torch.clamp(w_neg_phys + torch.randn_like(w_neg_phys) * HfO2_Device.NOISE_SIGMA, 0.0, 1.0)
        return (w_pos_phys - w_neg_phys) * self.dynamic_scale
    def forward(self, x): return nn.functional.linear(x, self.apply_physics(self.weight))

class Universal_SNN(nn.Module):
    def __init__(self, res_steps, mode):
        super().__init__()
        self.model = nn.Sequential(layer.Flatten(), RRAM_Linear_Universal(784, 128, res_steps, mode), neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True), RRAM_Linear_Universal(128, 10, res_steps, mode), neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid(), detach_reset=True))
    def forward(self, x): return self.model(x)

class EaPU_OptimizerWrapper:
    def __init__(self, optimizer, dw_th=1e-5):
        self.optimizer = optimizer
        self.dw_th = dw_th

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        old_weights = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    old_weights[p] = p.data.clone()
                    
        self.optimizer.step()
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    delta_w = p.data - old_weights[p]
                    abs_dw = torch.abs(delta_w)
                    
                    mask_large = (abs_dw >= self.dw_th)
                    mask_small = (abs_dw < self.dw_th)
                    
                    prob = (abs_dw / self.dw_th)
                    random_mask = (torch.rand_like(delta_w) < prob)
                    
                    dw_small_update = torch.sign(delta_w) * self.dw_th * random_mask
                    dw_n = (delta_w * mask_large) + (dw_small_update * mask_small)
                    
                    p.data = old_weights[p] + dw_n

def run_ablation_study(modes, res_steps, seeds, epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=1024, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)
    final_results = {}
    for mode in modes:
        mode_history = {e: [] for e in range(epochs)}
        for seed in seeds:
            set_seed(seed); model = Universal_SNN(res_steps=res_steps, mode=mode).to(device)
            base_optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            optimizer = EaPU_OptimizerWrapper(base_optimizer, dw_th=1e-4)
            criterion = nn.CrossEntropyLoss() 
            
            for epoch in range(epochs):
                model.train()
                for img, label in train_loader:
                    img, label = img.to(device), label.to(device); optimizer.zero_grad()
                    out_fr = sum(model(img) for _ in range(8))
                    loss = criterion(out_fr / 8, label); loss.backward(); optimizer.step(); functional.reset_net(model)
                model.eval(); correct, total = 0, 0
                with torch.no_grad():
                    for img, label in test_loader:
                        img, label = img.to(device), label.to(device)
                        out_fr = sum(model(img) for _ in range(8))
                        correct += (out_fr.argmax(1) == label).sum().item(); total += label.size(0); functional.reset_net(model)
                print(f"[{mode} | Seed {seed}] Epoch {epoch+1}/{epochs} - Acc: {correct/total:.4f}")
                mode_history[epoch].append(correct/total)
        final_results[mode] = mode_history
    return final_results

target_resolution = 16; baselines = ['Ideal_FP32', 'Linear_Quant', 'HfO2_Physics', 'Noise_Aware']
ablation_data = run_ablation_study(baselines, target_resolution, [0, 1, 2, 3, 4], epochs=30)
plt.figure(figsize=(12, 7))
for mode in baselines:
    epochs_range = list(ablation_data[mode].keys()); means = [np.mean(ablation_data[mode][e]) for e in epochs_range]; stds = [np.std(ablation_data[mode][e]) for e in epochs_range]
    plt.plot(epochs_range, means, label=mode, linewidth=2)
    plt.fill_between(epochs_range, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
plt.title(f'Ablation Study (Batch 1024 & EaPU, Res: {target_resolution}, 5 Seeds)'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.savefig('ablation_study__EaPU.png')
