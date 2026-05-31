import os
import glob
import re
import numpy as np
from collections import defaultdict

def analyze_ep100_logs_per_file(log_dir="."):
    log_files = glob.glob(os.path.join(log_dir, "log/output_*_ep100.log"))
    
    if not log_files:
        print(f"'{log_dir}' 경로에 output_***_ep100.log 파일이 없습니다.")
        return

    print(f"총 {len(log_files)}개의 로그 파일을 분석합니다.\n")

    pattern = re.compile(r"\[(.*?)\s*\|\s*Seed\s*(\d+)\]\s*Epoch\s*(\d+)/\d+\s*-\s*Acc:\s*([\d\.]+)")
    target_modes = ['Ideal_FP32', 'Linear_Quant', 'HfO2_Physics', 'Noise_Aware']
    target_epoch = 1
    final_epoch = 100

    for file_path in log_files:
        file_name = os.path.basename(file_path)
        print("=" * 70)
        print(f"[분석 대상 파일]: {file_name}")
        print("=" * 70)
        
        data = defaultdict(lambda: defaultdict(list))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    mode = match.group(1).strip()
                    seed = int(match.group(2))
                    epoch = int(match.group(3))
                    acc = float(match.group(4))
                    data[mode][epoch].append(acc)

        stats = {}
        for mode in target_modes:
            if mode in data and target_epoch in data[mode]:
                acc_list = data[mode][target_epoch]
                stats[mode] = {'mean': np.mean(acc_list), 'std': np.std(acc_list)}
            else:
                stats[mode] = {'mean': 0.0, 'std': 0.0}

        fp32_mean = stats['Ideal_FP32']['mean']
        quant_mean = stats['Linear_Quant']['mean']
        physics_mean = stats['HfO2_Physics']['mean']

        print(f"\n▶ [Epoch {target_epoch}] 정량적 오차 분석 (초기 충격량)")
        if fp32_mean > 0:
            res_penalty = fp32_mean - quant_mean
            asym_penalty = quant_mean - physics_mean
            print(f"  - [기준] Ideal_FP32 평균     : {fp32_mean*100:.2f}%")
            print(f"  - [차이] 해상도 한계 오차    : {res_penalty*100:+.2f}%p (FP32 - Quant)")
            print(f"  - [차이] 비대칭성 오차       : {asym_penalty*100:+.2f}%p (Quant - Physics)")
            print(f"  - [불안정성] FP32(σ {stats['Ideal_FP32']['std']*100:.2f}%) vs Physics(σ {stats['HfO2_Physics']['std']*100:.2f}%)")
        else:
            print("  - 데이터 없음")


        print(f"\n▶ [Epoch {final_epoch}] 최종 수렴 및 분산(σ) 평탄화 결과")
        has_final_data = any(final_epoch in data[mode] for mode in target_modes)
        
        if has_final_data:
            for mode in target_modes:
                if mode in data and final_epoch in data[mode]:
                    acc_list = data[mode][final_epoch]
                    mean_val = np.mean(acc_list)
                    std_val = np.std(acc_list)
                    print(f"  - {mode:15s} | 평균: {mean_val*100:.2f}% | 표준편차(Error Bar 두께): {std_val*100:.4f}%")
        else:
            print("  - 아직 Epoch 100 데이터가 기록되지 않았습니다.")
        print("\n")

if __name__ == "__main__":
    analyze_ep100_logs_per_file(".")