# check_audio.py
import torch
import torchaudio
from model_training.dataset import VSTDataset
import glob
import random
import os

# 配置
H5_PATH = "/scratch/hw3140/vimtopoeia/datasets/dataset_4k_pair.h5"
IR_DIR = "/scratch/hw3140/vimtopoeia/datasets/vimsketch_synth_vocals"

# 1. 准备 IR 列表
ir_files = glob.glob(os.path.join(IR_DIR, "*.wav"))
print(f"Found {len(ir_files)} IR files.")

# 2. 实例化 Dataset (强制 100% augment 以便测试)
ds = VSTDataset(
    H5_PATH, 
    ir_folder_list=ir_files, 
    subset='train', 
    augment_prob=1.0 # 强制开启
)

# 3. 抽取几个样本并保存
os.makedirs("debug_audio", exist_ok=True)

print("Generating 5 test samples...")
for i in range(5):
    # 随机取样
    idx = random.randint(0, len(ds)-1)
    
    # 获取数据
    target_aug, ref_clean, label = ds[idx]
    
    # 保存
    # target_aug 是被卷积过的
    # ref_clean 是纯净的
    torchaudio.save(f"debug_audio/test_{i}_target_convolved.wav", target_aug, 44100)
    torchaudio.save(f"debug_audio/test_{i}_ref_clean.wav", ref_clean, 44100)
    
    print(f"Saved pair {i} to debug_audio/")

print("Done! Please download 'debug_audio' folder and LISTEN to them.")