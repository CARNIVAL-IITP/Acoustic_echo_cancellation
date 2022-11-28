"""
Summary:  Calculate PESQ and overal stats of enhanced speech. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import argparse
import os
import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import soundfile as sf
from pystoi import stoi

speech_dir = "/root/share/jp_data/AFC/tr/v"
enh_dir = "./output/train_220903_0812/enhancements/best_checkpoint_28_epoch"
sr = 16000

stoi_tot = 0
estoi_tot = 0
nums = 0
enh_list = os.listdir(enh_dir)

for (cnt, na) in enumerate(enh_list):
    enh_path = os.path.join(enh_dir, na)
    if not os.path.isfile(enh_path):
        continue
    nums = nums + 1
    sp_name = na[:14]+"v"+na[15:]
    sp_path = os.path.join(speech_dir, sp_name)
    clean, fs = sf.read(sp_path)
    enh, fs = sf.read(enh_path)
    clean = clean[:len(enh)]
    d = stoi(clean, enh, fs, extended=False)
    stoi_tot = stoi_tot + d
    e = stoi(clean, enh, fs, extended=True)
    estoi_tot = estoi_tot + e
    print("stoi: ",d)
    print("estoi: ",e)
print("stoi avg : ",stoi_tot/nums)
print("estoi avg : ",estoi_tot/nums)


