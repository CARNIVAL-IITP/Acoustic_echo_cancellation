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

pesq_tot = 0
nums = 0
enh_list = os.listdir(enh_dir)

for (cnt, na) in enumerate(enh_list):
    
    enh_path = os.path.join(enh_dir, na)
    if not os.path.isfile(enh_path):
        continue
    sd = na.split('_')[0]               # 111000001111
    speech_na = na.split('.')[0]        # 111000001111_y_111
    speech_na = speech_na.split('_')[2] # 111
    speech_path = os.path.join(speech_dir, "%s_v_%s.wav"  % (sd,speech_na))
    print(cnt, enh_path, "====", speech_path)
    cmd = ' '.join(["./pesq_folder/pesq", speech_path, enh_path, '+'+str(sr)])
    result = os.popen(cmd).read()
    result = result.splitlines()
    #print(result[-1])                   # Prediction : PESQ_MOS = 2.212
    if "PESQ_MOS" not in result[-1]:
        continue
    pesq = float(result[-1].split('= ')[1])
    print(pesq)
    pesq_tot = pesq_tot + pesq
    nums = nums + 1

print("avg : ",pesq_tot/nums)
