from posixpath import split
from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
# from models.HY_IITP_ESNet_framebyframe import HY_IITP_ESNet1
from models.HY_IITP_ESNet_framebyframe import HY_IITP_ESNet2_framebyframe
from models.HY_IITP_ESNet_framebyframe import HY_IITP_ESNet3_framebyframe
from models.HY_IITP_ESNet import HY_IITP_ESNet3
import argparse, json, numpy as np, os, time, torch, glob, natsort
import soundfile as sf
import scipy.signal as ss
import random
import glob
import librosa
from time import time
from tqdm import tqdm

set_num_threads = 1
torch.set_num_threads(set_num_threads)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

mode1 = 'tt'
min_len = 128000

def vad(buf):
    src_idx = 0
    dst_idx = buf.shape[0]
    for i in range(buf.shape[0]):
        if abs(buf[i]) > 1e-8:
            src_idx = i
            break

    for i in range(buf.shape[0]-1, -1, -1):
        if abs(buf[i]) > 1e-8:
            dst_idx = i
            break
    
    return src_idx, dst_idx

def power_amplifier_clipping(buf, max_val):
    # clipping
    buf = torch.clamp(buf, min=-max_val, max=max_val)
    return buf

def mic_clipping(buf, max_val):
    # clipping
    buf = torch.clamp(buf, min=-max_val, max=max_val)
    return buf

def loudspeaker_asymmetric_nonlinear(buf):
    # b(n)
    buf_b = 1.5 * buf - 0.3 * (buf**2)
    buf_nl = torch.zeros_like(buf)
    palpha = torch.where(buf_b > 0, 4.0, 0.5)
    buf_nl = 4 * (2 / (1 + torch.exp( -palpha * buf_b )) -1 )
    return buf_nl

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

class tester:
    def __init__(self, args, loss_type):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        self.dataset = args.dataset
        self.loss_type = loss_type
        self.hop_size = args.model_options.win_inc
        self.win_size = self.hop_size * 2
        self.max_val = 0.8
        self.chan = args.model_options.chan
        if args.cuda_option == "True":
            print("GPU mode on...")
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print("self.device:", self.device)

        # build model
        self.near_model = self.init_model(args, args.model_name, args.model_options)
        print("Loaded the model...")
        self.feature_options = args.feature_options
        
        load_path_near_src = "./DB/"+mode1+"/mic/*_4ch_circular/*.wav"
        self.load_path_near_src_nm_list = glob.glob(load_path_near_src)
        self.load_path_near_src_nm_list.sort()

    def init_model(self, args, model_name, model_options):
        assert model_name is not None, "Model name must be defined!"
        assert "HY_IITP_ESNet" in model_name, \
            "Model name is not supported! Must be one of (HY_IITP_ESNet1,HY_IITP_ESNet2)"
        if model_name == "HY_IITP_ESNet1":
            model = HY_IITP_ESNet2_framebyframe(model_options)
            folder_name = './output/%s_%s_%s'%(model_name, args.dataset, args.loss_option)
            model_list = natsort.natsorted(glob.glob(folder_name+'/*'))            
            fin_model = model_list[-1]            
            model.load_state_dict(torch.load(fin_model, map_location='cpu'))
            print(folder_name)
            print(model_name)
            print(fin_model)
        elif model_name == "HY_IITP_ESNet2" or model_name == "AEC_HY_IITP_ESNet2":
            model = HY_IITP_ESNet2_framebyframe(model_options)
            # folder_name = './output/230622_311/%s_%s_%s_%s'%(model_name, args.dataset, args.loss_option, self.loss_type)      
            folder_name = './output/%s/%s_%s_%s'%(model_name, args.dataset, args.loss_option, self.loss_type)          
            model_list = natsort.natsorted(glob.glob(folder_name+'/*'))
            print(model_list)
            fin_model = model_list[-1]         
            # fin_model = model_list[23]         
            print("=============================")
            print(folder_name)
            print(model_name)
            print(fin_model)   
            print("=============================")
            model.load_state_dict(torch.load(fin_model, map_location='cpu'),strict=False)
        elif "HY_IITP_ESNet3" in model_name:
            model = HY_IITP_ESNet3(model_options)
            folder_name = './output/%s_%s_%s_%s_%schan_%s'%(self.model_name, self.dataset, self.loss_name, str(self.loss_type), str(args.feature_options.chan), args.feature_options.arr)
            print(folder_name)
            model_list = natsort.natsorted(glob.glob(folder_name+'/*'))
            fin_model = model_list[-1]            
            model.load_state_dict(torch.load(fin_model, map_location='cpu'))
            print(fin_model)
            
        model.to(self.device)
        return model
    
    def load_spks(self, mode1, mode2):
        near_end_spks = np.loadtxt("../data_gen/spks/IITP_near_end_spks_"+mode1+mode2, delimiter=",", dtype=str)
        far_end_spks = np.loadtxt("../data_gen/spks/IITP_far_end_spks_"+mode1+mode2, delimiter=",", dtype=str)
        return near_end_spks, far_end_spks

    def run(self):
        self.test()
        print("Model test is finished.")

    def test(self):
        
        self.near_model.eval()
        
        times = AverageMeter()
        times.reset()
        time_tot = 0
        time_max = 0
        sample_num = 0
        with torch.no_grad():
            for idx, load_path_mic_nm in enumerate(tqdm(self.load_path_near_src_nm_list, desc=f'cal_RTF.py', dynamic_ncols=True,)):
                
                # nm -> info (room info, arr info, SER)
                splited_nm = load_path_mic_nm.split('/')
                out_save_path = load_path_mic_nm.replace('/mic/','/out_fbf/')
                
                # load wavs
                mic_wav, sr = sf.read(load_path_mic_nm, dtype='float32')
                mic_wav = torch.from_numpy(mic_wav).permute(1,0)           # [4, 128000]
                far_wav, sr = sf.read(load_path_mic_nm.replace('/mic/','/far_src/'), dtype='float32')
                far_wav = torch.from_numpy(far_wav)             # [128000]
                near_noise, sr = sf.read(load_path_mic_nm.replace('/mic/','/noise/'), dtype='float32')
                near_noise = torch.from_numpy(near_noise).permute(1,0)    # [4,128000]
                near_noisy_wav = near_noise + mic_wav
                
                # buffer set
                near_hn = torch.zeros(2,1,256)
                near_cn = torch.zeros(2,1,256)
                d_n = torch.zeros(self.feature_options.chan, self.hop_size)
                near_ref = torch.zeros(self.win_size)
                near_mic = torch.zeros(self.feature_options.chan, self.win_size)
                near_rir_buf = np.zeros(self.hop_size + 511)
                block_num = min_len // self.hop_size
                near_spec = torch.zeros(self.feature_options.chan, 258, 1)
                             
                out = torch.zeros(0).to(self.device)                
                
                # 1st block process (Ref. signal is generated after 2nd block)
                near_mic[:, self.hop_size:] = near_noisy_wav[:,self.hop_size * 0 : self.hop_size * 1]
                near_ref[self.hop_size:] = far_wav[:self.hop_size]
                # mic[:, :self.hop_size] = near_mic[:, self.hop_size:]

                ###### Add input tensor change to device
                mic_wav = mic_wav.to(self.device)
                far_wav = far_wav.to(self.device)

                time_start = time()
                near_wav_hat = self.near_model([mic_wav[None, ...], far_wav[None, None, ...]]) # None: unsqueeze랑 같은 역할
                time_end = time()
                time_block = ((time_end - time_start)*1000)
                time_tot = time_tot + time_block
                if time_max < time_block:
                    time_max = time_block
                
                  
        print("\n")
        samples = len(self.load_path_near_src_nm_list)
        print("samples:", samples)
        print("RTF:", time_tot / samples / 1000 / 8) # 8 ms (buffering delay = hop_size)


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default='./configs/test_RTF_4ch.json',
                        help='The path to the config file. e.g. python train.py -c configs/test.json')
    parser.add_argument("-l", "--loss_type", type=int, default='0')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = tester(args, config.loss_type)
    t.run()


if __name__ == "__main__":
    main()

