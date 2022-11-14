from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import glob, librosa, numpy as np, csv

def IITP_ES2_dataloader(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            IITP_ES_dataset2(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=feature_options.num_workers,
            shuffle=True,
        )

class IITP_ES_dataset2(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda_option, cuda_device=None):
        self.sampling_rate = feature_options.sampling_rate
        self.model_name = model_name
        self.cuda_option = cuda_option
        self.file_list = []
        self.arr = feature_options.arr
        self.chan = feature_options.chan
        if self.chan == 1:
            self.mono = True
        elif self.chan > 1:
            self.mono = False
        else:
            exit()
        full_path = feature_options.data_path+partition+'/mic/{}/*.wav'.format(self.arr)
        self.file_list = glob.glob(full_path)

    def get_feature(self,fn):
        sample_rate = self.sampling_rate
        audio_mix, _ = librosa.load(fn, mono=False, sr=sample_rate)        
        audio_tar, _ = librosa.load(fn.replace('/mic/','/near/'), mono=False, sr=sample_rate)        
        audio_aux, _ = librosa.load(fn.replace('/mic/', '/far_src/'), mono=False, sr=sample_rate)        
        audio_echo, _ = librosa.load(fn.replace('/mic/', '/echo/'), mono=False, sr=sample_rate)
        audio_noise, _ = librosa.load(fn.replace('/mic/','/noise/'), mono=False, sr=sample_rate)        
        if self.mono == True:
            audio_mix = audio_mix[0]
            audio_tar = audio_tar[0]            
            audio_echo = audio_echo[0]
            audio_noise = audio_noise[0]
        audio_mix = audio_mix + audio_noise
        if self.model_name == "HY_IITP_ESNet1":
            input, label = [audio_mix, audio_aux], [audio_tar]
        elif self.model_name == "HY_IITP_ESNet2":
            input, label = [audio_mix, audio_aux], [audio_tar]
        return input, label

    def __getitem__(self, index):
        file_name_mix = self.file_list[index]
        return self.get_feature(file_name_mix)

    def __len__(self):
        return len(self.file_list)
