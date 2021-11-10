from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import glob, librosa, numpy as np

def IITP_ES_test_dataloader(model_name, feature_options, partition, cuda_option, cuda_device=None):
        return DataLoader(
            IITP_ES_test_dataset(model_name, feature_options, partition, cuda_option, cuda_device=cuda_device),
            batch_size=feature_options.batch_size,
            num_workers=feature_options.num_workers,
            shuffle=False,
        )

class IITP_ES_test_dataset(Dataset):
    def __init__(self, model_name, feature_options, partition, cuda_option, cuda_device=None):
        self.sampling_rate = feature_options.sampling_rate
        self.model_name = model_name
        self.cuda_option = cuda_option
        self.file_list = []
        full_path = feature_options.data_path+partition+'/*.wav'
        self.file_list = glob.glob(full_path)        

    def get_feature(self,fn,max_dur=160000):
        sample_rate = self.sampling_rate
        audio_mix, _ = librosa.load(fn, mono=False, sr=sample_rate)
        audio_aux, _ = librosa.load(fn.replace('/nearend_mic_signal', '/farend_speech').replace('nearend_mic_', 'farend_speech_'), mono=False, sr=sample_rate)
        org_len = audio_mix.shape[0]
        if audio_mix.shape[0] < max_dur:
            pad_len = max_dur - audio_mix.shape[0]
            audio_mix = np.concatenate((audio_mix, audio_mix[:pad_len]), axis=0)
            audio_aux = np.concatenate((audio_aux, audio_aux[:pad_len]), axis=0)
        if self.model_name == "HY_IITP_ESNet1":
            input, infdat = [audio_mix, audio_aux], [fn.split('/')[-1], org_len]
        return input, infdat

    def __getitem__(self, index):
        file_name_mix = self.file_list[index]
        return self.get_feature(file_name_mix)

    def __len__(self):
        return len(self.file_list)
