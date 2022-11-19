import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.stft import STFT
from utils.utils import initialize_config
import os

#import graph

def main(config, epoch):
    # root_dir = Path(config["experiments_dir"]) / config["name"]
    root_dir = Path(config["experiments_dir"])
    enhancement_dir = root_dir / "enhancements_concat"
    checkpoints_dir = root_dir / "checkpoints"

    """============== 加载数据集 =============="""
    dataset = initialize_config(config["dataset"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=16,
    )

    """============== 加载模型断点（"best"，"latest"，通过数字指定） =============="""
    model = initialize_config(config["model"])
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    stft = STFT(
        filter_length=320,
        hop_length=160
    ).to("cpu")

    if epoch == "best":
        model_path = checkpoints_dir / "best_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    elif epoch == "latest":
        model_path = checkpoints_dir / "latest_model.tar"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint["model"]
        checkpoint_epoch = model_checkpoint['epoch']
    else:
        model_path = checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth"
        model_checkpoint = torch.load(model_path.as_posix(), map_location=device)
        model_static_dict = model_checkpoint
        checkpoint_epoch = epoch

    print(f"Loading model checkpoint, epoch = {checkpoint_epoch}")
    model.load_state_dict(model_static_dict)
    model.to(device)
    model.eval()

    """============== 增强语音 =============="""
    if epoch == "best" or epoch == "latest":
        results_dir = enhancement_dir / f"{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    else:
        results_dir = enhancement_dir / f"checkpoint_{epoch}_epoch"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir_amp = root_dir / "enhancements_amplitude"
    results_dir_amp = results_dir_amp / f"{epoch}_checkpoint_{checkpoint_epoch}_epoch"
    results_dir_amp.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for i, (mixture, _, _, names) in enumerate(dataloader):
            print(f"Enhance {i + 1}th speech")
            name = names[0]

            # Mixture mag and Clean mag
            print("\tSTFT...")
            #print("1",mixture.shape)                                               # [1, 66384]
            mixture_D = stft.transform(mixture)
            mixture_real = mixture_D[:, :, :, 0]
            mixture_imag = mixture_D[:, :, :, 1]
            mixture_mag = torch.sqrt(mixture_real ** 2 + mixture_imag ** 2) # [1, T, F]


            print("\tEnhancement...")
            # enhanced_mag = model(mixture_mag_chunk).detach().cpu().unsqueeze(0)  # [1, T, F]
            mixture_reshape = mixture_D.permute(0, 3, 1, 2)         # [1, 2, T, F]
            test1 = mixture_reshape[:, 0, :, :]
            test2 = mixture_reshape[:, 1, :, :]
            
            #print("2",mixture_reshape.shape)                                       # [1, 2, 413, 161]
            # enhanced_mag = model(mixture_reshape).detach().cpu().unsqueeze(0)  # [1, T, F]
            """
            for t in range(2):
                if t == 0:
                    enhanced_concat, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b = self.model(mixture_reshape[:,:,:mixture_reshape.shape[2]//2,:], hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b)       # [B, 2, T, F]
                else:
                    enhanced_concat1, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b = self.model(mixture_reshape[:,:,mixture_reshape.shape[2]//2:,:], hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b)       # [B, 2, T, F]
                    #print(enhanced_concat.shape)                    # torch[2,161] -> [1, 2, 1, 161] 바꿔야댐
                    #print(enhanced_concat1.shape)                   # torch[2,161] -> [1, 2, 1, 161] 바꿔야댐
                    enhanced_concat = torch.cat((enhanced_concat, enhanced_concat1), 2)
            """
            cur_bat = mixture_reshape.shape[0]
            hn1_f = torch.zeros(1,cur_bat,512).to(device)
            cn1_f = torch.zeros(1,cur_bat,512).to(device)
            hn1_b = torch.zeros(1,cur_bat,512).to(device)
            cn1_b = torch.zeros(1,cur_bat,512).to(device)
            hn2_f = torch.zeros(1,cur_bat,512).to(device)
            cn2_f = torch.zeros(1,cur_bat,512).to(device)
            hn2_b = torch.zeros(1,cur_bat,512).to(device)
            cn2_b = torch.zeros(1,cur_bat,512).to(device)
            
            enhanced_concat, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b = model(mixture_reshape[:,:,0:1,:], hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b)       # [B, 2, T, F]
            for t in range(1, mixture_reshape.shape[2]):
                enhanced_concat1, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b = model(mixture_reshape[:,:,t:t+1,:], hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b)       # [B, 2, T, F]
                enhanced_concat = torch.cat((enhanced_concat, enhanced_concat1), 2)
            
            #enhanced_concat = model(mixture_reshape).detach().cuda() #cuda   
            # enhanced_concat = model(mixture_reshape).detach().cpu().unsqueeze(0)  # [B, 2, T, F]
            #print("3",enhanced_concat.shape)                                               #[2. 413, 161]
            # enhanced_mag = enhanced_mag.detach().cpu().data.numpy()      # [1, T, F]
            # mixture_mag = mixture_mag.cpu()

            # enhanced_real = enhanced_mag * mixture_real[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
            # enhanced_imag = enhanced_mag * mixture_imag[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
            #print(enhanced_concat.shape)                                    # [1, 2, 524, 161]
            enhanced_concat = enhanced_concat.squeeze()
            enhanced_real = enhanced_concat[0, :, :].unsqueeze(0)
            enhanced_imag = enhanced_concat[1, :, :].unsqueeze(0)

            # enhanced_mag = torch.sqrt(enhanced_real_tmp**2 + enhanced_imag_tmp**2)
            # enhanced_real = enhanced_mag * enhanced_real_tmp[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
            # enhanced_imag = enhanced_mag * enhanced_imag_tmp[:, :enhanced_mag.size(1), :] / mixture_mag[:, :enhanced_mag.size(1), :]
            #
            enhanced_D = torch.stack([enhanced_real.unsqueeze(3), enhanced_imag.unsqueeze(3)], 3).squeeze(0).permute(3, 0, 1, 2)

            enhanced = stft.inverse(enhanced_D)
            #print("4",enhanced.shape)                       #[1, 66240]

            enhanced = enhanced.detach().cpu().squeeze().numpy() #cpu
            #enhanced = enhanced.detach().cuda().squeeze().numpy() #cuda

            sf.write(f"{results_dir}/{name}.wav", enhanced, 16000)
            sf.write(f"{results_dir_amp}/{name}.wav", enhanced*15, 16000)
            # sf.write(f"{results_dir}/{name}.wav", enhanced, 8000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Spectrogram mapping: Speech Enhancement")
    parser.add_argument("-C", "--config", default="config/enhancement/enhancement.json", type=str,
                        help="Specify the configuration file for enhancement (*.json).")
    parser.add_argument("-E", "--epoch", default="best",
                        help="Model checkpoint for speech enhancement, can be set to 'best', 'latest' and specific epoch. (default: 'best')")
    args = parser.parse_args()

    config = json.load(open(args.config))
    config["name"] = os.path.splitext(os.path.basename(args.config))[0]
    main(config, args.epoch)
    #graph.get_PESQ()
    #graph.get_stats()




    