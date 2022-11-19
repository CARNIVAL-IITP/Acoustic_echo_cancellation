import librosa
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

from trainer.base_trainer import BaseTrainer
import matplotlib.pyplot as plt

plt.switch_backend("agg")
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
import librosa.display
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer(BaseTrainer):
    def __init__(self,
                 config,
                 resume,
                 model,
                 optimizer,
                 loss_function,
                 train_dataloader,
                 validation_dataloader):
        super(Trainer, self).__init__(config, resume, model, optimizer, loss_function)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.root_dir = (Path(config["save_location"]) / config["experiment_name"]).expanduser().absolute()
        self.checkpoints_dir = self.root_dir / "checkpoints"
        print(train_dataloader)
        print(self.train_dataloader)

    def _train_epoch(self, epoch):
        print('\n')
        loss_total = 0.0
        
        len_d = len(self.train_dataloader)
        pbar = tqdm(total = len_d)
        
        for mixture, clean, n_frames_list, _ in tqdm(self.train_dataloader, desc="Training"):
            self.optimizer.zero_grad()

            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D = self.stft.transform(mixture)   # filter_size=320 / hop_size=160 [batch, T, F, real/im]

            clean_D = self.stft.transform(clean)
            
            mixture_reshape = mixture_D.permute(0, 3, 1, 2)     # [B, 2, T, F]
            cur_bat = mixture_reshape.shape[0]
            hn1_f = torch.zeros(1,cur_bat,512).to(self.device)
            cn1_f = torch.zeros(1,cur_bat,512).to(self.device)
            hn1_b = torch.zeros(1,cur_bat,512).to(self.device)
            cn1_b = torch.zeros(1,cur_bat,512).to(self.device)
            hn2_f = torch.zeros(1,cur_bat,512).to(self.device)
            cn2_f = torch.zeros(1,cur_bat,512).to(self.device)
            hn2_b = torch.zeros(1,cur_bat,512).to(self.device)
            cn2_b = torch.zeros(1,cur_bat,512).to(self.device)
            enhanced_concat, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b = self.model(mixture_reshape, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b)       # [B, 2, T, F]
            
            clean_concat = clean_D.permute(0, 3, 1, 2)          # [B, 2, T, F]
            loss = self.loss_function(enhanced_concat, clean_concat, n_frames_list)
            loss.backward()
            
            self.optimizer.step()
            loss_total += float(loss)   # training loss
            
            text = 'Epoch: {:03d}, Training loss: {:06f}'.format(epoch, loss)
            pbar.set_description(text)
            pbar.update(1)

        save_path = "model_epoch%d" % epoch
        torch.save(self.model, self.checkpoints_dir/save_path)
        dataloader_len = len(self.train_dataloader)
        print("train loss = ", loss_total / dataloader_len)

        return loss_total/dataloader_len

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        loss_total = 0.0
        
        len_d = len(self.train_dataloader)
        pbar = tqdm(total = len_d)

        for mixture, clean, n_frames_list, _ in tqdm(self.validation_dataloader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            # Mixture mag and Clean mag
            mixture_D = self.stft.transform(mixture)
            clean_D = self.stft.transform(clean)

            mixture_reshape = mixture_D.permute(0, 3, 1, 2)     # [B, 2, T, F]

            cur_bat = mixture_reshape.shape[0]
            hn1_f = torch.zeros(1,cur_bat,512).to(self.device)
            cn1_f = torch.zeros(1,cur_bat,512).to(self.device)
            hn1_b = torch.zeros(1,cur_bat,512).to(self.device)
            cn1_b = torch.zeros(1,cur_bat,512).to(self.device)
            hn2_f = torch.zeros(1,cur_bat,512).to(self.device)
            cn2_f = torch.zeros(1,cur_bat,512).to(self.device)
            hn2_b = torch.zeros(1,cur_bat,512).to(self.device)
            cn2_b = torch.zeros(1,cur_bat,512).to(self.device)
            
            enhanced_concat, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b = self.model(mixture_reshape, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b)       # [B, 2, T, F]
            clean_concat = clean_D.permute(0, 3, 1, 2)          # [B, 2, T, F]

            loss = self.loss_function(enhanced_concat, clean_concat, n_frames_list)
            loss_total += float(loss)  # validation loss
            
            text = 'Epoch: {:03d}, Training loss: {:06f}'.format(epoch, loss)
            pbar.set_description(text)
            pbar.update(1)

        dataloader_len = len(self.validation_dataloader)
        print("validation loss = ", loss_total / dataloader_len)

        return loss_total / dataloader_len