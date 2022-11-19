import torch.nn as nn
import torch.nn.functional as F
import torch

class CRNN(nn.Module):
    # [Batch, Real/Complex, Time, Frequency]
    
    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2))   # 2-ch input : Real, Imag
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=256)

        # grouped LSTM ( K=2 )
        self.LSTM1_1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.LSTM1_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.LSTM2_1 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)
        self.LSTM2_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, batch_first=True)

        # Decoder for real
        self.convT1_real = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1_real = nn.BatchNorm2d(num_features=128)
        self.convT2_real = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2_real = nn.BatchNorm2d(num_features=64)
        self.convT3_real = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3_real = nn.BatchNorm2d(num_features=32)
        self.convT4_real = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1))
        self.bnT4_real = nn.BatchNorm2d(num_features=16)
        self.convT5_real = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5_real = nn.BatchNorm2d(num_features=1)
        
        # Decoder for imag
        self.convT1_imag = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.bnT1_imag = nn.BatchNorm2d(num_features=128)
        self.convT2_imag = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.bnT2_imag = nn.BatchNorm2d(num_features=64)
        self.convT3_imag = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.bnT3_imag = nn.BatchNorm2d(num_features=32)
        self.convT4_imag = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1))
        self.bnT4_imag = nn.BatchNorm2d(num_features=16)
        self.convT5_imag = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))
        self.bnT5_imag = nn.BatchNorm2d(num_features=1)
   


    def forward(self, x, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b):
        # (B, R/C, T, F)
        x1 = F.elu(self.bn1(self.conv1(x)))
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2)))
        x4 = F.elu(self.bn4(self.conv4(x3)))
        x5 = F.elu(self.bn5(self.conv5(x4)))
        # reshape
        out5 = x5.permute(0, 2, 1, 3) # [B, T, Ch, F]
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        
        # grouped lstm (K=2)
        lstm1_front, (hn1_f, cn1_f) = self.LSTM1_1(out5[:, :, :512],(hn1_f, cn1_f))
        lstm1_back, (hn1_b, cn1_b) = self.LSTM1_2(out5[:, :, 512:],(hn1_b, cn1_b))
        lstm1_front2 = lstm1_front.unsqueeze(3)
        lstm1_back2 = lstm1_back.unsqueeze(3)
        lstm1_reshape = torch.cat((lstm1_front2, lstm1_back2), 3)
        lstm1_trans = torch.transpose(lstm1_reshape, 2, 3)
        lstm1_rearrange = lstm1_trans.reshape(lstm1_trans.size()[0], lstm1_trans.size()[1], -1)
        lstm2_front, (hn2_f, cn2_f) = self.LSTM2_1(lstm1_rearrange[:, :, 0:512],(hn2_f, cn2_f))
        lstm2_back, (hn2_b, cn2_b) = self.LSTM2_2(lstm1_rearrange[:, :, 512:],(hn2_b, cn2_b))
        lstm2 = torch.cat((lstm2_front, lstm2_back), 2)

        # reshape
        output = lstm2.reshape(lstm2.size()[0], lstm2.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        
        # ConvTrans for real
        res_real = torch.cat((output, x5), 1)                               # skip connection
        res1_real = F.elu(self.bnT1_real(self.convT1_real(res_real)))
        res1_real = torch.cat((res1_real, x4), 1)                           # skip connection
        res2_real = F.elu(self.bnT2_real(self.convT2_real(res1_real)))
        res2_real = torch.cat((res2_real, x3), 1)                           # skip connection
        res3_real = F.elu(self.bnT3_real(self.convT3_real(res2_real)))
        res3_real = torch.cat((res3_real, x2), 1)                           # skip connection
        res4_real = F.elu(self.bnT4_real(self.convT4_real(res3_real)))
        res4_real = torch.cat((res4_real, x1), 1)                           # skip connection
        res5_real = self.bnT5_real(self.convT5_real(res4_real))   # [B, 1, T, 161]

        # ConvTrans for imag
        res_imag = torch.cat((output, x5), 1)                               # skip connection
        res1_imag = F.elu(self.bnT1_imag(self.convT1_imag(res_imag)))
        res1_imag = torch.cat((res1_imag, x4), 1)                           # skip connection
        res2_imag = F.elu(self.bnT2_imag(self.convT2_imag(res1_imag)))
        res2_imag = torch.cat((res2_imag, x3), 1)                           # skip connection
        res3_imag = F.elu(self.bnT3_imag(self.convT3_imag(res2_imag)))
        res3_imag = torch.cat((res3_imag, x2), 1)                           # skip connection
        res4_imag = F.elu(self.bnT4_imag(self.convT4_imag(res3_imag)))
        res4_imag = torch.cat((res4_imag, x1), 1)                           # skip connection
        res5_imag = self.bnT5_imag(self.convT5_imag(res4_imag))   # [B, 1, T, 161]

        # concat real & imag
        res5 = torch.cat((res5_real, res5_imag), 1)     # [B, 2, T, 161]
        return res5, hn1_f, cn1_f, hn1_b, cn1_b, hn2_f, cn2_f, hn2_b, cn2_b