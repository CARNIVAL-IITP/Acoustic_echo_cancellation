clear;
close all;
clc;

mex -setup c++
mex rir_generator.cpp

sampling_rate = 16000;      % sampling rate
c = 340;                    % Sound velocity [m/s]
n = 8192;                   % length of impulse response

s = [7.0 2.1 0.7];              % signal ls point coordinate
r = [3.5 2.1 0.7];            % signal mic point coordinate
l = [7.0 4.2 2.7];          % Room dimensions [x y z] [m]
beta = 0.22;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"409_rir.txt");

s = [7.9 3.5 0.7];              % signal ls point coordinate
r = [3.95 3.5 0.7];            % signal mic point coordinate
l = [7.9 7.0 2.7];          % Room dimensions [x y z] [m]
beta = 0.2;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"819_rir.txt");

s = [8.5 3.6 0.7];              % signal ls point coordinate
r = [4.25 3.6 0.7];            % signal mic point coordinate
l = [8.5 7.2 2.9];          % Room dimensions [x y z] [m]
beta = 0.24;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"111_rir.txt");

s = [10.0 4.25 0.7];              % signal ls point coordinate
r = [5.0 4.25 0.7];            % signal mic point coordinate
l = [10.0 8.5 3.5];          % Room dimensions [x y z] [m]
beta = 0.26;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"222_rir.txt");

s = [6.8 3.4 0.7];              % signal ls point coordinate
r = [3.4 3.4 0.7];            % signal mic point coordinate
l = [6.8 6.8 2.3];          % Room dimensions [x y z] [m]
beta = 0.28;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"333_rir.txt");

s = [4.4 1.5 0.7];              % signal ls point coordinate
r = [2.2 1.5 0.7];            % signal mic point coordinate
l = [4.4 3.0 3.0];          % Room dimensions [x y z] [m]
beta = 0.3;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"444_rir.txt");

s = [9.2 3.9 0.7];              % signal ls point coordinate
r = [4.6 3.9 0.7];            % signal mic point coordinate
l = [9.2 7.8 3.2];          % Room dimensions [x y z] [m]
beta = 0.45;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"555_rir.txt");

s = [2.5 0.7 0.7];              % signal ls point coordinate
r = [1.25 0.7 0.7];            % signal mic point coordinate
l = [2.5 1.4 1.0];          % Room dimensions [x y z] [m]
beta = 0.15;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"666_rir.txt");

s = [3.0 2.1 1.0];              % signal person point coordinate
r = [3.5 2.1 0.7];            % signal mic point coordinate
l = [7.0 4.2 2.7];          % Room dimensions [x y z] [m]
beta = 0.22;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"409_p_rir.txt");

s = [3.45 3.5 1.0];              % signal person point coordinate
r = [3.95 3.5 0.7];            % signal mic point coordinate
l = [7.9 7.0 2.7];          % Room dimensions [x y z] [m]
beta = 0.2;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"819_p_rir.txt");

s = [3.75 3.6 1.0];              % signal person point coordinate
r = [4.25 3.6 0.7];            % signal mic point coordinate
l = [8.5 7.2 2.9];          % Room dimensions [x y z] [m]
beta = 0.24;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"111_p_rir.txt");

s = [4.5 4.25 1.0];              % signal person point coordinate
r = [5.0 4.25 0.7];            % signal mic point coordinate
l = [10.0 8.5 3.5];          % Room dimensions [x y z] [m]
beta = 0.26;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"222_p_rir.txt");

s = [2.9 3.4 1.0];              % signal person point coordinate
r = [3.4 3.4 0.7];            % signal mic point coordinate
l = [6.8 6.8 2.3];          % Room dimensions [x y z] [m]
beta = 0.28;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"333_p_rir.txt");

s = [1.7 1.5 1.0];              % signal person point coordinate
r = [2.2 1.5 0.7];            % signal mic point coordinate
l = [4.4 3.0 3.0];          % Room dimensions [x y z] [m]
beta = 0.3;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"444_p_rir.txt");

s = [4.1 3.9 1.0];              % signal person point coordinate
r = [4.6 3.9 0.7];            % signal mic point coordinate
l = [9.2 7.8 3.2];          % Room dimensions [x y z] [m]
beta = 0.45;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"555_p_rir.txt");

s = [0.75 0.7 1.0];              % signal person point coordinate
r = [1.25 0.7 0.7];            % signal mic point coordinate
l = [2.5 1.4 1.0];          % Room dimensions [x y z] [m]
beta = 0.15;                 % Reverberation time [s] RT60
rir = rir_generator(c, sampling_rate, r, s, l, beta, n);
writematrix(rir,"666_p_rir.txt");

% -------------- python convolution -------------- 
%path_1st = np.loadtxt("rir.txt", delimiter=",")
%path_1st = torch.from_numpy(path_1st).float()       # shape: torch[n]
%path_1st = path_1st.unsqueeze(0)        # torch[1,n]
%path_1st = path_1st.unsqueeze(0)        # torch[1,1,n]
%path_1st_fliped = torch.flip(path_1st, [2])     # torch[1,1,n]
%# x shape: torch[1,1,samples], d shape: torch[1,1,n]
%x = torch.cat([x, torch.zeros(1,1,x.shape[2]%order)], dim=2)
%d = torch.nn.functional.pad(x, (path_1st.shape[2]-1, 0))
%d = torch.nn.functional.conv2d(d[None, ...], path_1st_fliped[None, ...])[0]     # primary noise shape: torch[1,1,samples]