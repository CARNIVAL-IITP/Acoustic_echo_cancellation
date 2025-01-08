from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
from data.IITP_ES import IITP_ES3_dataloader
from models.HY_IITP_ESNet import HY_IITP_ESNet1
from models.HY_IITP_ESNet import HY_IITP_ESNet2
from models.HY_IITP_ESNet import HY_IITP_ESNet3
import argparse, json, numpy as np, os, time, torch, datetime, glob, natsort
from torch.utils.tensorboard import SummaryWriter
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

class trainer:
    def __init__(self, args, loss_type, resume):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        self.dataset = args.dataset
        self.loss_type = loss_type
        self.resume = resume
        if args.cuda_option == "True":
            print("GPU mode on...")
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')        
        #resume
        self.output_path = args.output_path+'/%s_%s_%s_%s_%schan_%s'%(self.model_name, self.dataset, self.loss_name, str(self.loss_type), str(args.feature_options.chan), args.feature_options.arr)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # build model
        self.model = self.init_model(args.model_name, args.model_options)
        print("Loaded the model...")
        # build loss fn
        self.loss_fn = self.build_lossfn(args.loss_option)
        print("Built the loss function...")
        # build optimizer
        self.optimizer = self.build_optimizer(self.model.parameters(), args.optimizer_options)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = args.gamma_base**(1/args.gamma_power_den))
        print("Built the optimizer...")
        # build DataLoaders
        if args.dataset == "IITP_ES1":
            self.train_loader = IITP_ES3_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
            self.valid_loader = IITP_ES3_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)
        elif args.dataset == "IITP_ES2":
            self.train_loader = IITP_ES3_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
            self.valid_loader = IITP_ES3_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)
        elif "IITP_ES3" in args.dataset:
            self.train_loader = IITP_ES3_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
            self.valid_loader = IITP_ES3_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)

        # training options
        self.num_epoch = args.num_epoch
        self.output_path = args.output_path+'/%s_%s_%s_%s_%schan_%s'%(self.model_name, self.dataset, self.loss_name, str(self.loss_type), str(args.feature_options.chan), args.feature_options.arr)
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.min_loss = float("inf")
        self.early_stop_count = 0
        self.epoch = 0
        if self.resume == 'True':
            model_list = natsort.natsorted(glob.glob(self.output_path+'/*'))
            saved_params = torch.load(model_list[-1], map_location='cpu')
            self.model.load_state_dict(saved_params['state_dict'])
            self.model.to(self.device)
            self.optimizer.load_state_dict(saved_params['optimizer'])
            self.epoch = saved_params['epoch']
            print('epoch {} is resumed!\n'.format(self.epoch-1))
        self.max_early_stop_count = args.max_early_stop_count

    def init_model(self, model_name, model_options):
        assert model_name is not None, "Model name must be defined!"
        assert "HY_IITP_ESNet" in model_name, \
            "Model name is not supported! Must be one of (HY_IITP_ESNet1, HY_IITP_ESNet2)"
        if model_name == "HY_IITP_ESNet1":
            model = HY_IITP_ESNet1(model_options)
        elif model_name == "HY_IITP_ESNet2":
            model = HY_IITP_ESNet2(model_options)
        elif "HY_IITP_ESNet3" in model_name:
            model = HY_IITP_ESNet3(model_options)
        model.to(self.device)
        return model

    def build_lossfn(self, fn_name):
        return get_lossfns()[fn_name]

    def build_optimizer(self, params, optimizer_options):
        if optimizer_options.name == "adam":
            return torch.optim.Adam(params, lr=optimizer_options.lr)
        if optimizer_options.name == "sgd":
            return torch.optim.SGD(params, lr=optimizer_options.lr, momentum=0.9)
        if optimizer_options.name == "rmsprop":
            return torch.optim.RMSprop(params, lr=optimizer_options.lr)

    def run(self):
        for epoch in range(self.num_epoch):
            self.train(epoch)
            self.validate(epoch)
            if self.early_stop_count == self.max_early_stop_count:
                print("Model stops improving, stop the training")
                break
        print("Model training is finished.")

    def train(self, epoch):
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        self.model.train()
        len_d = len(self.train_loader)
        end = time.time()
        for i, data in enumerate(self.train_loader):
            input, fn, label = data
            input = [ele.to(self.device) for ele in input]
            label = [ele.to(self.device) for ele in label]
            output = self.model(input)
            
            # loss_snr, corrxd, corrys, corryd = self.loss_fn(output, label, input)
            loss_snr = self.loss_fn(output, label, input)
            if self.loss_type == 0:
                bat_loss = loss_snr
            
            bat_loss_avg = torch.mean(bat_loss)
            if torch.isnan(bat_loss_avg):
                    print("\ntraining loss NaN!")
                    torch.save(self.model.state_dict(),self.output_path+"/model.epoch%d_NaN"%(epoch-1))
                    exit()
            losses.update(bat_loss_avg.item())
            self.optimizer.zero_grad()
            bat_loss_avg.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            times.update(time.time() - end)
            end = time.time()
            writer.add_scalar('train_loss/batch_loss', bat_loss_avg, epoch * len_d + i)
            print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, bat_loss_avg, times.avg * len_d), end='\r')
        self.scheduler.step()
        print("\n")
        writer.add_scalar('train_loss/train_loss', losses.avg, epoch)


    def validate(self, epoch):
        self.model.eval()
        losses = AverageMeter()
        times = AverageMeter()
        losses.reset()
        times.reset()
        len_d = len(self.valid_loader)
        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                input, fn, label = data
                input = [ele.to(self.device) for ele in input]
                label = [ele.to(self.device) for ele in label]
                output = self.model(input)
                
                loss_snr = self.loss_fn(output, label, input)
                if self.loss_type == 0:
                    bat_loss = loss_snr
                
                bat_val_loss_avg = torch.mean(bat_loss)
                if torch.isnan(bat_val_loss_avg):
                    print("\nvalidation loss NaN!")
                    torch.save(self.model.state_dict(),self.output_path+"/model.epoch%d_NaN"%(epoch-1))
                    exit()
                losses.update(bat_val_loss_avg.item())
                times.update(time.time() - end)
                end = time.time()
                writer.add_scalar('valid_loss/batch_loss', bat_val_loss_avg, epoch * len_d + i)
                print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds' % (epoch, i + 1, len_d, bat_val_loss_avg, times.avg * len_d), end='\r')
            print("\n")
        writer.add_scalar('valid_loss/valid_loss', losses.avg, epoch)
        if losses.avg < self.min_loss:
            self.early_stop_count = 0
            self.min_loss = losses.avg
            torch.save(self.model.state_dict(),self.output_path+"/model.epoch%d"%epoch)
            print("Saved new model, loss:", str(losses.avg))
        else:
            self.early_stop_count += 1


def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default='./configs/train.json',
                        help='The path to the config file. e.g. python train.py -c configs/train.json')
    parser.add_argument("-l", "--loss_type", type=int, default='0')
    parser.add_argument("-r", "--resume", default='False')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
        
    t = trainer(args, config.loss_type, config.resume)
    t.run()
    

if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "./tensorboard/IITP3_baseline/"+current_time
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)
    main()
    writer.close()
