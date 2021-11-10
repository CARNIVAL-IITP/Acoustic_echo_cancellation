from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, models, numpy as np, os, time, torch, datetime
from torch.utils.tensorboard import SummaryWriter

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

class trainer:
    def __init__(self, args):
        self.model_name = args.model_name
        self.loss_name = args.loss_option
        self.dataset = args.dataset
        if args.cuda_option == "True":
            print("GPU mode on...")
            available_device = get_free_gpu()
            print("We found an available GPU: %d!"%available_device)
            self.device = torch.device('cuda:%d'%available_device)
        else:
            self.device = torch.device('cpu')
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
            self.train_loader = data.IITP_ES_dataloader(args.model_name, args.feature_options, 'tr', args.cuda_option, self.device)
            self.valid_loader = data.IITP_ES_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)

        # training options
        self.num_epoch = args.num_epoch
        self.output_path = args.output_path+'/%s_%s_%s'%(self.model_name, self.dataset, self.loss_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.min_loss = float("inf")
        self.early_stop_count = 0
        self.max_early_stop_count = args.max_early_stop_count

    def init_model(self, model_name, model_options):
        assert model_name is not None, "Model name must be defined!"
        assert model_name in ["HY_IITP_ESNet1"], \
            "Model name is not supported! Must be one of (HY_IITP_ESNet1)"
        if model_name == "HY_IITP_ESNet1":
            model = models.HY_IITP_ESNet1(model_options)
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
            input, label = data
            input = [ele.to(self.device) for ele in input]
            label = [ele.to(self.device) for ele in label]
            output = self.model(input)
            bat_loss = self.loss_fn(output, label)
            bat_loss_avg = torch.mean(bat_loss)
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
                input, label = data
                input = [ele.to(self.device) for ele in input]
                label = [ele.to(self.device) for ele in label]
                output = self.model(input)
                bat_val_loss = self.loss_fn(output, label)
                bat_val_loss_avg = torch.mean(bat_val_loss)
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
            print("Saved new model")
        else:
            self.early_stop_count += 1

def main(args):
    t = trainer(args)
    t.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default='./configs/train.json',
                        help='The path to the config file. e.g. python train.py -c configs/train.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "./tensorboard/"+args.model_name+"/"+current_time
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)
    main(args)
    writer.close()