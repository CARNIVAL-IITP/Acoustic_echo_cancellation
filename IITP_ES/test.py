from attrdict import AttrDict
from losses.loss_util import get_lossfns
from utils import AverageMeter
import argparse, data, json, models, numpy as np, os, time, torch
import glob
import soundfile as sf

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

class tester:
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
        self.model = self.init_model(args, args.model_name, args.model_options)
        print("Loaded the model...")
        self.feature_options = args.feature_options
        # build DataLoaders
        if args.dataset == "IITP_ES1":
            self.test_loader = data.IITP_ES_test_dataloader(args.model_name, args.feature_options, 'cv', args.cuda_option, self.device)

        # training options
        self.output_path = args.output_path+'/%s_%s_%s'%(self.model_name, self.dataset, self.loss_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def init_model(self, args, model_name, model_options):
        assert model_name is not None, "Model name must be defined!"
        assert model_name in ["HY_IITP_ESNet1"], \
            "Model name is not supported! Must be one of (HY_IITP_ESNet1)"
        if model_name == "HY_IITP_ESNet1":
            model = models.HY_IITP_ESNet1(model_options)
            folder_name = './output/%s_%s_%s'%(model_name, args.dataset, args.loss_option)            
            model_list = sorted(glob.glob(folder_name+'/*'), key=os.path.getctime)            
            fin_model = model_list[-1]            
            model.load_state_dict(torch.load(fin_model, map_location='cpu'))
            print(folder_name)
            print(model_name)
            print(fin_model)

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
        self.test()
        print("Model test is finished.")

    def test(self):
        times = AverageMeter()
        times.reset()
        len_d = len(self.test_loader)
        end = time.time()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):                
                input, infdat = data
                input = [ele.to(self.device) for ele in input]
                output = self.model(input)
                audio_out = output[0].squeeze().cpu().detach().numpy()
                out_path = self.output_path + '/out'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                fn = out_path + '/' + infdat[0][0]
                audio_out = audio_out.squeeze().T
                sf.write(fn, audio_out, self.feature_options.sampling_rate, subtype='PCM_16')
                times.update(time.time() - end)
                end = time.time()
                print('%d/%d, time estimated: %.2f seconds' % (i + 1, len_d, times.avg * len_d), end='\r')
        print("\n")
def main():
    parser = argparse.ArgumentParser(description='Parse the config path')
    parser.add_argument("-c", "--config", dest="path", default='./configs/test.json',
                        help='The path to the config file. e.g. python train.py -c configs/test.json')

    config = parser.parse_args()
    with open(config.path) as f:
        args = json.load(f)
        args = AttrDict(args)
    t = tester(args)
    t.run()

if __name__ == "__main__":
    main()
