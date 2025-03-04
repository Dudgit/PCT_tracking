import omegaconf
from utils.data import DataLoader
from utils.model import PosPredictor
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

DEFAULT_DATA_PATH = '/home/bdudas/PCT_tracking/data'


def createConfig(args):
    conf = omegaconf.OmegaConf.load('configs/baseConfig.yaml')
    #varconf = omegaconf.OmegaConf.load('configs/varConfig.yaml')
    conf.LoaderParams.dims = args.dims
    conf.ModelParams.in_dims = args.dims
    conf.ModelParams.out_neurons = conf.ModelParams.in_dims
    conf.ValLoaderParams.dims = args.dims

    conf.deviceNum = args.gpu
    conf.ModelParams.targetLayer = args.targetLayer if args.targetLayer else conf.ModelParams.targetLayer
    
    pnum = conf.LoaderParams.ParticleNumber
    conf.ValLoaderParams.ParticleNumber = pnum
    conf.ModelParams.numParticles = pnum

    conf.comment = args.comment if args.comment else  f"ParticleNumber_{int(pnum)}_targetLayer_{conf.ModelParams.targetLayer}"

    return conf

def main(conf,tqdm_disable = True,mode:str = 'Train'):
    
    trainLoader = DataLoader(DEFAULT_DATA_PATH+"/train",**conf.LoaderParams)
    valLoader = DataLoader(DEFAULT_DATA_PATH+"/test",**conf.ValLoaderParams)
    conf.TrainWPTS = trainLoader.wpts
    conf.TestWPTS = valLoader.wpts
    
    device = torch.device(f'cuda:{conf.deviceNum}' if torch.cuda.is_available() else 'cpu')
    
    model = PosPredictor(**conf.ModelParams)
    model.initSinkhornArgs(**conf.SinkhornParams)
    model.compile(optimizer=torch.optim.Adam,loss=torch.nn.MSELoss,device = device)
    
    if mode == 'Train':
        writer = SummaryWriter(comment ="_" + conf.comment)
        logdir = writer.log_dir
        model.add_logger(writer)
        with open(f'{logdir}/config.yaml', 'w') as f:
            omegaconf.OmegaConf.save(conf, f)
    
    model.fit(trainLoader,epochs = conf.TrainingParams.epochs,valLoader = valLoader,saveSelf=True,disable = tqdm_disable)

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g','--gpu',type=int,default=0)
    argparser.add_argument('-c','--comment',type=str,default='')
    argparser.add_argument('-d','--dims',type=int,default=3)
    argparser.add_argument('-t','--targetLayer',type=int,default=None)
    argparser.add_argument('-m','--mode',type=str,default='Train')

    args = argparser.parse_args()
    conf = createConfig(args)
   

    main(conf,mode = args.mode)