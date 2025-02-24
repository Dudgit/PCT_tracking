import omegaconf
from utils.data import DataLoader
from utils.model import PosPredictor
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse


def main(conf):
    trainLoader = DataLoader(**conf.LoaderParams)
    valLoader = DataLoader(**conf.ValLoaderParams)
    conf.TrainWPTS = trainLoader.wpts
    conf.TestWPTS = valLoader.wpts
    
    device = torch.device(f'cuda:{conf.deviceNum}' if torch.cuda.is_available() else 'cpu')
    
    writer = SummaryWriter(comment = conf.comment)
    logdir = writer.log_dir
    
    model = PosPredictor(**conf.ModelParams)
    model.initSinkhornArgs(**conf.SinkhornParams)
    model.compile(optimizer=torch.optim.Adam,loss=torch.nn.MSELoss,device = device)
    
    with open(f'{logdir}/config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(conf, f)
    
    model.add_logger(writer)
    model.fit(trainLoader,epochs = conf.TrainingParams.epochs,valLoader = valLoader,saveSelf=True)

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g','--gpu',type=int,default=0)
    argparser.add_argument('-c','--comment',type=str,default='')
    args = argparser.parse_args()
    
    conf = omegaconf.OmegaConf.load('configs/baseConfig.yaml')
    conf.ValLoaderParams.ParticleNumber = conf.LoaderParams.ParticleNumber
    conf.ModelParams.numParticles = conf.LoaderParams.ParticleNumber
    conf.deviceNum = args.gpu
    conf.comment = args.comment
    
    for SinkTemp in conf.SinkhornParams.SinkhornTemps:
        for pnum in conf.LoaderParams.ParticleNumbers:
            conf.LoaderParams.ParticleNumber = pnum
            conf.SinkhornParams.SinkhornTemp = SinkTemp
            conf.comment = f'SinkhornTemp_{SinkTemp}_ParticleNumber_{pnum}'
            main(conf)