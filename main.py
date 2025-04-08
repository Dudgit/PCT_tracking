import omegaconf
from utils.data import Dataset
from utils.model import PosPredictor, Tracker, Trainer, PosPred2
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

DEFAULT_DATA_PATH = '/home/bdudas/PCT_tracking/data'


def createConfig(args):
    conf = omegaconf.OmegaConf.load('configs/baseConfig.yaml')
    ##
    ## Setting up the used dimensions 
    conf.ValLoaderParams = conf.LoaderParams
    conf.ValLoaderParams.numWPTS = 2000
    conf.deviceNum = args.gpu if args.gpu is not None else conf.deviceNum
    return conf


def main(conf):

    trainDataset = Dataset(DEFAULT_DATA_PATH+"/train",**conf.LoaderParams) 
    trainLoader = torch.utils.data.DataLoader(trainDataset,batch_size=conf.TrainingParams.batch_size,shuffle=True)
    valdataset = Dataset(DEFAULT_DATA_PATH+"/test",**conf.ValLoaderParams)
    valLoader = torch.utils.data.DataLoader(valdataset,batch_size=conf.TrainingParams.batch_size)

    device = torch.device(f'cuda:{conf.deviceNum}' if torch.cuda.is_available() else 'cpu')
    
    model = PosPred2(**conf.ModelParams)
    tracker = Tracker(**conf.SinkhornParams)
    optimizer = torch.optim.Adam(model.parameters(),lr=conf.TrainingParams.lr)
    loss = torch.nn.MSELoss()
    trainer = Trainer(device=device,tracker=tracker,optimizer=optimizer,loss=loss,targetLayer=conf.TrainingParams.targetLayer)

    writer = SummaryWriter(comment ="_" + conf.comment)
    logdir = writer.log_dir
    with open(f'{logdir}/config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(conf, f)
    sample = (next(iter(trainLoader)))[:,:,0]
    writer.add_graph(model,(sample,sample))
    
    #model = torch.compile(model)
    model = model.to(device)
    trainer.fit(model = model, loader=trainLoader,epochs=conf.TrainingParams.epochs,writer = writer,replace=conf.TrainingParams.replace,valLoader=valLoader)

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g','--gpu',type=int,default=None)
    argparser.add_argument('-c','--comment',type=str,default='')

    args = argparser.parse_args()

    conf = createConfig(args)
    for neurons in [32,64,128,256]:
        conf.ModelParams.hidden = neurons
        for numLayers in [1,2,3,4]:
            conf.ModelParams.numLayers = numLayers
            conf.comment = args.comment + f'_{numLayers=}_{neurons=}' 
            main(conf)