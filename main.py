import omegaconf
from utils.data import DataLoader
from utils.model import PosPredictor,DistModel
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

DEFAULT_DATA_PATH = '/home/bdudas/PCT_tracking/data'


def createConfig(args,key1=None,key2=None,value=None):
    conf = omegaconf.OmegaConf.load('configs/baseConfig.yaml')
    ##
    ## Setting up the used dimensions
    conf.LoaderParams.dims = args.dims
    conf.ModelParams.in_dims = args.dims
    conf.LoaderParams.ParticleNumber = args.pnum 
    conf.ModelParams.out_neurons = conf.ModelParams.in_dims
    conf.ValLoaderParams = conf.LoaderParams
    conf.LoaderParams.norm = args.norm
    conf.ValLoaderParams.numWPTS = 2000

    ## Checking aviablity of GPU
    conf.deviceNum = args.gpu if args.gpu is not None else conf.deviceNum
    conf.ModelParams.targetLayer = args.targetLayer if args.targetLayer else conf.ModelParams.targetLayer
    
    pnum = conf.LoaderParams.ParticleNumber
    conf.ModelParams.numParticles = pnum    
    if key1 and key2 and value:
        conf[key1][key2] = value
    return conf

def prepModel(modelType:str,conf,device,trainloader,valLoader):
    if modelType =='pospred':
        model = PosPredictor(**conf.ModelParams)
        model.initSinkhornArgs(**conf.SinkhornParams)
        model.compile(optimizer=torch.optim.Adam,loss=torch.nn.MSELoss,device = device)
        fitkwgs = {'loader':trainloader,"epochs":conf.TrainingParams.epochs,
                   'valLoader':valLoader,'saveSelf':True,'replace':conf.TrainingParams.replace}
    else:
        model = DistModel(embedDim=conf.distModelParams.embedDim,particleNumber=conf.LoaderParams.ParticleNumber,inDims=conf.LoaderParams.dims)
        model.configure(optimizer=torch.optim.Adam,criterion=torch.nn.CrossEntropyLoss,targetLayer=conf.ModelParams.targetLayer,device=device,opt_kwgs={'lr':conf.TrainingParams.lr}
                        )
        fitkwgs = {'loader':trainloader,'numEpochs':conf.TrainingParams.epochs,'valloader':valLoader}
    return model,fitkwgs

def main(conf,mode:str = 'Train',modelType:str = 'pospred'):
    
    trainLoader = DataLoader(DEFAULT_DATA_PATH+"/train",**conf.LoaderParams)
    valLoader = DataLoader(DEFAULT_DATA_PATH+"/test",**conf.ValLoaderParams)
    conf.TrainWPTS = trainLoader.wpts
    conf.TestWPTS = valLoader.wpts
    
    device = torch.device(f'cuda:{conf.deviceNum}' if torch.cuda.is_available() else 'cpu')
    model, fitkwgs = prepModel(modelType=modelType,conf=conf,device=device,trainloader=trainLoader,valLoader=valLoader)
    if mode == 'Train':
        writer = SummaryWriter(comment ="_" + conf.comment)
        logdir = writer.log_dir
        model.add_logger(writer)
        with open(f'{logdir}/config.yaml', 'w') as f:
            omegaconf.OmegaConf.save(conf, f)
    
    model = torch.compile(model)
    model.fit(**fitkwgs)

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g','--gpu',type=int,default=None)
    argparser.add_argument('-c','--comment',type=str,default='')
    argparser.add_argument('-d','--dims',type=int,default=3) 
    argparser.add_argument('-t','--targetLayer',type=int,default=None)
    argparser.add_argument('-m','--mode',type=str,default='Train')
    argparser.add_argument('-p',"--pnum",type=int,default=200)
    argparser.add_argument('-n',"--norm",type=bool,default=True)

    args = argparser.parse_args()
    varconf = omegaconf.OmegaConf.load('configs/varConfig.yaml')

    conf = createConfig(args)
    embedDim = 256
    #for embedDim in [64,128,256,512]:
    conf.distModelParams.embedDim = embedDim
    conf.comment = args.comment + f"embed_dims{embedDim}"
    main(conf,mode = args.mode,modelType='Distmodel')