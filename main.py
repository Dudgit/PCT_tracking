import omegaconf
from utils.data import DataLoader
from utils.model import PosPredictor
import torch
from torch.utils.tensorboard import SummaryWriter


def main(conf):
    trainLoader = DataLoader(**conf.LoaderParams)
    
    model = PosPredictor(**conf.ModelParams)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.compile(optimizer=torch.optim.Adam,loss=torch.nn.MSELoss,device = device)
    
    writer = SummaryWriter()
    logdir = writer.log_dir
    with open(f'{logdir}/config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(conf, f)
    
    model.add_logger(writer)
    model.fit(trainLoader,epochs = conf.TrainParams.epochs)

if __name__ == '__main__':
    conf = omegaconf.OmegaConf.load('configs/baseConfig.yaml')
    main(conf)