import numpy as np
from glob import glob

ROOT_TO_DATA = '/home/bdudas/PCT/data/npy_data'
NUM_WPTS = 10_000
    
class Dataset():
    def __init__(self,rootData:str = ROOT_TO_DATA,numWPTS:int = NUM_WPTS,ParticleNumber:int = 100,norm = False,dims:int = 3):
        self.iterationsDone = 0
        self.ParticleNumber = ParticleNumber
        self.norm = norm
        self.dims = dims
        wpt_dirs = glob(rootData + f'/wpt_*')
        self.wpts = [wpt_dir.split('wpt_')[-1] for wpt_dir in wpt_dirs]
        self.allPaths = np.array([f'{wpt_dir}/{i}.npy' for wpt_dir in wpt_dirs for i in range(numWPTS)])

        
    def __getitem__(self,idx):
        npData = np.load(self.allPaths[idx])
        randomIndexes = np.random.choice(npData.shape[0],self.ParticleNumber,replace=False)
        npData = npData[randomIndexes]
        return self.__normalize__(npData[:,:,:self.dims]) if self.norm else npData[:,:,:self.dims]

    def __normalize__(self,x):
        return (x - x.min()) / (x.max() - x.min())

    def __iter__(self):
        return self
        
    def __len__(self):
        return len(self.allPaths)
