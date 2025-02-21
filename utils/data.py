import numpy as np
from glob import glob

ROOT_TO_DATA = '/home/bdudas/PCT/data/npy_data'
NUM_WPTS = 10_000
    
class DataLoader:
    def __init__(self,rootData:str = ROOT_TO_DATA,numWPTS:int = NUM_WPTS,batch_size:int = 32, excludes = ['test'],ParticleNumber:int = 100):
        self.iterationsDone = 0
        self.batch_size = batch_size
        self.ParticleNumber = ParticleNumber
        
        wpt_dirs = glob(rootData + f'/wpt_*')
        for exlude in excludes:
            wpt_dirs = list(set(wpt_dirs) - set(glob(rootData + f'/wpt_*{exlude}')))    
        
        self.allPaths = np.array([f'{wpt_dir}/{i}.npy' for wpt_dir in wpt_dirs for i in range(numWPTS)])
        np.random.shuffle(self.allPaths)
        
    def getRandomElements(self,npPath):
        npData = np.load(npPath)
        randomIndexes = np.random.choice(npData.shape[0],self.ParticleNumber,replace=False)
        return npData[randomIndexes]

    def __get_item(self):
        res = np.array([self.getRandomElements(path) for path in self.allPaths[self.iterationsDone:self.iterationsDone+self.batch_size]])
        self.iterationsDone += self.batch_size
        return res

    def restart(self):
        self.iterationsDone = 0
        np.random.shuffle(self.allPaths)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterationsDone >= len(self.allPaths):
            self.restart()
            raise StopIteration
        return self.__get_item()
    
    def __len__(self):
        return len(self.allPaths) // self.batch_size
