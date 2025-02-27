import numpy as np
from glob import glob

ROOT_TO_DATA = '/home/bdudas/PCT/data/npy_data'
NUM_WPTS = 10_000
    
class DataLoader:
    def __init__(self,rootData:str = ROOT_TO_DATA,numWPTS:int = NUM_WPTS,batch_size:int = 32, excludes = ['test'],ParticleNumber:int = 100,mode= 'Train',norm = True):
        self.iterationsDone = 0
        self.batch_size = batch_size
        self.ParticleNumber = ParticleNumber
        self.norm = norm
        wpt_dirs = glob(rootData + f'/wpt_*') if mode == 'Train' else glob(rootData + f'/wpt_*test')
        for exlude in excludes:
            wpt_dirs = list(set(wpt_dirs) - set(glob(rootData + f'/wpt_*{exlude}')))    
        if mode == 'val':
            for exclude in excludes:
                wpt_dirs = list(set(wpt_dirs) - set(glob(rootData + f'/wpt_{exclude}_test')))
        self.wpts = [wpt_dir.split('wpt_')[-1] for wpt_dir in wpt_dirs]
        self.allPaths = np.array([f'{wpt_dir}/{i}.npy' for wpt_dir in wpt_dirs for i in range(numWPTS)])
        np.random.shuffle(self.allPaths)
        
    def __getRandomElements(self,npPath):
        npData = np.load(npPath)
        randomIndexes = np.random.choice(npData.shape[0],self.ParticleNumber,replace=False)
        return npData[randomIndexes]

    def __get_item(self):
        res = np.array([self.__getRandomElements(path) for path in self.allPaths[self.iterationsDone:self.iterationsDone+self.batch_size]])
        self.iterationsDone += self.batch_size
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        if self.norm:
            res = normalize(res)
        return res

    def restart(self):
        self.iterationsDone = 0
        np.random.shuffle(self.allPaths)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterationsDone+self.batch_size >= len(self.allPaths):
            self.restart()
            raise StopIteration
        return self.__get_item()
    
    def __len__(self):
        return len(self.allPaths) // self.batch_size
