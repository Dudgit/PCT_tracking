import numpy as np
import pandas as pd
from glob import glob
import os

rootDIR = '/home/bdudas/PCT_DATA2/output/'
NumExperiments = 10_000
TrainSize = 0.8
TestSize = 0.2

def padarray(A,size = 41):
    t = size - len(A)
    if t > 0:
        return np.pad(A, [(0,t),(0,0)], 'constant', constant_values=0)
    else:
        return A

def readhits(path:str):
    hit = pd.DataFrame(np.load(path))
    hit = hit[hit.parentID == 0]
    hit['Layer'] =  2*(hit['volumeID[2]'])+hit['volumeID[3]']
    hit.sort_values(by=['eventID','Layer'],inplace=True)
    return hit.loc[:,['eventID','posX','posY','edep','Layer']]

def filterhits(hit):
    # Group by eventID and check for non-unique Layer values
    non_unique_layers = hit.groupby('eventID')['Layer'].nunique() != hit.groupby('eventID')['Layer'].size()
    # Filter the eventIDs with non-unique Layer values
    non_unique_eventIDs = non_unique_layers[non_unique_layers].index
    # Display the eventIDs with non-unique Layer values
    #print("EventIDs with non-unique Layer values:", non_unique_eventIDs.tolist())
    return hit[~hit.eventID.isin(non_unique_eventIDs)]


def padhits(hit):
    return np.array([padarray(hit.loc[hit.eventID ==eid,['posX','posY','edep'] ]) for eid in hit.eventID.unique()] )

def main():
    wpts = [70,200]
    for wpt in wpts:
        hits = glob(rootDIR + f'wpt_{wpt}/*.npy')
        trainhits = hits[:int(TrainSize*NumExperiments)]
        testhits = hits[int(TrainSize*NumExperiments):]
        os.makedirs(f'/home/bdudas/PCT_tracking/data/train/wpt_{wpt}',exist_ok=True)
        os.makedirs(f'/home/bdudas/PCT_tracking/data/test/wpt_{wpt}',exist_ok=True)
        
        for i,hit in enumerate(trainhits):
            progression = i/len(trainhits)
            print(f'Progress: {progression*100:.2f}%',end='\r')
            hit = readhits(hit)
            hit = filterhits(hit)
            hit = padhits(hit)
            np.save(f'/home/bdudas/PCT_tracking/data/train/wpt_{wpt}/{i}.npy',hit)
        
        for i,hit in enumerate(testhits):
            hit = readhits(hit)
            hit = filterhits(hit)
            hit = padhits(hit)
            np.save(f'/home/bdudas/PCT_tracking/data/test/wpt_{wpt}/{i}.npy',hit)

if __name__ == '__main__':
    main()