import torch

class Tracker:
    def __init__(self,models = None):
        self.loader = None
        self.modelList = models if models is not None else None
        self.temp = .9
        self.n_iter = 10
    def add_loader(self, loader):
        self.loader = loader
    
    def match(self,x_prev,x_curr):
        distMX = torch.cdist(x_prev,x_curr,p=2)
        LikelyMatches = torch.argmin(distMX,dim=1)
        mask = self.CorrectionMask(distMX,x_prev)
        return LikelyMatches,mask
    
    
    def SinkhornMatch(x1,x2,temp = .9,n_iter:int = 10):
        distMX = torch.cdist(x1,x2,p=2)
        S = torch.softmax(-distMX/temp,dim=1)
        for _ in range(n_iter):
            S /= torch.sum(S,dim=1,keepdim=True)
            S /= torch.sum(S,dim=2,keepdim=True)
        mask = None#self.CorrectionMask(S,x_prev,Sinkhorn=True)
        res = torch.argmax(S,dim=1)
        return res,mask
    
    def CorrectionMask(self,S,x_prev,Sinkhorn:bool = False):
        ##
        ## Flag values where true has 0 values
        ## Or simply accept flags as args
        xpreds = torch.argmin(S,dim=1) if not Sinkhorn else torch.argmax(S,dim=1)
        ypreds = torch.argmin(S,dim=2) if not Sinkhorn else torch.argmax(S,dim=2)
        #x_prev[torch.arange(x_prev.size(0)).unsqueeze(1),xpreds] == 0
        zmask = x_prev == 0 #+x_prev[torch.arange(x_prev.size(0)).unsqueeze(1),ypreds] == 0
        return (xpreds == ypreds) + zmask.all(dim = -1)

    def step(self, x_curr,x1,x2,targetlayer,useSinkhorn:bool = False):
        x_prev = x1 if targetlayer >= len(self.modelList) else self.modelList[targetlayer](x1,x2)
        #x_prev = x1 if targetlayer > 1 else self.model1(x1,x2)
        res, mask = self.match(x_prev,x_curr) if not useSinkhorn else self.SinkhornMatch(x_prev,x_curr)
        return res, mask
    
    def scoreMatch(self,x_target,res,mask):
        acc = 0.0
        purity = 0.0
        for x,idx,m in zip(x_target,res,mask):
            acc += x.eq(x[idx].view_as(x)).sum().item()/x.nelement()
            #purity += x[m].eq(x[idx][m].view_as(x[m])).sum().item()/x[m].nelement()
        return acc/len(x_target),purity/len(x_target)
    
    def evalTracks(self,target,preds,masks,maxLayer,minLayer = 0):
        testMask = masks[:,:,minLayer:maxLayer]
        testMask = testMask == 1
        trackmatch = target == preds
        allMatches = trackmatch.all(axis = 2)
        outTracks = ~testMask.all(axis=2)
        acc = target.eq(preds).sum().item()/target.nelement()
        overall = torch.sum(allMatches+outTracks).item()/allMatches.nelement()
        numPureTracks = testMask.all(axis = 2).sum()//3
        numTracks = target.shape[1]*numPureTracks.item()/(target.shape[1]*target.shape[0])
        return acc, overall, numTracks
    
    def evalTrack(self,batch,useSinkhorn:bool = True ,temp = 0.1):
        tracks = torch.zeros(batch.shape)
        for targetLayer in range(39):
            x_target = batch[:,:,targetLayer]
            x1 = batch[:,:,targetLayer+1]
            x2 = batch[:,:,targetLayer+2]
            x_prev = x1 if targetLayer >= len(self.modelList) else self.modelList[targetLayer](x2,x1)
            SinkhornMatches,_ = SinkhornMatch(x_prev,x_target,temp= temp) if useSinkhorn else self.match(x_prev,x_target)
            reconstructed = x_target[torch.arange(x_target.size(0)).unsqueeze(1),SinkhornMatches]         
            tracks[:,:,targetLayer] = reconstructed
        y_true = batch[:,:,:]
        y_pred = tracks[:,:,:]
        comp = y_true == y_pred
        numPureTracks = comp.all(dim = -1).all(dim = -1).sum()
        pureTrackRatio = numPureTracks/(comp.shape[0]*comp.shape[1])
        print(f"Accurcy: {comp.sum()/(comp.numel())}")
        return y_true,y_pred
    
    def tracking(self,data,maxLayer,useSinkhorn:bool = False,minLayer:int = 0):
        target = data[:,:,minLayer:maxLayer]
        tracks = torch.zeros_like(data)
        masks = torch.zeros_like(data)
        resDict = {}
        for t in range(minLayer,maxLayer):
            x_curr = data[:,:,t]
            x_prev = data[:,:,t+1]
            x_prev2 = data[:,:,t+2]
            res,mask = self.step(x_curr,x_prev,x_prev2,t,useSinkhorn=useSinkhorn)
            tracks[:,:,t] =  x_curr[torch.arange(x_curr.size(0)).unsqueeze(1),res]
            masks[:,:,t] = torch.stack([mask,mask,mask],dim=-1)
            acc,purity = self.scoreMatch(x_curr,res,mask)
            resDict[t] = (acc,purity)
        preds = tracks[:,:,minLayer:maxLayer]
        acc,puretracks,numRemaining = self.evalTracks(target,preds,masks,maxLayer,minLayer)
        resDict['overall']= acc
        resDict['puretracks'] = puretracks
        resDict['numRemaining'] = numRemaining
        return resDict

        

def main():
    return 0


if __name__ == '__main__':
    main()
