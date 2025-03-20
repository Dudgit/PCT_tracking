import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad()
def SinkhornMatching(distMx,temp=0.59,n_iter=10):
    S = torch.softmax(-distMx/temp,dim=1)
    for i in range(n_iter):
        S /= torch.sum(S,dim=1,keepdim=True)
        S /= torch.sum(S,dim=2,keepdim=True)
    return S

###
### Batch -> B_szie (256) x NumParticles[50,100,150,200] x (posx,posY,energy)
###



class PosPredictor(nn.Module):
    def __init__(self, in_dims:int, out_neurons:int, hidden1:int,hidden2:int,targetLayer:int,numParticles:int):
        super(PosPredictor, self).__init__()
        self.dims = in_dims
        self.targetLayer = targetLayer
        self.numParticles = numParticles

        self.left1 = nn.Linear(in_dims, hidden1)
        self.leftact1 = nn.ReLU()
        self.left2 = nn.Linear(hidden1, hidden2)

        self.right1 = nn.Linear(in_dims, hidden1)
        self.rightact1 = nn.ReLU()
        self.right2 = nn.Linear(hidden1, hidden2)

        self.out = nn.Linear(hidden2,out_neurons)


    def initSinkhornArgs(self,temp=0.59,n_iter=10):
        self.temp = temp
        self.n_iter = n_iter

    def forward(self, xl,xr):
        xl = self.left1(xl)
        xl = self.leftact1(xl)
        xl = self.left2(xl)

        xr = self.right1(xr)
        xr = self.rightact1(xr)
        xr = self.right2(xr)

        x = self.out(xl + xr)
        return x

    def add_logger(self,logger):
        self.logger = logger
        x1Sample = torch.randn(256,self.numParticles,self.dims).to(self.device)
        x2Sample = torch.randn(256,self.numParticles,self.dims).to(self.device)
        self.logger.add_graph(self,(x1Sample,x2Sample))

    def compile(self,optimizer,loss,optimizer_params = {},loss_params = {},device = 'cpu'):
        self.optimizer = optimizer(self.parameters(),**optimizer_params)
        self.loss = loss(**loss_params)
        self.device = device
        self.to(device)

    def trainStep(self,data):
        self.train()
        xl = data[:,:,self.targetLayer+2]
        xr  = data[:,:,self.targetLayer+1]
        y = data[:,:,self.targetLayer]
        self.optimizer.zero_grad()
        y_pred = self.forward(xl, xr)
        loss = self.loss(y_pred,y)
        loss.backward()
        self.optimizer.step()
        return loss.item(),y_pred,y
    
    @torch.no_grad()
    def valStep(self,data):
        xl = data[:,:,self.targetLayer+2]
        xr  = data[:,:,self.targetLayer+1]
        y = data[:,:,self.targetLayer]
        y_pred = self.forward(xl,xr)
        loss = self.loss(y_pred,y)
        return loss.item(),y_pred,y

    @torch.no_grad()
    def getMatches(self,y_hat,y):
        distMX = torch.cdist(y_hat,y)
        S = SinkhornMatching(distMX,temp=self.temp,n_iter=self.n_iter)
        return S
    
    @torch.no_grad()
    def getMatchesSimple(self,y_hat,y):
        distMX = torch.cdist(y_hat,y)
        LikelyMatches = torch.argmin(distMX,dim=1)
        return LikelyMatches

    @torch.no_grad()
    def CorrectionMask(self,S):
        xpreds = torch.argmin(S,dim=1)
        ypreds = torch.argmin(S,dim=2)
        return xpreds == ypreds



    ############################# Matching Module #############################

        #distMx = torch.cdist(y_hat,y)
        #S = SinkhornMatching(distMx,temp=self.temp,n_iter=self.n_iter)
        #LikelyMatches = torch.argmax(S,dim=1)
    @torch.no_grad()
    def MatchingModule(self,y_hat,y,tryReplace = False):
        y,y_hat = y.cpu(),y_hat.cpu()
        if tryReplace:
            y[y == 0] = 1000
            y_hat[y_hat == 0] = 1000
        S = torch.cdist(y_hat,y)
        LikelyMatches = torch.argmin(S,dim=1)
        mask = self.CorrectionMask(S)
        purity,acc, numRec, = 0,0,0
        #Calculating accuracy by NVIDIA
        # x.eq(y.view_as(x)).sum().item()/x.nelement()
        for b in range(y.shape[0]):
            numReconstructed = torch.sum(mask[b])
            numRec += numReconstructed
            truepoints = y[b]
            truepoints_masked = truepoints[mask[b]]
            comparepoints = y[b,LikelyMatches[b]]
            comparepoints_masked = comparepoints[mask[b]]
    
            purity += torch.sum(comparepoints_masked == truepoints_masked)/(self.dims*numReconstructed)
            acc += torch.sum(comparepoints_masked == truepoints_masked)/(self.dims*y.shape[1])
        purity /= y.shape[0]
        acc /= y.shape[0]
        trackRatio =  float(self.dims*numRec)/y.nelement()
        numRec = float(numRec)/y.shape[0]
        return acc, numRec,  trackRatio, purity
    
    @torch.no_grad()
    def saveRandomSample(self,x,y,epoch,mode='Train'):
        x =  x.cpu().numpy()
        y = y.cpu().numpy()
        fig = plt.figure()
        plt.scatter(y[:,0],y[:,1],label = 'True')
        plt.scatter(x[:,0],x[:,1],label = 'Predicted')
        plt.legend()
        self.logger.add_figure(f'{mode}/positions',fig,epoch)

    @torch.no_grad()
    def SavePredDistribution(self,x,y,epoch,mode='Train'):
        logFormat = lambda x : x.view(-1).cpu().numpy()
        dx = logFormat(x[:,:,0] - y[:,:,0])
        dy = logFormat(x[:,:,1] - y[:,:,1])
        dE = logFormat(x[:,:,2] - y[:,:,2]) if self.dims == 3 else None
        self.logger.add_histogram(f'{mode}/dx',dx,epoch)
        self.logger.add_histogram(f'{mode}/dy',dy,epoch)
        if self.dims == 3:
            self.logger.add_histogram(f'{mode}/dE',dE,epoch)

    @torch.no_grad()
    def loggingMOdule(self,y_hat,y,acc,trackRatio,numUsed,purity,epoch,mode):
        self.saveRandomSample(y_hat[0],y[0],epoch,mode)
        self.SavePredDistribution(y_hat,y,epoch,mode)
        self.logger.add_scalar(f'{mode}/Matching/Acc',acc,epoch)
        self.logger.add_scalar(f'{mode}/Matching/TrackRatio',trackRatio,epoch)
        self.logger.add_scalar(f'{mode}/Matching/Tracks',numUsed,epoch)
        self.logger.add_scalar(f'{mode}/Matching/Purity',purity,epoch)



    def TrainModule(self,loader,epoch,mode = 'Train',tryReplace = False):
        epochLoss,matchAcc,TrackNums,dropRates,purity = 0.0,0.0,0.0,0.0,0.0
        loopObj = tqdm(loader,colour='blue')
        descColor = "\033[93m"
        RESETColor = "\033[0m"
        postColor = "\033[92m"
        loopObj.set_description(descColor+f'{mode} Epoch: {epoch} TargetLayer: {self.targetLayer}'+ RESETColor)
        for step,data in enumerate(loopObj):
            loss,y,y_pred = self.trainStep(torch.from_numpy(data).float().to(self.device)) if mode == 'Train' else self.valStep(torch.from_numpy(data).float().to(self.device))
            stepAcc, numTracks, dropRate,stepPurity = self.MatchingModule(y_pred,y,tryReplace=tryReplace)
            currentStep = step + 1
            epochLoss += loss
            matchAcc += stepAcc
            TrackNums += numTracks
            dropRates += dropRate
            purity += stepPurity
            fmt = lambda x : f'{x:.4f}'
            loopObj.set_postfix({f'{postColor}Loss':fmt(epochLoss/currentStep),'Acc':fmt(matchAcc.item()/currentStep),
                                 'TrackRatio':fmt(dropRates/currentStep),'NumTracks':fmt(TrackNums/currentStep),'Purity':fmt(purity/currentStep)+RESETColor})
        if hasattr(self,'logger'):
            self.logger.add_scalar(f'{mode}/Loss',epochLoss/len(loader),epoch)
            self.loggingMOdule(y_hat = y_pred,y=y,acc = matchAcc/len(loader),trackRatio=dropRates/len(loader),numUsed=TrackNums/len(loader),purity=purity/len(loader),epoch = epoch,mode=mode)
    
    def fit(self,loader,epochs, valLoader,saveSelf=False,replace = False):
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            
            self.TrainModule(loader,epoch,mode = 'Train',tryReplace=  replace)
            self.TrainModule(valLoader,epoch,mode = 'Validation',tryReplace=  replace)
        
        if saveSelf:
            torch.save(self.state_dict(),f'{self.logger.log_dir}/model.pth')
        