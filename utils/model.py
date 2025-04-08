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


@torch.no_grad()
def saveRandomSample(writer,x,y,epoch,mode='Train'):
    x =  x.cpu().numpy()
    y = y.cpu().numpy()
    fig = plt.figure()
    plt.scatter(y[:,0],y[:,1],label = 'True')
    plt.scatter(x[:,0],x[:,1],label = 'Predicted')
    plt.legend()
    writer.add_figure(f'{mode}/positions',fig,epoch)

@torch.no_grad()
def SavePredDistribution(writer,x,y,epoch,mode='Train',dims = 3):
    logFormat = lambda x : x.view(-1).cpu().numpy()
    dx = logFormat(x[:,:,0] - y[:,:,0])
    dy = logFormat(x[:,:,1] - y[:,:,1])
    dE = logFormat(x[:,:,2] - y[:,:,2]) if dims == 3 else None
    writer.add_histogram(f'{mode}/dx',dx,epoch)
    writer.add_histogram(f'{mode}/dy',dy,epoch)
    if dims == 3:
        writer.add_histogram(f'{mode}/dE',dE,epoch)

@torch.no_grad()
def acc(preds,target,axi):
    f = lambda x: torch.argmax(x,axis = axi)
    y_hat = f(preds)
    y_target = f(target)
    return y_hat.view(-1).eq(y_target.view(-1)).sum()/y_target.numel()


class Tracker():
    def __init__(self,temp:float = 0.59,n_iter:int = 10):
        self.temp = temp
        self.n_iter = n_iter

    @torch.no_grad()
    def getMatches(self,y_hat,y):
        distMX = torch.cdist(y_hat,y)
        S = SinkhornMatching(distMX,temp=self.temp,n_iter=self.n_iter)
        LikelyMatches = torch.argmax(S,dim=1)
        return LikelyMatches
    
    @torch.no_grad()
    def getMatchesSimple(self,y_hat,y):
        distMX = torch.cdist(y_hat,y)
        LikelyMatches = torch.argmin(distMX,dim=1)
        return LikelyMatches
    
    @torch.no_grad()
    def MatchingModule(self,y_hat,y,tryReplace = False):
        y,y_hat = y.cpu(),y_hat.cpu()
        if tryReplace:
            y[y == 0] = 1000
            y_hat[y_hat == 0] = 1000
        LikelyMatches = self.getMatches(y_hat,y)
        acc = y[torch.arange(y.shape[0]).unsqueeze(1),LikelyMatches].eq(y).sum().item()/y.nelement()
        return acc

class Trainer():
    def __init__(self,device:str,tracker,targetLayer:int,optimizer,loss):
        self.device = device
        self.targetLayer = targetLayer
        self.tracker = tracker
        self.optimizer = optimizer
        self.loss = loss

    def trainStep(self,model,data):
        model.train()
        xl = data[:,:,self.targetLayer+2]
        xr  = data[:,:,self.targetLayer+1]
        y = data[:,:,self.targetLayer]
        self.optimizer.zero_grad()
        y_pred = model.forward(xl, xr)
        loss = self.loss(y_pred,y)
        loss.backward()
        self.optimizer.step()
        return loss.item(),y_pred,y
    
    @torch.no_grad()
    def valStep(self,model,data):
        xl = data[:,:,self.targetLayer+2]
        xr  = data[:,:,self.targetLayer+1]
        y = data[:,:,self.targetLayer]
        y_pred = model.forward(xl,xr)
        loss = self.loss(y_pred,y)
        return loss.item(),y_pred,y
    
    def TrainModule(self,model,loader,epoch,writer,mode = 'Train',tryReplace = False):
        if mode == 'Train':
            model.train()
        else:
            model.eval()
        epochLoss,matchAcc = 0.0,0.0
        loopObj = tqdm(loader,colour='blue')
        descColor = "\033[93m"
        RESETColor = "\033[0m"
        postColor = "\033[92m"
        loopObj.set_description(descColor+f'{mode} Epoch: {epoch} TargetLayer: {self.targetLayer}'+ RESETColor)
        for step,data in enumerate(loopObj):
            loss,y,y_pred = self.trainStep(model,data.float().to(self.device)) if mode == 'Train' else self.valStep(model,data.float().to(self.device))
            stepAcc = self.tracker.MatchingModule(y_pred,y,tryReplace=tryReplace)
            currentStep = step + 1
            epochLoss += loss
            matchAcc += stepAcc
            fmt = lambda x : f'{x:.4f}'
            loopObj.set_postfix({f'{postColor}Loss':fmt(epochLoss/currentStep),'Acc':fmt(matchAcc/currentStep)})
            writer.add_scalar(f'{mode}/Loss',epochLoss/len(loader),epoch)
            writer.add_scalar(f'{mode}/Matching/Acc',matchAcc/len(loader),epoch)

            saveRandomSample(writer,y_pred,y,epoch,mode='Train')
            SavePredDistribution(writer,y_pred,y,epoch,mode)
        
    def fit(self,model,loader,epochs,writer, valLoader,replace = False):
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            self.TrainModule(model,loader,epoch,writer,mode = 'Train',tryReplace=  replace)
            self.TrainModule(model,valLoader,epoch,writer,mode = 'Validation',tryReplace=  replace)
        
        # Turn it into save the best model    
        torch.save(model.state_dict(),f'{writer.log_dir}/model.pth')

class PosPredictor(nn.Module):
    def __init__(self, in_dims:int, out_neurons:int, hidden1:int,hidden2:int):
        super(PosPredictor, self).__init__()
        self.dims = in_dims
        self.left1 = nn.Linear(in_dims, hidden1)
        self.leftact1 = nn.ReLU()
        self.left2 = nn.Linear(hidden1, hidden2)
        self.right1 = nn.Linear(in_dims, hidden1)
        self.rightact1 = nn.ReLU()
        self.right2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2,out_neurons)


    def forward(self, xl,xr):
        xl = self.left1(xl)
        xl = self.leftact1(xl)
        xl = self.left2(xl)

        xr = self.right1(xr)
        xr = self.rightact1(xr)
        xr = self.right2(xr)
        x = self.out(xl + xr)
        return x
    
class PosPred2(nn.Module):
    def __init__(self,in_dims:int, hidden:int,numLayers:int = 2):
        super(PosPred2,self).__init__()
        layers = [nn.Linear(in_dims*2,hidden),nn.ReLU()]
        for i in range(numLayers):
            layers.append(nn.Linear(hidden * (i+1),hidden * (i+2)))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden * (numLayers+1),in_dims))
        self.backbone = nn.Sequential(*layers)
    def forward(self,x1,x2):
        x = torch.cat((x1,x2),dim=-1)
        x = self.backbone(x)
        return x