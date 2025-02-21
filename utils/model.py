import torch
import torch.nn as nn
import torch.nn.functional as F


class PosPredictor(nn.Module):
    def __init__(self, in_dims:int, out_neurons:int, hidden1:int,hidden2:int,targetLayer:int):
        super(PosPredictor, self).__init__()
        self.targetLayer = targetLayer
        self.left1 = nn.Linear(in_dims, hidden1)
        self.leftact1 = nn.ReLU()
        self.left2 = nn.Linear(hidden1, hidden2)

        self.right1 = nn.Linear(in_dims, hidden1)
        self.rightact1 = nn.ReLU()
        self.right2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2,out_neurons)

    def forward(self, xl,xr):
        # Check With Batchnorm too
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

    def compile(self,optimizer,loss,optimizer_params = {},loss_params = {},device = 'cpu'):
        self.optimizer = optimizer(self.parameters(),**optimizer_params)
        self.loss = loss(**loss_params)
        self.device = device
        self.to(device)
    
    def trainStep(self,data):
        xl = data[:,:,self.targetLayer+2]
        xr  = data[:,:,self.targetLayer+1]
        y = data[:,:,self.targetLayer]
        
        self.optimizer.zero_grad()
        y_pred = self.forward(xl,xr)
        loss = self.loss(y_pred,y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def log_scalar(self,msg,value,epoch):
        if hasattr(self,'logger'):
            self.logger.add_scalar(msg,value,epoch)
    
    #TODO: Add Validation And Scatter visualization
    def fit(self,loader,epochs):
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            epochLoss = 0
            for data in loader:
                loss = self.trainStep(torch.from_numpy(data).float().to(self.device))
                epochLoss += loss
                self.log_scalar('Train/Loss/Step',loss,epoch)
                print(f'Loss: {loss}',end = '\r')
            print(f'Loss: {epochLoss/len(loader)}') 
            self.log_scalar('Train/Loss/Epoch',epochLoss/len(loader),epoch)
