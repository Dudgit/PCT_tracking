# PCT Tracking
This project is made for the Bergen pCT collaboration.

## Data
Data is coming from [GATE](http://www.opengatecollaboration.org/) simulations. The simulations were prepared for the pCT detector system. The folder is split into two main folders train and test. Every folder has subfolders named wpt_{number}. This stands for the water phantom thickness. The folders contains particle simulation data, that stores the tracks of the particle (every position they had in the detector system and their energy deposition when they hit the specified detector layer). If you load a certain datapoint its shape is going to be $N\times L\times d$, where $N$ stands for the number of particles belonging to a certain event (beaming in particles), $L$ stands for the detector layers and $d$ stands for the dimensions of the hit (currently it's just their position in compared to the middle of the detector layer and the deposited energy, later this might be changed).

## utils
### createTrainData
This file reads the output of the simulations and converts them into the format we are using.  

### data
This is the python file that reads our data into a dataset. In the constructor of this dataset one can specify the number of WPTs, which is the way I solved to specify how many samples I want to read from the data folders, and the number of particles read by the dataloader, since the detector system can read a certain amount of data at once (this is due to the detector read frequency, which is ~100 track). There is also an option to normalize the data or not (even though it might be included as a BatchNorm layer in the model) and a dimension specifier, meaning how much dimension do we want to use for the training.  
  
After one creates the dataset you can pass it to a pytorch dataloader.
```
mydataset = Dataset("data/train")  
train_loader = torch.utils.data.DataLoader(mydataset,batch_size=conf.TrainingParams.batch_size,shuffle=True)  
```

### model 
This contains the model structures that we tried. Also the Tracker object, which is made to track particles and the Trainer object which helps us training models, so we don't have to contain the training loop in the main. This is very similar with the pytorch lightning trainer object.  
  
The Tracker Object can be initialized by hyperparameters for the Sinkhorn matching [algorithm](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem). It uses 2 parameters, the temperature and the number of iterations. To the object two vector pair is passed and then it will yield accuracy of how well the algorithm can match these pairs.

TODO: Equations to explain.  

The trainer object is just a convinient way to solve the model training. It will take care of the data preparation, modell calling and model parameter updates. It will also log the data into tensorboard and use tqdm progress bar to show the training results.  

The model I'm currently using is a simple MLP model, that takes the positions of a particle along with it's previous positions and tries to predict it's position on the next detector layer (or target layer as I call it). The model is characterized by it's input neuron size and the number of linear layers it contains. It is possible that the model architecture will change to contain extra layers if I figure out some ways to include them in a way that it improves the accuracy.

## configs
Contains a config.yaml that is used for the basic configuration for the training.The whole training process is based on using this configuration file, every parameter of the training and the particle matching is stored in the config file. If a grid search is done, the changed config is saved next to the tensorboard event file that belongs to a specific training.  
The Loaderparams contains the inputs for our data loader:
``` 
  numWPTS (int): how many training/testing sample do we want to use.
  ParticleNumber (int): How many particles are passed to the model from a certain event.
  dims (int,range = [2,3]): How many dimensions of the data we want to pass to the model .
  norm (bool): To normalize the data or not.
```  
  
The ModelParams contains the information how to construct the model:
  ```
  in_dims (int): The dimension of the data, same as the dimension in data loader.
  hidden (int): Number of neurons, the dimension the first layer transforms our data into.
  numLayers (int) : How many Linear layers + activation should the model contain.
```

The TrainingParams contains some information how the training should go:
```
  batch_size (int): Simply the batch size.
  targetLayer (int): Which layer the model should project the positions.
  epochs (int): How many times the model should go through the whole dataset.
  replace (bool): Missing values are replaced with zeroes by default, but if we want to change them to bigger number, since 0 can be a possible value we can change this to True.
  lr (float): The learning rate of the training.
  ```

The SinkhornParams contains the hyperparameters for the Sinkhorn algorithm:
  ```
  temp (float): The temperature used in the Sinkhorn algorithm.
  n_iter (int): How many times should the Sinkhorn algorithm iterate it's normalization method.
  ```
## main
In the main we are creating all the objects, such as the datasets, loaders, trainers and models.
First we read the config file for the training and parse the input arguments. In this version I can add 2 arguments such as the gpu and the comment. Gpu argument specify which gpu I want to use (by ID) and the comment is just an extra comment appended to the logging directory.  

We pass the config to the main where based on these parameters the train and validation loaders are created as explained above.  
Then we initialize the model with the given hyperparameters read from the config file which we have to the model as key word arguments.
  ```
  model = PosPred2(**conf.ModelParams)
  ```
 Note that if I want to make any kind of grid search I just directly change these parameter before calling the main function.
We also initialize the tracker the same way as we did with the model:
  ```
  tracker = Tracker(**conf.SinkhornParams)
  ```  
And the trainer:
  ```
  trainer = Trainer(device=device,tracker=tracker,optimizer=optimizer,loss=loss,targetLayer=conf.TrainingParams.targetLayer)
  ```
Then we initialize the writer and save the config file there and the model architecture and then start the training with the trainer:
  ```
  trainer.fit(model = model, loader=trainLoader,epochs=conf.TrainingParams.epochs,writer = writer,replace=conf.TrainingParams.replace,valLoader=valLoader)
  ```

## Quick start
#### How to use an already existing training setup.
First we need to load the config file belonging to the specific training that we used.
  ```
  pathToConfig = 'runs/date_comment'
  conf = omegaconf.OmegaConf.load(f'{pathToConfig}/config.yaml')
  ```
Note that this path is actually the training path, probabbly in the runs directory.  
  
Then we initialize the dataset and the loader:
  ```
  myDataset = Dataset("data/test",**conf.LoaderParams) 
  dataLoader = torch.utils.data.DataLoader(myDataset,batch_size=conf.TrainingParams.batch_size)
  ```
We also need to load the model that was used for the certain experiment:
  ```
  model = PosPred2(**conf.ModelParams)
  model.load_state_dict(torch.load(f'{pathToConfig}/model.pth'))
  ```
And to obtain a sample prediction from the model:
  ```
  tmp = next(iter(dataLoader))
  xl = tmp[:,:,conf.TrainingParams.targetLayer+2]
  xr = tmp[:,:,conf.TrainingParams.targetLayer+1]
  y = tmp[:,:,conf.TrainingParams.targetLayer]
  model.eval()
  y_pred = model(xl,xr)
  ```
