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

## configs
Contains a config.yaml that is used for the basic configuration for the training.  
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

## Quick start