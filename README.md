# PCT Tracking
This project is made for the Bergen pCT collaboration.

## Data
Data is coming from [GATE](http://www.opengatecollaboration.org/) simulations. The simulations were prepared for the pCT detector system.

## utils
### createTrainData
This file reads the output of the simulations and converts them into the format we are using.

### data
This is the python file that reads our data into a dataset.

### model 
This contains the model structures that we tried. Also the Tracker object, which is made to track particles and the Trainer object which helps us training models, so we don't have to contain the training loop in the main. This is very similar with the pytorch lightning trainer object.

## configs
Contains a config.yaml that is used for the basic configuration for the training

## main
In the main we are creating all the objects, such as the datasets, loaders, trainers and models.