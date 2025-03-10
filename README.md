# Proton Computed Tomography Particle Tracking
The aim of this module to help the Bergen pCT collaboration. This is tomography project for Hadron theraphy (an alternate radioton treatment for cancer). To be able to treat pacients efficiently we need to calculate the relative stopping power on the incoming protons (that are particles used during this theraphy).
For this reason the pCT collaboration built and currently develop a detector system. In this detector system we want to track the incoming particles, reconstruct their scattering angles and kinetic energy after their left the patient (or the so called phantom).  

In this repo I'm developing a system that aims to reconstruct particle paths in the pCT detector system. For data generation we used open GATE [TODO: cite] tool to generate the training and validation data. This is currently the standard in the Bergen pCT collaboration.

## Data
We use a specific configuration for the GATE tool, where we specified the detector system in which we want to track particles. The simulation logs and writes every meaningfull infomration into .npy files.  
The data used for training is generated from these files using, the "utils/create_data.py"
