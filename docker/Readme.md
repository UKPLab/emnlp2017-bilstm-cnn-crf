# Docker Environment
Docker allows an easy execution of the provided scripts by creating a container that installs all the needed dependencies. This folder contains a dockerfile that installs either a Python 3.6 environment with Keras 2.1.5 and TensorFlow 1.7.0 are installed. The environment can be used to run and train the provided BiLSTM-sequence tagger.


## Setup
First, install Docker as described in the documentation: https://docs.docker.com/engine/installation/

Then, build from the root folder of the repository the docker container. Run:
```
docker build ./docker/ -t bilstm 
```

This builds the Python 3.6 container and assigns the name *bilstm* to it. 

To run our code, we first must start the container and mount the current folder $PWD into the container:
```
docker run -it -v "$PWD":/src bilstm bash
```

The command `-v "$PWD":/src` maps the current folder `$PWD` into the docker container at the position `/src`. Changes made on the host system as well as in the container are synchronized. We can change / add / delete files in the current folder and its subfolder and can access those files directly in the docker container. 

Windows users can use instead of `$PWD` the command `%cd%` to get a path to current folder.


In this container, you can run execute the network as usual. For example to train the POS tagger, run:
```
python Train_POS.py
```
