# Docker Environment
Docker allows an easy execution of the provided scripts by creating a virtual environment that installs all the needed dependencies. This folder contains a dockerfile that installs either a Python 2.7 (Dockerfile.Python_2.7) or Python 3.6 environment (Dockerfile.Python_3.6). In both versions, Keras 1.2.2 and TensorFlow 1.2.1 are installed. The environment can be used to run and train the provided BiLSTM-sequence tagger.


## Setup
First, install Docker as described in the documentation: https://docs.docker.com/engine/installation/

Then, build from the root folder of the repository the docker container. Run:
```
docker build ./docker/ -f ./docker/Dockerfile.python_3.6 -t bilstm 
```

This builds the Python 3.6 container and assigns the name *bilstm* to it. 

To run our code, we first must start the container and mount the current folder ${PWD} into the container:
```
docker run -it -v ${PWD}:/usr/src/app bilstm bash
```

The command `-v ${PWD}:/usr/src/app` maps the current folder ${PWD} into the docker container at the position `/usr/src/app`. Changes made on the host system as well as in the container are synchronized. We can change / add / delete files in the current folder and its subfolder and can access those files directly in the docker container. 



In this container, you can run execute the network as usual. For example to train the POS tagger, run:
```
python Train_POS.py
```
