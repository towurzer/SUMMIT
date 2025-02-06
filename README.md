# SUMMIT
- Symbolic Unified Model for Multilingual Inference and Translation

# Overview
This git project consists of two parts, the python-based ML program with flask web-API and a Vue-based frontend.

# Docker
To just spin up the containers, it's required to have `docker` and `docker-compose` installed on your system. Launch the containers using:
```
docker compose up
```

# Prerequesites
Install `git` and `git-lfs`. After checking out the repository, run `git lfs fetch` and `git lfs pull` to download LFS files. There are no submodules to worry about.

# Application
Requirements:
* Install [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/get-started/locally/) on your system. The application was built with Python 3.12 and Cuda 12.6. 
  * To make life easier with parallel python installations, the use of either venv or [anaconda](https://anaconda.org/anaconda/python) is recommended. Make sure that `pip` is available and functioning.
* Install `cuda` packages for your system if you have an Nvidia GPU.
* Install pip packages:
    ```py
    pip install -r requirements_cpu.txt # for CPU only
    pip install -r requirements_cuda.txt # if CUDA is available on your system
    ```
    **Note for developers:**
    
    The `requirements.txt` file is used exclusively for building the docker image. Please do not use it.
    If a new package is required or updates recommended, please create new `requirements.txt` files using: `pip freeze > requirements.txt`

You can launch the application in evaluation mode using:
```
python src/main.py
```
This will also start the Web API.

**Be aware that for that to work out, existing tokenizers and model files are required!**

Those files should be placed into `train/model` and `train/tokenize` respectively, unless different paths are chosen in the `config.json` file. 

To put the model into training mode, run the application using an argument:
```
python src/main.py train
```
**Note:** Training is 10+ times faster with a GPU

# Website
The simple website is built using [Vue.js](https://vuejs.org) and its server [Vite](https://vite.dev). Both can easily be installed using `bun` (see below).
To install `bun`, make sure you have [nodeJS](https://nodejs.org/en) (current LTS, v22.13.1) installed. Multiple parallel `node` versions can be managed using [`nvm`](https://github.com/nvm-sh/nvm). After installation of `nvm` (see installation instructions on the site), make sure to install the current LTS version of `node`:
```
nvm ls-remote
nvm install --lts
nvm use --lts
```

Install [`bun`](https://bun.sh) either via `node` or directly on the system using the executable, the latter is recommended if using `nvm` as otherwise, `bun` might only be installed to the currently selected node instance.

Once all requirements are installed, install the node modules using:
```
bun install
```

The website can then be launched using:
```
bun dev
```

**Hint:** if this should throw errors, check the currently used `node` version using `node --version`