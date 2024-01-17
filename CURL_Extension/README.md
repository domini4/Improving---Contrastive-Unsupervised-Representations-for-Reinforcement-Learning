Name: James Dominic

Paper title: CURL: Contrastive Unsupervised Representations for Reinforcement Learning.

Improvements made:
1. Impliment episodic memory.
2. Add dynamic exploraion.

CURL Rainbow
=======
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

**Status**: Archive (code is provided as-is, no updates expected)

Implementation of [CURL: Contrastive Unsupervised Representations for
Reinforcement Learning](https://arxiv.org/abs/2004.04136) coupled with the [Data Efficient Rainbow method](https://arxiv.org/abs/1906.05243) for Atari
games.

Episodic memory implimentation is based on [Episodic Memory Deep Q-Networks](https://arxiv.org/pdf/1805.07603.pdf)

The code by default uses the 100k timesteps benchmark and has not been
tested for any other setting.

To install all dependencies, use the 'enviro_curl.yml' file to create a conda environment. To do so use the below command. Make sure to navigate into the project folder before running this command.

```
conda env create -f environment.yml
```

Now activate conda environments using below comment.

```
conda activate curl
```

Install torchvision dependency. This was triggering an error while packaging with the yml file.

```
conda install -c pytorch torchvision
```

Install atari_py older version.

```
conda install -c conda-forge gym-atari
```

Instal sklearn dependency.

```
conda install -c conda-forge sklearn-contrib-lightning
```

The commands to run each game is provded in the run_curl.sh file.

Example EMCURL run command

```
python main.py --game alien --eps-start 1
``` 

Example EMCURL-EPS run command

```
python main.py --game alien
```
