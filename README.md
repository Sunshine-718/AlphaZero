# AlphaZero 
## Keywords
`AlphaGo Zero` `AlphaZero` `Monte Carlo Tree Search` `Reinforcement Learning (RL)` `Model-based RL` `Tree Search` `Heuristic Search` `Zero-sum Game`  

## Environment
[`python 3.12`](https://www.python.org)  
[`torch 2.3.0`](https://pytorch.org)  

## Get started
### Option 1  
Just simply run [`build.bat`](./build.bat) on Windows or [`build.sh`](./build.sh) on UNIX-like systems.  

### Option 2  
Run the command lines below in terminal.
``` shell
pip install -r requirements.txt
```
Pytorch official website: [click here](https://pytorch.org)  
If you prefer to run this code beyond cuda device, just simply run the code below to install the __CPU version__ of Pytorch.
``` shell
pip install torch
```  
Run [`setup.py`](./setup.py) by typing the command below in terminal  
```
python setup.py build_ext --inplace  
```  
## How to play against AlphaZero?
### 1. Play in terminal  
Type one of the command lines below in terminal:  
``` shell
python3 play.py --env Connect4 -x    # play as X
```
``` shell
python3 play.py --env Connect4 -o    # play as O
```
and input 0-6 for each column, i.e., 0 for the 1st column, 1 for the 2nd column.  
Mandatory argument:  
`--env`: Environment name, such as: `Connect4`  
Optional argument:  
`-x`: Play as X  
`-o`: Play as O  
`-n`: Number of simulation before AlphaZero make an action, set higher for more powerful policy (theoretically), default: `500`.  
`--self_play`: AlphaZero will play against itself if using this option.  
`--model`: current model or best model, default: `current`.  
`--name`: model name, default: `AlphaZero`.  
### 2. Play in GUI  
run `gui_play.py` to play Connect4 with AlphaZero in GUI.  
## How to train your own AlphaZero?
Open [`train.ipynb`](./train.ipynb) and run it __after backing up the latest parameter files__.  
__*IMPORTANT!!!*__:  
Remember to __back up__ the __latest__ parameter files, if the parameter files are corrupted or overwritten, it will take __several hours to 1 day__, even longer, to retrain the model from scratch, the training procedure is extremely slow!  
### How to monitor the training procedure?
The training procedure is monitored using tensorboard, you can open tensorboard by typing the command below:
```shell
tensorboard --logdir=runs
```
After running this command, open the browser and type [```http://localhost:6006/```](http://localhost:6006/) in the URL bar.  
## References
[Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).](https://doi.org/10.1038/nature24270)  

[David Silver et al. ,A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.](https://doi.org/10.1126/science.aar6404)   

## Future works
1. Implement MCTS in parallel. 
2. Add other board games
