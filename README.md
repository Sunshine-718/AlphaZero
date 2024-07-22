# AlphaZero for Connect Four  
## Environment
python 3.12.3 64-bit  
torch 2.3.0  

## How to install dependencies?
Run the command lines below in terminal.
``` shell
pip install numpy
pip install numba
pip install tqdm
```
Pytorch official site: [click here](https://pytorch.org)  
If you prefer to run this code beyond cuda device, just simply run the code below to install the cpu version of Pytorch.
``` shell
pip install torch
```

## How to play connect four against AlphaZero?
Type one of the command lines below in terminal:  
``` shell
python3 human_play.py -x    # play as X
python3 human_play.py -o    # play as O
```
and input 0-6 for each column, i.e 0 for the 1st column, 1 for the 2nd column.
Optional argument:  
`-s`: Number of simulation before AlphaZero make an action, set higher for more powerful policy (theoretically), default: 100.
## How to train your own AlphaZero?
Just open `train.ipynb` and run it __after backing up the latest parameter files__.  
__*IMPORTANT!!!*__:  
Remember to __back up__ the __latest__ parameter files, if the parameter files are corrupted or overwritten, it will take __several hours to 1 day__, even longer, to retrain the model from scratch, the training procedure is extremely slow!
## References
[Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).](https://doi.org/10.1038/nature24270)  

[David Silver et al. ,A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.](https://doi.org/10.1126/science.aar6404)  

[Connect Four - Wikipedia](https://en.wikipedia.org/wiki/Connect_Four)  


## Future works
1. Implement MCTS in parallel. 
