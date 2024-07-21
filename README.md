# AlphaZero for Connect Four  
## Environment
python 3.12.3 64-bit  
torch 2.3.0  

## How to install dependencies?
Run the code below on terminal.
```
pip install numpy
pip install numba
pip install tqdm
```
Pytorch official site: [click here](https://pytorch.org)
If you prefer to run this code beyond cuda device, just simply run the code below to install the cpu version of Pytorch.
```
pip install torch
```

## References
[Silver, D., Schrittwieser, J., Simonyan, K. et al. Mastering the game of Go without human knowledge. Nature 550, 354–359 (2017).](https://doi.org/10.1038/nature24270)
[David Silver et al. ,A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.](https://doi.org/10.1126/science.aar6404)
[Connect Four - Wikipedia](https://en.wikipedia.org/wiki/Connect_Four)


## Future works
1. Fix the negative leaf value problem in env.py, MCTS.py and MCTS_AZ.py
2. Implement MCTS in parallel. 
