training_config = {
    "lr": 3e-3,
    "temp": 1.0,
    "n_playout": 100,
    "first_n_steps": 1,
    "c_puct": 1.,
    "buffer_size": 10000,
    "batch_size": 64,
    "play_batch_size": 1,
    "pure_mcts_n_playout": 1000,
    "dirichlet_alpha": 0.1,
    "init_elo": 1500,
    "num_eval": 50,
    "win_rate_threshold": 0.65,
}

env_config = {'row': 3, 
              'col': 3}

network_config = {"in_dim": 3,
                  "h_dim": 128,
                  "out_dim": 9}