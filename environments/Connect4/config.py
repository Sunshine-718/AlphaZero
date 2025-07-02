training_config = {
    "lr": 1e-3,
    "temp": 1.0,
    "n_playout": 50,
    "first_n_steps": 6,
    "c_puct": 1.5,
    "buffer_size": 10000,
    "batch_size": 512,
    "play_batch_size": 1,
    "epochs": 5,
    "pure_mcts_n_playout": 1000,
    "dirichlet_alpha": 0.03,
    "init_elo": 1500,
    "num_eval": 50,
    "win_rate_threshold": 0.641,
}

env_config = {'row': 6, 
              'col': 7}

network_config = {"in_dim": 3,
                  "h_dim": 64,
                  "out_dim": env_config['col']}
