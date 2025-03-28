training_config = {
    "lr": 2e-4,
    "temp": 1.0,
    "n_playout": 100,
    "first_n_steps": 12,
    "c_puct": 1.5,
    "buffer_size": 100000,
    "batch_size": 64,
    "discount": 0.99,
    "play_batch_size": 1,
    "epochs": 4,
    "pure_mcts_n_playout": 1000,
    "dirichlet_alpha": 0.1,   # depends on the size of action space
    "init_elo": 1500,
    "num_eval": 50,
    "win_rate_threshold": 0.6,
}

env_config = {'row': 6, 
              'col': 7}

network_config = {"in_dim": 3,
                  "h_dim": 128,
                  "out_dim": env_config['col']}
