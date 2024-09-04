config = {
    "lr": 3e-4,
    "temp": 1.0,
    "n_playout": 100,
    "first_n_steps": 12,
    "c_puct": 1.5,
    "buffer_size": 50000,
    "batch_size": 128,
    "discount": 0.99,
    "play_batch_size": 1,
    "epochs": 5,
    "kl_targ": 0.02,
    "pure_mcts_n_playout": 1000,
    "dirichlet_alpha": 0.1,   # depends on the size of action space
    "init_elo": 1500,
}

network_config = {"in_dim": 3,
                  "h_dim": 128,
                  "out_dim": 7}
