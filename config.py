config = {
    "lr": 1e-3,
    "min_lr": 2e-4,
    "lr_discount": 0.9999,
    "warmup_lr": 1e-4,
    "temp": 1.0,
    "n_playout": 100,
    "first_n_steps": 5,
    "c_puct": 4,
    "buffer_size": 10000,
    "batch_size": 128,
    "discount": 0.99,
    "play_batch_size": 1,
    "epochs": 5,
    "kl_targ": 0.02,
    "check_freq": 10,
    "pure_mcts_n_playout": 1000,
    "dirichlet_alpha": 0.03,   # depends on the size of action space
    "init_elo": 1500,
}

network_config = {"in_dim": 3,
                  "h_dim": 32,
                  "out_dim": 7}