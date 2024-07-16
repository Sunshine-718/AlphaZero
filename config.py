config = {
    "lr": 2e-4,
    "lr_multiplier": 1.0,
    "temp": 1.0,
    "n_playout": 50,
    "first_n_steps": 10,
    "c_puct": 4,
    "buffer_size": 2000,
    "batch_size": 512,
    "discount": 0.99,
    "play_batch_size": 1,
    "epochs": 5,
    "kl_targ": 0.02,
    "check_freq": 50,
    "game_batch_num": 100000,
    "best_win_ratio": 0.0,
    "pure_mcts_n_playout": 10,
    "soft_update_rate": 1e-2
}
