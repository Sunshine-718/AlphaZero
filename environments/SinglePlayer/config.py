training_config = {
    "lr": 1e-3,
    "temp": 1.0,
    "n_playout": 100,
    "first_n_steps": 100,
    "c_puct": 1.5,
    "buffer_size": 100000,
    "batch_size": 128,
    "discount": 0.99,
    "play_batch_size": 1,
    "epochs": 1,
    "dirichlet_alpha": 0.1,   # depends on the size of action space
}
