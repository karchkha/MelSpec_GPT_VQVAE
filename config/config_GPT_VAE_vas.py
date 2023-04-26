params={
    'vocab_size': 128,
    'block_size': 265,  # 53*5 + 1
    'n_layer': 24,
    'n_head': 16,
    'n_embd': 1024,
    'learning_rate': 1e-6,
    'epochs': 10000,
    'batch_size': 24,
    'spec_dir_path':'./data/vas/features/*/melspec_10s_22050hz',
    'sample_rate': 22050,
    'embd_pdrop': 0.3,
    'resid_pdrop': 0.3,
    'attn_pdrop': 0.3,
    'n_unmasked': 0,
    'last_linear': None,
}