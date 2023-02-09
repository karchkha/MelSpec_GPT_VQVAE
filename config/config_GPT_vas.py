params={
    'vocab_size': 128,
    'block_size': 266,  # 53*5 + 1
    'n_layer': 24,
    'n_head': 16,
    'n_embd': 1024,
    'class_size': 8,   # number of classes
    'learning_rate': 1e-6,
    'epochs': 300,
    'batch_size': 8,
    'spec_dir_path':'./data/vas/features/*/melspec_10s_22050hz',
    'sample_rate': 22050,
    'embd_pdrop': 0.5,
    'resid_pdrop': 0.5,
    'attn_pdrop': 0.5,
    'n_unmasked': 0,
    'last_linear': None,

}