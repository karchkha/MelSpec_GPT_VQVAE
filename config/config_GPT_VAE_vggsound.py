# ### GPT-medium

# params={
#     'vocab_size': 1024,
#     'block_size': 265,  # 53*5 + 1
#     'n_layer': 24,
#     'n_head': 16,
#     'n_embd': 1024,
#     'learning_rate': 1e-6,
#     'epochs': 10000,
#     'batch_size': 32,
#     'spec_dir_path':'./data/vggsound/melspec_10s_22050hz/',
#     'sample_rate': 22050,
#     'embd_pdrop': 0.0,
#     'resid_pdrop': 0.0,
#     'attn_pdrop': 0.0,
#     'n_unmasked': 0,
#     'last_linear': None,
# }

### GPT-Large

# params={
#     'vocab_size': 1024,
#     'block_size': 265,  # 53*5 + 1
#     'n_layer': 36,
#     'n_head': 20,
#     'n_embd': 1280,
#     'learning_rate': 1e-6,
#     'epochs': 10000,
#     'batch_size': 7,
#     'spec_dir_path':'./data/vggsound/melspec_10s_22050hz/',
#     'sample_rate': 22050,
#     'embd_pdrop': 0.0,
#     'resid_pdrop': 0.0,
#     'attn_pdrop': 0.0,
#     'n_unmasked': 0,
#     'last_linear': None,
# }

### GPT-XL

params={
    'vocab_size': 1024,
    'block_size': 265,  # 53*5 + 1
    'n_layer': 40,  #48
    'n_head': 23,   #25
    'n_embd': 1472, #1600
    'learning_rate': 1e-6,
    'epochs': 10000,
    'batch_size': 1,
    'spec_dir_path':'./data/vggsound/melspec_10s_22050hz/',
    'sample_rate': 22050,
    'embd_pdrop': 0.0,
    'resid_pdrop': 0.0,
    'attn_pdrop': 0.0,
    'n_unmasked': 0,
    'last_linear': None,
}