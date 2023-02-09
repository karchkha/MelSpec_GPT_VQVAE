
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 8,
    'epochs': 150,
    'test_nepoch': 5,
    'spec_dir_path':'./data/vas/features/*/melspec_10s_22050hz'
}
