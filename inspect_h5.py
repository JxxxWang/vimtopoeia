import h5py
import sys

try:
    with h5py.File('dataset_10k.h5', 'r') as f:
        print("Keys:", list(f.keys()))
        if 'mel_spec' in f:
            dset = f['mel_spec']
            print("Shape:", dset.shape)
            print("Chunks:", dset.chunks)
            print("Compression:", dset.compression)
            print("Compression opts:", dset.compression_opts)
            
            print("Attempting to read data...")
            data = dset[0]
            print("Read success. Shape:", data.shape)
except Exception as e:
    print(e)
