import h5py
import sys

def check_shape(h5_path):
    with h5py.File(h5_path, 'r') as f:
        print(f"File: {h5_path}")
        for k in f.keys():
            ds = f[k]
            print(f"  {k}: {ds.shape}, dtype={ds.dtype}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_shape(sys.argv[1])
