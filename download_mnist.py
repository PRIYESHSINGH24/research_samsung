import urllib.request
import os
import gzip
import shutil
import ssl

# SSL fix for Windows
ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs('data/MNIST/raw', exist_ok=True)

files = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

for f in files:
    print(f'Downloading {f}...')
    urllib.request.urlretrieve(base_url + f, f'data/MNIST/raw/{f}')
    print(f'Extracting {f}...')
    with gzip.open(f'data/MNIST/raw/{f}', 'rb') as f_in:
        with open(f'data/MNIST/raw/{f[:-3]}', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f'Done!')

print('All files downloaded!')