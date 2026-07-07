import urllib.request
import ssl
import os
import tarfile
import sys

ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs('data', exist_ok=True)

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filename = 'data/cifar-10-python.tar.gz'

def show_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    mb_done = count * block_size / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    sys.stdout.write(f'\r  Downloading: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)')
    sys.stdout.flush()

print('Downloading CIFAR-10...')
urllib.request.urlretrieve(url, filename, show_progress)
print('\nDownload done!')

print('Extracting...')
with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall('data/')
print('Done! CIFAR-10 ready!')