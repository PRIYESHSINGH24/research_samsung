import urllib.request
import ssl
import os
import sys

ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs('data', exist_ok=True)

# SVHN — torchvision se download hoga
# USPS — torchvision se download hoga

# Test download
from torchvision import datasets, transforms

print("Downloading SVHN...")
try:
    svhn = datasets.SVHN('data/', split='train', download=True,
                          transform=transforms.ToTensor())
    print(f"  ✅ SVHN Train: {len(svhn)}")
    svhn_test = datasets.SVHN('data/', split='test', download=True,
                               transform=transforms.ToTensor())
    print(f"  ✅ SVHN Test: {len(svhn_test)}")
except Exception as e:
    print(f"  ❌ SVHN failed: {e}")

print("\nDownloading USPS...")
try:
    usps = datasets.USPS('data/', train=True, download=True,
                          transform=transforms.ToTensor())
    print(f"  ✅ USPS Train: {len(usps)}")
    usps_test = datasets.USPS('data/', train=False, download=True,
                               transform=transforms.ToTensor())
    print(f"  ✅ USPS Test: {len(usps_test)}")
except Exception as e:
    print(f"  ❌ USPS failed: {e}")