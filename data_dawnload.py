from torchvision import datasets

datasets.MNIST(root="data/MNIST", train=True,  download=True)
datasets.MNIST(root="data/MNIST", train=False, download=True)
datasets.SVHN(root="data/SVHN", split="train", download=True)
datasets.SVHN(root="data/SVHN", split="test",  download=True)
