import torch
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader


def download_minst_data(batch_size=128, data_dir="./data"):
    transforms = T.Compose(
        [T.ToTensor(), T.Normalize((0.5,), (0.5,))]
    )

    # Load MINST dataset if exists, otherwise download it
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transforms
    )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transforms
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("✅ MNIST download finished")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batches per epoch (train): {len(train_loader)}")

    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = download_minst_data()
    
    # Read a batch
    images, labels = next(iter(train_loader))
    print("One batch images shape:", images.shape)  # torch.Size([128, 1, 28, 28])
    print("One batch labels shape:", labels.shape)  # torch.Size([128])
    
    