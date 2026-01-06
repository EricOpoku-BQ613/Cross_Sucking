import torch
from src.data.loaders import make_supervised_loader, make_ssl_loader

def main():
    sup = make_supervised_loader("data/manifests/train.csv", mode="train", batch_size=4, num_workers=0)
    x, y = next(iter(sup))
    print("[SUP]", x.shape, y.shape, y[:5])

    ssl = make_ssl_loader("data/manifests/unlabeled_ssl.csv", batch_size=4, num_workers=0)
    v1, v2 = next(iter(ssl))
    print("[SSL]", v1.shape, v2.shape)

    # Optional: move to GPU if available
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        v1 = v1.cuda(non_blocking=True)
        v2 = v2.cuda(non_blocking=True)
        print("[GPU OK]")

if __name__ == "__main__":
    main()
