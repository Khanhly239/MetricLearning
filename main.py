from trainer import trainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on the CelebA dataset.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--idx', type=int, default=0, help='CUDA device index')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CelebA dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b0', help='Model architecture')
    parser.add_argument('--log', action='store_true', help='wandb logging')
    args = parser.parse_args()

    trainer(args=args)