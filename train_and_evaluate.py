import sys
import argparse
import torch
from torch.utils.data import DataLoader
from grasp_dataset import GraspDataset
from network import GraspNet
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss, total_cls_loss, total_rect_loss = 0.0, 0.0, 0.0
    total_correct, total_samples = 0, 0

    for imgs, annotations in train_loader:
        imgs = imgs.to(device)
        gt_classes = annotations[0].long().to(device)
        gt_rects = annotations[1].float().to(device)

        optimizer.zero_grad()
        rect_pred, cls_score = model(imgs)

        loss_cls = F.cross_entropy(cls_score, gt_classes)
        bbox_weights = gt_rects.new(gt_rects.size()).zero_()
        bbox_weights[gt_classes != 0] = 1.0  # Pesando apenas as caixas válidas
        loss_rect = F.smooth_l1_loss(rect_pred * bbox_weights, gt_rects * bbox_weights, reduction='sum')

        loss = loss_cls + loss_rect
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_rect_loss += loss_rect.item()

        _, predicted_classes = cls_score.max(1)
        total_correct += (predicted_classes == gt_classes).sum().item()
        total_samples += gt_classes.size(0)

    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_rect_loss = total_rect_loss / len(train_loader)
    accuracy = (total_correct / total_samples) * 100

    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Rect Loss: {avg_rect_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def save_model(model, optimizer, epoch, args, avg_loss, accuracy):
    save_path = os.path.join(args.model_path, f'model_epoch_{epoch}.ckpt')
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy}, save_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--model_path', type=str, default='./models', help='Path to save model')
    return parser.parse_args(argv)

def main(args):
    dataset = GraspDataset('grasp', 'train', './dataset/grasp')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GraspNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(args.epochs):
        avg_loss, accuracy = train_epoch(model, train_loader, optimizer, epoch)
        save_model(model, optimizer, epoch, args, avg_loss, accuracy)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

# python train_and_evaluate.py --epochs 5 --lr 0.001 --batch_size 32 --model_path ./saved_models