import os
import sys
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import itertools
from networks.DDCM import DDCMNet
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sam import SAM
from ptflops import get_model_complexity_info
eps = sys.float_info.epsilon

class FERPlusDataSetCSV(Dataset):
    def __init__(self, data_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path

        if phase == 'train':
            df = pd.read_csv(os.path.join(self.data_path, 'Training.csv'))
            file_before = r"D:\dataset\datasets\FERPlus"
        else:
            df = pd.read_csv(os.path.join(self.data_path, 'PrivateTest.csv')) #PrivateTest.csv
            file_before = r"D:\dataset\datasets\FERPlus"

        file_names = df["Image name"]
        self.label = df["expression"]

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []

        for f in file_names:
            path = os.path.join(self.data_path, file_before, f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.label

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_path', type=str, default=r'D:\datasets\FERPlus', help='ferPlus-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=80, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    return parser.parse_args()


class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()

    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head - 1):
                for j in range(i + 1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt + 1
                    loss = loss + mse
            loss = cnt / (loss + eps)
        else:
            loss = 0
        return loss


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          fontsize=10,
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, rotation=45, fontsize=12)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=fontsize, fontweight='bold')

    plt.tight_layout()

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']

def run_training():
    args = parse_args()

    project_root = os.path.abspath(os.path.dirname(__file__))
    os.chdir(project_root)

    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DDCMNet(num_class=8, num_head=args.num_head)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomApply([transforms.RandomRotation(10), transforms.RandomCrop(112, padding=16)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = FERPlusDataSetCSV(args.fer_path, phase='train', transform=data_transforms)
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = FERPlusDataSetCSV(args.fer_path, phase='test', transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_at = AttentionLoss()

    params = list(model.parameters())
    optimizer = SAM(model.parameters(), torch.optim.SGD, lr=args.lr, rho=0.05, adaptive=False, weight_decay=1e-4,
                    momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads, _ = model(imgs)
            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

            loss.backward()
            optimizer.first_step(zero_grad=True)

            out, feat, heads, _ = model(imgs)
            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)
            loss.backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        # Save training history
        train_losses.append(running_loss.item())
        train_accuracies.append(acc.item())

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            y_true = []
            y_pred = []
            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                out, feat, heads, _ = model(imgs)

                loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)
                running_loss += loss

                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

                iter_cnt += 1

            running_loss = running_loss / iter_cnt
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            # Save validation history
            val_losses.append(running_loss.item())
            val_accuracies.append(acc.item())

            tqdm.write("[Epoch %d] Validation accuracy: %.4f. bacc: %.4f. Loss: %.3f" % (
            epoch, acc, balanced_acc, running_loss))
            tqdm.write("best_acc: " + str(best_acc))

            if acc > 0.9 and acc == best_acc:
                # Save model checkpoint
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join('checkpoints', f"1FERPlus_epoch{epoch}_acc{acc}_bacc{balanced_acc}.pth"))
                tqdm.write('Model saved.')

                # Compute confusion matrix and save
                matrix = confusion_matrix(y_true, y_pred)
                np.set_printoptions(precision=2)
                plt.figure(figsize=(7, 5))
                plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                      title=f'FERPlus Confusion Matrix (acc: {acc * 100:.2f}%)', fontsize=10)
                svg_path = os.path.join('checkpoints',
                                        f'1FERPlus_epoch{epoch}_acc{acc}_bacc{balanced_acc}.svg')
                plt.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    run_training()