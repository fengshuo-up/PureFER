import os
import sys
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import itertools
from networks.DDCM import DDCMNet
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

eps = sys.float_info.epsilon

class FERPlusDataSetCSV(Dataset):
    def __init__(self, data_path, csv_file, image_root, transform=None):

        self.transform = transform

        df = pd.read_csv(os.path.join(data_path, csv_file))
        file_names = df["Image name"]
        self.label = df["expression"].values

        if os.path.isabs(image_root):
            root = image_root
        else:
            root = os.path.join(data_path, image_root)

        self.file_paths = [os.path.join(root, f) for f in file_names]

        _, sample_counts = np.unique(self.label, return_counts=True)
        print(f'distribution of eval samples: {sample_counts}')

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.label

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = int(self.label[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class AttentionLoss(torch.nn.Module):
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
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)  # ,fontweight='bold'
    plt.yticks(tick_marks, classes, rotation=45, fontsize=12)

    fmt = '.1f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=fontsize, fontweight='bold')

    plt.tight_layout()

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']


def parse_args():
    parser = argparse.ArgumentParser(description='FERPlus confusion matrix')
    parser.add_argument('--fer_path', type=str, default=r'FERPlus dataset path',
                        help='FERPlus dataset root path.')
    parser.add_argument('--image_root', type=str, default=r'Images',
                        help='Image root under fer_path, or an absolute path to images.')
    parser.add_argument('--csv_file', type=str, default='PrivateTest.csv',
                        help='CSV filename for evaluation set.')
    parser.add_argument('--weights',type=str,default='FERPlus_acc0.9126.pth',
                        help='Path to pretrained weights file.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head in model.')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Dir to save confusion matrix svg file.')
    parser.add_argument('--save_name', type=str, default='FERPlus.svg',
                        help='Filename for saved confusion matrix svg.')
    return parser.parse_args()

def evaluate_and_plot():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DDCMNet(num_class=8, num_head=args.num_head).to(device)

    ckpt = torch.load(args.weights, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = FERPlusDataSetCSV(
        data_path=args.fer_path,
        csv_file=args.csv_file,
        image_root=args.image_root,
        transform=data_transforms_val
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True
    )
    print('Evaluation set size:', len(val_dataset))

    bingo_cnt = 0
    sample_cnt = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (imgs, targets) in tqdm(val_loader, desc='Evaluating'):
            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads, _ = model(imgs)

            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

            y_true.append(targets.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())

    acc = (bingo_cnt.float() / float(sample_cnt)).numpy()
    acc = np.around(acc, 4)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(7, 5))
    plot_confusion_matrix(
        matrix,
        classes=class_names,
        normalize=True,
        title=f'FERPlus Confusion Matrix (acc: {acc * 100:.2f}%)',
        fontsize=10
    )

    svg_path = os.path.join(args.save_dir, args.save_name)
    plt.savefig(svg_path, format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to: {svg_path}')

if __name__ == "__main__":
    evaluate_and_plot()
