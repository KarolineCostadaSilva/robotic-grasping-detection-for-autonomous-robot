# evaluate.py
import torch
from torch.utils.data import DataLoader
from grasp_dataset import GraspDataset
from network import GraspNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model(dataset_path, image_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GraspDataset('grasp', image_set, dataset_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GraspNet().to(device)
    checkpoint = torch.load('./models/model_wideresnet_10199.ckpt', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    all_gt_classes = []
    all_pred_classes = []

    with torch.no_grad():
        for img, gt_rect in data_loader:
            img = img.to(device)
            gt_cls = gt_rect[0].long().to(device)
            rect_pred, cls_score = model(img)
            predicted_classes = torch.argmax(cls_score, dim=1)
            all_gt_classes.extend(gt_cls.cpu().numpy())
            all_pred_classes.extend(predicted_classes.cpu().numpy())

    accuracy = accuracy_score(all_gt_classes, all_pred_classes)
    precision = precision_score(all_gt_classes, all_pred_classes, average='macro')
    recall = recall_score(all_gt_classes, all_pred_classes, average='macro')
    f1 = f1_score(all_gt_classes, all_pred_classes, average='macro')
    conf_matrix = confusion_matrix(all_gt_classes, all_pred_classes)

    print('Acurácia: {:.4f}'.format(accuracy))
    print('Precisão: {:.4f}'.format(precision))
    print('Recall: {:.4f}'.format(recall))
    print('F1-Score: {:.4f}'.format(f1))
    print('Matriz de Confusão:\n')

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Classes Preditas')
    plt.ylabel('Classes Verdadeiras')
    plt.show()
    plt.savefig('matriz_confusao.png')

if __name__ == '__main__':
    dataset_path = './dataset/grasp'
    image_set = 'test'
    evaluate_model(dataset_path, image_set)
