import torch
from torch.utils.data import DataLoader
from grasp_dataset import GraspDataset
from network import GraspNet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torchvision
from torchvision import transforms
from shapely.geometry import Polygon

def Rotate2D(pts, cnt, ang):
    '''Rotates points(nx2) about center cnt(2) by angle ang(1) in radian'''
    return np.dot(pts-cnt, np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])) + cnt

def visualize_multi_grasps(image, grasps):
    plt.imshow(image.permute(1, 2, 0))
    ax = plt.gca()

    for grasp in grasps:
        pts = np.array([[grasp['x'], grasp['y']], [grasp['x'] + grasp['width'], grasp['y']],
                        [grasp['x'] + grasp['width'], grasp['y'] + grasp['height']],
                        [grasp['x'], grasp['y'] + grasp['height']]])
        cnt = np.array([(grasp['x'] + grasp['x'] + grasp['width']) / 2, (grasp['y'] + grasp['y'] + grasp['height']) / 2])
        
        r_bbox = Rotate2D(pts, cnt, grasp['angle'])
        pred_label_polygon = Polygon([(r_bbox[0,0], r_bbox[0,1]), (r_bbox[1,0], r_bbox[1,1]), 
                                      (r_bbox[2,0], r_bbox[2,1]), (r_bbox[3,0], r_bbox[3,1])])
        pred_x, pred_y = pred_label_polygon.exterior.xy

        plt.plot(pred_x, pred_y, color='r', alpha=0.7, linewidth=1, solid_capstyle='round', zorder=2)

    plt.show()

def convert_predictions_to_grasps(rect_preds, cls_scores, threshold=0.5):
    grasps = []
    for rect, score in zip(rect_preds, cls_scores):
        for s in score:
            if s > threshold:
                x, y, w, h = rect
                # Calcula o ângulo baseado na pontuação da classe (s)
                angle = -np.pi/2 - np.pi/20 * (s - 1)

                grasp = {
                    'x': x.item(),
                    'y': y.item(),
                    'width': w.item(),
                    'height': h.item(),
                    'angle': angle,
                    'score': s.item()
                }
                grasps.append(grasp)
    return grasps


def evaluate_model(dataset_path, image_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GraspDataset('grasp', image_set, dataset_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GraspNet().to(device)
    checkpoint = torch.load('./models/model_wideresnet_10199.ckpt', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    with torch.no_grad():
        for img, _ in data_loader:
            img = img.to(device)
            rect_preds, cls_scores = model(img)
            
            # Converta as previsões em uma lista de dicionários com as informações de apreensão
            grasps = convert_predictions_to_grasps(rect_preds, cls_scores)
            
            # Visualize as apreensões
            if isinstance(img, torch.Tensor):
                img = img.cpu().squeeze().numpy()  # Converte o tensor para NumPy se ainda for um tensor
            visualize_multi_grasps(img, grasps)
            # break  # Remova essa linha para processar o dataset inteiro


if __name__ == '__main__':
    dataset_path = './dataset/grasp'
    image_set = 'test'
    evaluate_model(dataset_path, image_set)
