import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

# Importando as classes definidas em outros arquivos
from grasp_dataset import GraspDataset
from network import GraspNet

# Configuração do dispositivo para treinamento (CPU ou GPU)
# device = torch.device("cpu")
device = torch.device("cuda")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    """
    Função principal que executa o treinamento do modelo.
    :param args: Argumentos de linha de comando para configuração do treino.
    """
    # Definição do conjunto de dados de treinamento
    dataset_name = 'grasp' # 'grasp' or 'cornell'
    dataset_path = './dataset/grasp' # path to dataset
    image_set = 'train' # 'train' or 'test'

    # Carregamento do conjunto de dados e definição do DataLoader
    dataset = GraspDataset(dataset_name, image_set, dataset_path)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) #, num_workers=4)
    # Inicialização de variáveis para armazenar as perdas
    loss_cls = 0
    loss_rect = 0
    # Inicialização do modelo e movendo-o para o dispositivo apropriado
    model = GraspNet() # Cria uma instância da classe GraspNet
    model = model.to(device) # Move o modelo para o dispositivo apropriado
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) # Cria um otimizador SGD
    # Iteração sobre os epochs
    total_step = len(train_loader) # Número total de batches
    for epoch in range(args.epochs): # Iteração sobre os epochs
        for i, (img, gt_rect) in enumerate(train_loader): # Iteração sobre os batches
            #img, gt_rect  = train_iter.next()
            # Preparando a entrada e as anotações para o modelo
            img = img.to(device)
            gt_cls = gt_rect[0]       # a batch of angle bin classes
            gt_cls = gt_cls.long()
            gt_cls = gt_cls.to(device) # Classes de ângulos
            gt_rect = gt_rect[1]    # a batch of rect coordinates
            gt_rect = gt_rect.float()
            gt_rect = gt_rect.to(device)  # Coordenadas das caixas delimitadoras
            
            #print('img.requires_grad: {}'.format(img.requires_grad))
            # Execução do modelo
            rect_pred, cls_score = model(img) 
            #print('rect_pred.requires_grad: {}'.format(rect_pred.requires_grad))
            # Cálculo da probabilidade da classe e da perda de classificação. 
            cls_prob = F.softmax(cls_score, 1)
            #print('cls_prob.requires_grad: {}'.format(cls_prob.requires_grad))
            
            loss_cls = F.cross_entropy(cls_score, gt_cls)
            
            #print('loss_cls.requires_grad: {}'.format(loss_cls.requires_grad))
            # Cálculo das ponderações internas da caixa delimitadora
            bbox_inside_weights = gt_rect.new(gt_rect.size()).zero_() # Cria um tensor de zeros com o mesmo tipo e tamanho de gt_rect
            
            for b in range(gt_cls.numel()): # Iteração sobre os elementos de gt_cls
                #print('gt_cls[b]: {0}'.format(gt_cls[b]))
                if gt_cls[b] != 0: # Se a classe não for 0 (ângulo 0), então a caixa delimitadora é considerada válida
                    bbox_inside_weights[b, :] = torch.tensor([1., 1., 1., 1.])  # 1.0 for valid boxes
                
                
            #print('bbox_inside_weights: {0}'.format(bbox_inside_weights))
            #print('rect_pred.shape: {0}, gt_rect.shape: {1}'.format(rect_pred.shape, gt_rect.shape))
            #print('rect_pred: {0} \n gt_rect: {1}'.format(rect_pred, gt_rect))
            # Aplicação de pesos às previsões e às anotações reais.
            gt_rect = torch.mul(gt_rect, bbox_inside_weights)
            rect_pred = torch.mul(rect_pred, bbox_inside_weights)
            
            # Cálculo da perda de regressão da caixa delimitadora e perda total
            loss_rect = F.smooth_l1_loss(rect_pred, gt_rect, reduction='mean')
            loss = loss_cls + loss_rect
            # Exibição das perdas
            print('epoch {}/{}, step: {}, loss_cls: {:.3f}, loss_rect: {:.3f}, loss: {:.3f}'.format(epoch + 1, args.epochs, i, loss_cls.item(), loss_rect.item(), loss.item()))
            
            # Backward and optimize
            # Backpropagation e otimização
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
    # Salvamento do modelo após o treinamento
    save_name = os.path.join(args.model_path, 'model_{}.ckpt'.format(epoch))
    torch.save({
      'epoch': epoch + 1,
      'model': model.state_dict(), # 'model' should be 'model_state_dict'
      'optimizer': optimizer.state_dict(),
      'loss': loss.item(),
    }, save_name)
    
    
def parse_arguments(argv):
    """
    Análise dos argumentos de linha de comando.
    :param argv: Argumentos da linha de comando.
    :return: Argumentos analisados.
    """
    parser = argparse.ArgumentParser() # Cria um objeto parser
    parser.add_argument('--epochs', type=int, help='number of epochs', default=1) # Adiciona um argumento --epochs
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001) # Adiciona um argumento --lr
    parser.add_argument('--batch-size', type=int, help='batch size', default=1) # Adiciona um argumento --batch-size
    parser.add_argument('--model-path', type=str, help='path to save model', default='./models') # Adiciona um argumento --model-path
    
    return parser.parse_args(argv) # Retorna os argumentos analisados

if __name__ == '__main__':
    # Chama a função main com os argumentos analisados
    main(parse_arguments(sys.argv[1:]))




































    
    
    
    
    
    
    
