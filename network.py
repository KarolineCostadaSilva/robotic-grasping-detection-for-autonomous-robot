import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary

# Classe GraspNet para modelagem de rede neural
class GraspNet(nn.Module):
    def __init__(self, classes=None):
        """
        Inicializa a rede GraspNet.
        :param classes: Lista de classes para classificação (opcional).
        """
        super(GraspNet, self).__init__() # Inicializa a classe base nn.Module
        self.classes = classes
        self.n_classes = 20 # len(classes) | Número de classes, assumido como 20
        
        # vgg16 = models.vgg16(pretrained=True)
        wide_resnet101 = models.wide_resnet101_2(pretrained=True)
        #print('vgg16: {}'.format(vgg16))
        
        # until the maxpool_5
        # self.GraspNet_base = nn.Sequential(*list(wide_resnet101.features._modules.values())[:])
        # Fix the layers before conv3?:
        #for layer in range(10):
            #for p in self.GraspNet_base[layer].parameters(): p.requires_grad = False
        
        # Remove the last layer (fully connected)
        # Removendo a última camada (fully connected) do Wide ResNet-101-2. 
        # modules = list(wide_resnet101.children())[:-2]  # Remove the last fc layer | Remove a última camada fc
        # self.GraspNet_base = nn.Sequential(*modules) # Base da rede
        self.GraspNet_base = nn.Sequential(*list(wide_resnet101.children())[:-1])

        # Since we have removed the last layer, we use the output features of the second last layer
        # Obtendo o número de características da camada anterior à removida. 
        num_features = wide_resnet101.fc.in_features

        # Adicionar uma camada de Adaptive Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          
        # Remove the last fc, fc8 pre-trained for 1000-way ImageNet classification. Use the * operator to expand the list into positional arguments
        # self.GraspNet_classifier = nn.Sequential(*list(wide_resnet101.classifier._modules.values())[:-1])

        # Replace the classifier with a new one (omitting the last layer as well)
        # Criando um novo classificador, omitindo a última camada do original. 
        self.GraspNet_classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # Normally another Linear layer would be here, but we're customizing below
        )
        
        # Create a new module having 2 FCs (identical for FC6 and FC7 in VGG16)
        # self.GraspNet_regressor = nn.Sequential(*list(wide_resnet101.classifier._modules.values())[:-1])
        
        # Custom regressor (similar structure as the classifier)
        # Criando um regressor personalizado com estrutura semelhante ao classificador. 
        self.GraspNet_regressor = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        # Create 2 fc8s
        # fc8_1
        # self.GraspNet_cls_score = nn.Linear(4096, self.n_classes)
        # fc8_2
        # self.GraspNet_rect_pred = nn.Linear(4096, 4)

        # Create two separate classifiers for the class score and bounding box regression
        # Criando dois classificadores separados: um para pontuação da classe e outro para regressão da caixa delimitadora. 
        self.GraspNet_cls_score = nn.Linear(4096, self.n_classes) # Para pontuação da classe
        self.GraspNet_rect_pred = nn.Linear(4096, 4) # Para previsão da caixa delimitadora
        # Inicialização dos pesos
        self._init_weights()
        
        
    def forward(self, img):
        """
        Define o fluxo de dados através da rede.
        :param img: Imagem de entrada.
        :return: Previsões da caixa delimitadora e pontuação da classe.
        """
        base_feat =  self.GraspNet_base(img) # Passa a imagem pela base da rede. 
        # base_feat = base_feat.view(base_feat.size(0), -1) # Achata os recursos
        # Passa os recursos pela camada classificadora e regressora
        # Aplicar o pooling aqui para reduzir as dimensões
        pooled_feat = self.avgpool(base_feat)
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1) # Achata os recursos

        cls_feat = self.GraspNet_classifier(pooled_feat)
        reg_feat = self.GraspNet_regressor(pooled_feat)
        
        # compute angle bin classification probability
        # Calcula a pontuação da classe e a previsão da caixa delimitadora
        cls_score = self.GraspNet_cls_score(cls_feat)
        #cls_prob = F.softmax(cls_score, 1)

        rect_pred = self.GraspNet_rect_pred(reg_feat)

        return rect_pred, cls_score # Retorna previsões da caixa delimitadora e pontuação da classe.
        
    def _init_weights(self):
        """
        Inicializa os pesos da rede.
        """
        def normal_init(m, mean, stddev, truncated=False):
            """
            Inicializador de peso: normal truncado e normal aleatório.
            :param m: Módulo a ser inicializado.
            :param mean: Média para a inicialização.
            :param stddev: Desvio padrão para a inicialização.
            :param truncated: Se True, usa inicialização normal truncada.

            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                # Inicialização normal truncada
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                # Inicialização normal aleatória
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_() # Zera os bias

        # Aplica inicialização normal aos classificadores e regressores
        normal_init(self.GraspNet_cls_score, 0, 0.01, True)
        normal_init(self.GraspNet_rect_pred, 0, 0.001, True)
    
    def create_architecture(self):
        """
        Cria a arquitetura da rede, inicializando os pesos.
        """
        self._init_weights()  # 2-specific-task fc layers  
    
    def unfreeze(self):
        """
        Descongela todos os parâmetros da rede para permitir o treinamento.
        """
        for param in self.parameters():
            param.requires_grad = True # Permite a atualização dos pesos durante o treinamento. 

if __name__ == '__main__':
    model = GraspNet() # Cria uma instância da GraspNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(3, 224, 224)) # Imprime um resumo da rede








