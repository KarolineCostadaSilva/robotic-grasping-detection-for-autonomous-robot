import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class GraspNet(nn.Module):
    def __init__(self, classes=None):
        super(GraspNet, self).__init__()
        self.classes = classes
        self.n_classes = 20 # len(classes)
        
        # vgg16 = models.vgg16(pretrained=True)
        wide_resnet101 = models.wide_resnet101_2(pretrained=True, progress= True)
        #print('vgg16: {}'.format(vgg16))
        
        # until the maxpool_5
        # self.GraspNet_base = nn.Sequential(*list(wide_resnet101.features._modules.values())[:])
        # Fix the layers before conv3?:
        #for layer in range(10):
            #for p in self.GraspNet_base[layer].parameters(): p.requires_grad = False
        
        # Remove the last layer (fully connected)
        modules = list(wide_resnet101.children())[:-1]  # Remove the last fc layer
        self.GraspNet_base = nn.Sequential(*modules)

        # Since we have removed the last layer, we use the output features of the second last layer
        num_features = wide_resnet101.fc.in_features
          
        # Remove the last fc, fc8 pre-trained for 1000-way ImageNet classification. Use the * operator to expand the list into positional arguments
        # self.GraspNet_classifier = nn.Sequential(*list(wide_resnet101.classifier._modules.values())[:-1])

        # Replace the classifier with a new one (omitting the last layer as well)
        self.GraspNet_classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # Normally another Linear layer would be here, but we're customizing below
        )
        
        # Create a new module having 2 FCs (identical for FC6 and FC7 in VGG16)
        # self.GraspNet_regressor = nn.Sequential(*list(wide_resnet101.classifier._modules.values())[:-1])
        
        # Custom regressor (similar structure as the classifier)
        self.GraspNet_regressor = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        # Create 2 fc8s
        # fc8_1
        # self.GraspNet_cls_score = nn.Linear(4096, self.n_classes)
        # fc8_2
        # self.GraspNet_rect_pred = nn.Linear(4096, 4)

        # Create two separate classifiers for the class score and bounding box regression
        self.GraspNet_cls_score = nn.Linear(4096, self.n_classes)
        self.GraspNet_rect_pred = nn.Linear(4096, 4)
        
        self._init_weights()
        
        
    def forward(self, img):
        base_feat =  self.GraspNet_base(img)
        base_feat = base_feat.view(base_feat.size(0), -1)
        fc7 = self.GraspNet_classifier(base_feat)
        
        fc7_reg = self.GraspNet_regressor(base_feat)
        
        # compute angle bin classification probability
        cls_score = self.GraspNet_cls_score(fc7)
        #cls_prob = F.softmax(cls_score, 1)

        rect_pred = self.GraspNet_rect_pred(fc7_reg)

        return rect_pred, cls_score
        
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.GraspNet_cls_score, 0, 0.01, True)
        normal_init(self.GraspNet_rect_pred, 0, 0.001, True)
    
    def create_architecture(self):
        self._init_weights()  # 2-specific-task fc layers  
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    model = GraspNet()








