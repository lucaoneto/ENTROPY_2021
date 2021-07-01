import numpy as np
import torch
from torch import nn
import pretrained_zoo as ptz
from itertools import groupby


class checkValidation_sigmoid(nn.Module):
    def forward(self, x):
        if not self.training:
            x = x.sigmoid()
            x = (x>=.5).type(torch.int)
        return x

class checkValidation_sign(nn.Module):
    def forward(self, x):
        if not self.training:
            x = x.sign()
        return x

class checkValidation_softmax(nn.Module):
    def forward(self, x):
        if not self.training:
            x = x.softmax(dim=1)
            x = x.argmax(dim=1)
        return x

class checkValidation_logsoftmax(nn.Module):
     def forward(self, x):
        if not self.training:
            x = x.log_softmax(dim=1)
            x = x.argmax(dim=1)
        return x


class Adapted_features_CNN(nn.Module): # abstract class, embeddings extraction
    def __init__(self, model_name, req_grad, pretrained, multiGPU):
        super(Adapted_features_CNN, self).__init__()

        features, classifier_inFeat = getattr(ptz, "vgg" if model_name in ["vgg11","vgg13","vgg16","vgg19"] else model_name)(pretrained=pretrained, model_name=model_name)
        self.features = nn.Sequential(features, nn.Flatten())

        if multiGPU and torch.cuda.device_count()>1:
            self.features = nn.DataParallel(self.features) # multi-gpus execution
            self.multiGPU = True
        else:
            self.multiGPU = False

        self.fc = nn.Linear(in_features=classifier_inFeat, out_features=1, bias=True)
        self.fc_depth = 1

        if not isinstance(req_grad, bool): raise ValueError("req_grad parameter must be of boolean type")
        for param in self.parameters(): param.requires_grad = req_grad

        self.model_name = f"adapted_{model_name}"
        

    def forward(self, x, reg_layer=None):
        x_split = self.features(x)

        if reg_layer!=None:
            assert isinstance(reg_layer, int) and reg_layer>=-1, "reg_layer should be an integer >=-1 stating the regualised layer (-1 for full output, 0 for features)"
            if reg_layer==-1:
                x_split = self.fc(x_split)
                x = x_split
            else:
                split = [key for key,_ in groupby(map(lambda np: int(np[0].split('.')[-2]), self.fc.named_parameters()))]
                split = split[reg_layer] if reg_layer<self.fc_depth else split[-1]+2 # not reg_layer-1: keeps the activation
                x_split = self.fc[:split](x_split)
                x = self.fc[split:](x_split)
        else:
            x = self.fc(x_split)
        
        return x if reg_layer==None else (x, x_split)


    def set_embedd_grad(self, grad_required, last_layers=0):
        if not isinstance(grad_required, bool): raise TypeError("The argument must be an instance of the bool type")
        if not isinstance(last_layers, int) or not last_layers>=0 : raise TypeError("The layers argument must be an integer greater or equal than 0")
        
        to_set = [key for key,_ in groupby((name.split('.')[-2] for name,_ in self.features[0].named_parameters()))][-last_layers:] if last_layers!=0 else []
        for name, data in self.features[0].named_parameters():
            if name.split('.')[-2] in to_set: data.requires_grad = grad_required
   

    def save_neuralNet(self, name_extension=""):
        torch.save(self.state_dict(), f"./data/models/{self.model_name}{name_extension}.pth")
    

    def load_neuralNet_full(self, torch_device, name_extension="", root_folder=""):
        self.load_state_dict(torch.load(f"./data/models/{root_folder}{self.model_name}{name_extension}.pth", map_location=torch_device))
        return self


    def load_neuralNet_features(self, torch_device, name_extension="", root_folder=""):
        filtered = [named_param 
                    for named_param in torch.load(f"./data/models/{root_folder}{self.model_name}{name_extension}.pth", map_location=torch_device).items()
                    if named_param[0].split('.')[0]=="features"]    # select only the parameters whose name starts with "features"
        filtered = {name[9:]:param for name,param in filtered}      # get rid of "features" from the names

        if self.multiGPU: self.features.module.load_state_dict(filtered)
        else: self.features.load_state_dict(filtered)
        return self



# FINETUNING CLASSIFICATION MODELS
class Adapted_fullyConn_classifier(Adapted_features_CNN):
    def __init__(self, model_name, dim_out, hiddens, pretrained=True, req_grad=False, sigm_activ=True, multiGPU=False):
        super(Adapted_fullyConn_classifier, self).__init__(model_name, req_grad, pretrained, multiGPU)
        
        modules = [nn.Linear(in_features=self.fc.in_features, out_features=dim_out, bias=True)]
        
        for neurons in hiddens:
            modules[-1] = nn.Linear(in_features=modules[-1].in_features, out_features=neurons, bias=True)
            #nn.init.normal_(modules[-1].weight, std=.01)
            #nn.init.zeros_(modules[-1].weight)
            #nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Sigmoid() if sigm_activ or dim_out!=1 else nn.Tanh())
            modules.append(nn.Linear(in_features=neurons, out_features=dim_out, bias=True))
        #nn.init.normal_(modules[-1].weight, std=.01)
        #nn.init.zeros_(modules[-1].weight)
        #nn.init.zeros_(modules[-1].bias)
        
        if dim_out==1:
            if sigm_activ: # don't append a sigmoid layer because it's more stable to use the combined Sigm+BCE loss during training
                modules.append(checkValidation_sigmoid())
            else:
                modules.append(nn.Tanh())
                modules.append(checkValidation_sign())
            modules.append(nn.Flatten(start_dim=0))
        else: # don't append a softmax layer because the CE loss in pytorch requires the raw ouput (it already includes a softmax)
            modules.append(checkValidation_softmax())
            # modules.append(checkValidation_logsoftmax())
            
        self.fc = nn.Sequential(*modules)
        self.fc_depth = len(hiddens)+1
        # self.output_range = (0,1) if sigm_activ or dim_out!=1 else (-1,1)
