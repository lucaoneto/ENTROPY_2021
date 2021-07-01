import pretrained_zoo as ptz
from itertools import groupby
from torch import nn, save, load, no_grad, tensor
import neuroNets as mnn
import pickle
import tenseal as ts
from tqdm import tqdm


class _adapted_features_CNN(nn.Module): # abstract class, embeddings extraction
    def __init__(self, model_name, req_grad, pretrained, feature_reduction=None):
        super(_adapted_features_CNN, self).__init__()

        features, classifier_inFeat = getattr(ptz, "vgg" if model_name in ["vgg11","vgg13","vgg16","vgg19"] else model_name)(pretrained=pretrained, model_name=model_name)
        if feature_reduction!=None:
            self.features = nn.Sequential(features,
                                                nn.Flatten(),
                                                nn.Linear(in_features=classifier_inFeat, out_features=feature_reduction, bias=True),
                                                nn.Sigmoid(),
                                                )
        else:
            self.features = nn.Sequential(features,
                                                nn.Flatten()
                                                )
        
        self.fc = nn.Linear(in_features=feature_reduction if feature_reduction!=None else classifier_inFeat, out_features=1, bias=True)
        self.fc_depth = 1

        if not isinstance(req_grad, bool): raise ValueError("req_grad parameter must be of boolean type")
        for param in self.parameters(): param.requires_grad = req_grad
        for p in self.features[1:].parameters(): p.requires_grad = True

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
        save(self.state_dict(), f"./data/models/{self.model_name}{name_extension}.pth")
    

    def load_neuralNet_full(self, torch_device, name_extension="", root_folder=""):
        self.load_state_dict(load(f"./data/models/{root_folder}{self.model_name}{name_extension}.pth", map_location=torch_device))
        return self

    def load_finalLayers(self, torch_device, reg=0, lr=.03, folder='crypt'):
        prefix = f"{str(reg).replace('.','d')}Reg_{str(lr).replace('.','d')}LR_{self.fc[0].in_features}-{self.fc[0].out_features}"
        with open(f"data/models/crypt_inference/mine-{folder}/biased9010Embeddings_biased9010Training/1000/{prefix}.pkl", 'rb') as handler: 
            load_dict = pickle.load(handler)
            with no_grad():
                self.fc[0].weight = nn.Parameter(load_dict["w1"].T.to(torch_device))
                self.fc[0].bias = nn.Parameter(load_dict["b1"].to(torch_device))
                self.fc[2].weight = nn.Parameter(load_dict["w2"].T.to(torch_device))
                self.fc[2].bias = nn.Parameter(load_dict["b2"].to(torch_device))
        return self
        
        
class Square(nn.Module):
    def forward(self, x):
        return x**2


class Adapted_fullyConn_classifier(_adapted_features_CNN):
    def __init__(self, model_name, dim_out, hiddens, pretrained=True, req_grad=False, sigm_activ=True, feature_reduction=False):
        super(Adapted_fullyConn_classifier, self).__init__(model_name, req_grad, pretrained, feature_reduction)
        
        modules = [nn.Linear(in_features=self.fc.in_features, out_features=dim_out, bias=True)]
        
        for neurons in hiddens:
            modules[-1] = nn.Linear(in_features=modules[-1].in_features, out_features=neurons, bias=True)
            modules.append(nn.Sigmoid() if sigm_activ==True else Square())
            modules.append(nn.Linear(in_features=neurons, out_features=dim_out, bias=True))
        
        # don't append a softmax layer for training because the CE loss in pytorch requires the raw ouput (it already includes a softmax)
        modules.append(mnn.checkValidation_softmax())
        
        self.fc = nn.Sequential(*modules)
        self.fc_depth = len(hiddens)+1

class CKKSTensor(ts.CKKSTensor):
    def decrypt(self):
        return super().decrypt().tolist()


class SquareActivation():
    @staticmethod
    def forward(x_in):
        return x_in.square()
    
    @staticmethod
    def prime(x_in):
        return 2*x_in

class SigmoidActivation():
    # We use the polynomial approximation of degree 3
    # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
    # from https://eprint.iacr.org/2018/462.pdf
    # which fits the function pretty well in the range [-5,5]
    @staticmethod
    def forward(x_in):
        return x_in.polyval([0.5, 0.197, 0, -0.004]) if isinstance(x_in, ts.tensors.abstract_tensor.AbstractTensor) else x_in.sigmoid()
    
    @staticmethod
    def prime(x_in):
        return x_in.polyval([0.197, 0, -0.012]) if isinstance(x_in, ts.tensors.abstract_tensor.AbstractTensor) else x_in.sigmoid()*(1-x_in.sigmoid())


class SquareError(object):
    @staticmethod
    def risk(pred_out, desired_out):
        return (pred_out-desired_out).square().sum(axis=1)

    @staticmethod
    def delta(pred_out, desired_out, activation_class=None, preactivation_values=None):
        # no need for the derivation if we don't have an activation after the last linear layer
        return (pred_out-desired_out) if activation_class==None else (pred_out-desired_out)*activation_class.prime(preactivation_values)

class CrossEntropyError(object):
    @staticmethod
    def risk(pred_out, desired_out):
        raise NotImplementedError("Trova approssimazione polinomiale del logaritmo (magari mclaurin visto che i valori dovrebbero essere in [0,1])")
        # sum(nan_to_num(-y*log(a)-(1-y)*log(1-a)))

    @staticmethod
    def delta(pred_out, desired_out, activation_class=None, preactivation_values=None):
        if isinstance(activation_class, Encr_sigmoidActivation): # exploit cross-entropy/sigmoid derivation trick if there is a sigmoid activation after the last linear layer
            return (pred_out-desired_out)
        else:
            raise NotImplementedError()



class EncryptedNN():
    
    def __init__(self, torch_nn=None, nodes=None, final_sigmoid=False, verbose=True):

        if torch_nn!=None:
            self.weight1 = torch_nn.linear1.weight.data.T
            self.bias1 = torch_nn.linear1.bias.data#.unsqueeze(0)
            self.weight2 = torch_nn.linear2.weight.data.T
            self.bias2 = torch_nn.linear2.bias.data#.unsqueeze(0)
        else:
            assert isinstance(nodes, (list,tuple)) and len(nodes)==3, "nodes must be a triple of integer (network 3-layers cardinalities)"
            linear = nn.Linear(in_features=nodes[0], out_features=nodes[1], bias=True)
            self.weight1 = linear.weight.data.T
            self.bias1 = linear.bias.data#.unsqueeze(0) #tbc
            linear = nn.Linear(in_features=nodes[1], out_features=nodes[2], bias=True)
            self.weight2 = linear.weight.data.T
            self.bias2 = linear.bias.data#.unsqueeze(0) #tbc

        self.activ1 = SquareActivation
        # not allowed for training, just inference from a pretrained network (not enough bitscale for performing all the backpropagation operations)
        if final_sigmoid: self.activ2 = SigmoidActivation

        self.is_encrypted = False

        # accumulate gradients and counts the number of iterations
        self._delta_w1 = 0
        self._delta_b1 = 0
        self._delta_w2 = 0
        self._delta_b2 = 0
        self._count = 0

        # for large batches of tensors (faster)
        # self._regdelta_w = 0

        # for small batches of (encrypted) tensors
        self._sum_inp_pop0 = 0
        self._sum_inp_pop1 = 0
        self._sum_out_pop0 = 0
        self._sum_out_pop1 = 0
        self._count_pop0 = 0
        self._count_pop1 = 0

        self.verbose = verbose

    def forward(self, x):
        x = x.mm(self.weight1) + self.bias1
        x = self.activ1.forward(x)
        x = x.mm(self.weight2) + self.bias2
        if getattr(self, "activ2", None): x = self.activ2.forward(x)
        return x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def test_evaluation(self, test_loader, fairness=False):
        acc = []
        fair = []

        for (x,s),y in test_loader:
            out = self(x)
            if isinstance(out, CKKSTensor): out = tensor(out.decrypt())
            if out.shape[1]>1: out = out.softmax(dim=1).argmax(dim=1)#, keepdim=True)
            acc.append((out==y).sum()/len(out))
            if fairness:
                fair.append(((out[s==0]==1).sum()/len(out), (out[s==1]==1).sum()/len(out)))

        if fairness:
            fair = tensor(fair).float().mean(dim=0)
            fair = round(100.*(fair[0]-fair[1]).abs().item(), 2)
            
        return round(100.*sum(acc).item()/len(acc),2), (fair if fairness else None)


    def encrypt(self, context):
        if self.is_encrypted: raise RuntimeError("The net is already encrypted")

        self.weight1 = CKKSTensor(context, self.weight1)
        self.bias1 = CKKSTensor(context, self.bias1)
        self.weight2 = CKKSTensor(context, self.weight2)
        self.bias2 = CKKSTensor(context, self.bias2)

        self.is_encrypted = True

    
    def decrypt(self):
        if not self.is_encrypted: raise RuntimeError("The net is not encrypted")

        self.weight1 = tensor(self.weight1.decrypt())
        self.bias1 = tensor(self.bias1.decrypt())
        self.weight2 = tensor(self.weight2.decrypt())
        self.bias2 = tensor(self.bias2.decrypt())

        self.is_encrypted = False


    def train(self, num_epochs, train_loader, criterion, train_net_context=None,
              learning_rate=.1, weight_decay=0, regularisation_strength=None,
              plain_ts_loader=None, fairness_test=False, save_prefix=''):
        
        self.regularised_training = regularisation_strength!=None
                
        if self.verbose and plain_ts_loader!=None:
            res = self.test_evaluation(plain_ts_loader, fairness_test)
            print(f"\nEpoch #0\n\tAccuracy: {res[0]}"+f"\tUnfairness: {res[1]}" if fairness_test else "")

        for epoch in range(1,num_epochs+1):
            
            self.tr_loss = 0

            for (x, s), y in tqdm(train_loader):
                self.backward(x, s, y, criterion)
                #return
            
            print(f"\nEpoch #{epoch}\n\tTrain Loss: {self.tr_loss/len(train_loader)}")
            if self.regularised_training:
                if train_net_context!=None:
                    meanRepr_pop0 = tensor(self._sum_out_pop0.decrypt())/self._count_pop0
                    meanRepr_pop1 = tensor(self._sum_out_pop1.decrypt())/self._count_pop1
                else:
                    meanRepr_pop0 = self._sum_out_pop0/self._count_pop0
                    meanRepr_pop1 = self._sum_out_pop1/self._count_pop1
                print(f"\tFair Loss: {(meanRepr_pop0-meanRepr_pop1).square().sum()}\n")
            else:
                print() # just for print spacing

            if train_net_context!=None: self.encrypt(train_net_context)
            self.update_parameters(lrn_rate=learning_rate, wght_decay=weight_decay, reg_str=regularisation_strength)
            if train_net_context!=None: self.decrypt()
            
            if self.verbose and plain_ts_loader!=None and epoch%5==0:
                res = self.test_evaluation(plain_ts_loader, fairness_test)
                print(f"\nEpoch #{epoch}\n\tAccuracy: {res[0]}"+f"\tUnfairness: {res[1]}" if fairness_test else "")

            self.save_network(lr=learning_rate, reg=regularisation_strength,
                              prefix=f"{'mine-plain' if train_net_context==None else 'mine-crypt'}/{save_prefix}") # store

    def backward(self, x, s, y, criterion):
        assert len(x.shape)>1, "input data must be in a matrix form [n_inputs, features]"
        z1 = x.mm(self.weight1) + self.bias1
        activ1 = self.activ1.forward(z1)
        z2 = activ1.mm(self.weight2) + self.bias2
        
        if isinstance(criterion, CrossEntropyError):
            assert isinstance(getattr(self, "activ2", None), Encr_sigmoidActivation), "The final activation need to be the polynomial approx of a sigmoid for exploiting the CE+sigmoid trick"
            activ2 = self.activ2.forward(z2)
            if self.verbose: print(criterion.risk(activ2, y))
            delta = (activ2 - y)    # CE + sigmoid activation trick
        
        elif isinstance(criterion, SquareError):
            # if self.verbose:
                # print(criterion.risk(z2, y))
                # tbd: print fairness loss if self.fairness_regularised is true
            #print(z2.shape, y.shape)
            loss = tensor(criterion.risk(z2, y).decrypt()) if isinstance(z2, CKKSTensor) else criterion.risk(z2, y)
            #print(loss)
            self.tr_loss += loss.mean(axis=0)
            
            delta = (z2 - y)        # MSE without final activation (ending linear layer)
        
        else:
            raise ValueError("Criterion must be one of SquareError or CrossEntropyError")

        # TBD: management of activ2
        
        self._delta_w2 += (activ1.transpose() if isinstance(activ1, CKKSTensor) else activ1.T).mm(delta)
        self._delta_b2 += delta.sum(axis=0)

        delta = delta.mm(self.weight2.T) * self.activ1.prime(z1)

        self._delta_w1 += (x.transpose() if isinstance(x, CKKSTensor) else x.T).mm(delta)
        self._delta_b1 += delta.sum(axis=0)

        self._count += x.shape[0]

        if self.regularised_training:
            if isinstance(x, CKKSTensor):
                assert x.shape[0]==1, "Subscript operator is not yet available for CKKSTensor, need to process batches of 1 element for fairness "
                
                #getattr(self, f"_sum_inp_pop{s}") += x.sum(axis=0)
                #getattr(self, f"_sum_out_pop{s}") += z1.sum(axis=0)
                #getattr(self, f"_count_pop{s}") += 1

                if s==0:
                    self._sum_inp_pop0 += x.sum(axis=0)
                    self._sum_out_pop0 += z1.sum(axis=0)
                    self._count_pop0 += 1
                elif s==1:
                    self._sum_inp_pop1 += x.sum(axis=0)
                    self._sum_out_pop1 += z1.sum(axis=0)
                    self._count_pop1 += 1
                else:
                    raise ValueError(f"Invalid value for the sensitive attribute: {s.item()}")
            
            else:
                pop0, pop1 = s==0, s==1

                # for large batches
                # inp_diff = (x[pop0].mean(axis=0) - x[pop1].mean(axis=0)).unsqueeze(1)
                # out_diff = 2*(z1[pop0].mean(axis=0) - z1[pop1].mean(axis=0)).unsqueeze(0)
                # self._regdelta_w += inp_diff.mm(out_diff)

                # for small batches
                self._sum_inp_pop0 += x[pop0].sum(axis=0)
                self._sum_inp_pop1 += x[pop1].sum(axis=0)
                self._sum_out_pop0 += z1[pop0].sum(axis=0)
                self._sum_out_pop1 += z1[pop1].sum(axis=0)
                self._count_pop0 += pop0.sum().item()
                self._count_pop1 += pop1.sum().item()


    def update_parameters(self, lrn_rate, wght_decay, reg_str):
        if self._count == 0: raise RuntimeError("You should at least run one forward iteration")
        # update weights: use a small regularization term to keep the output of the linear layer in the range of the sigmoid approximation
        self.weight1 -= self._delta_w1 * (lrn_rate / self._count) # + self.weight1 * wght_decay
        
        if self.regularised_training and reg_str>0:
            mean_inp = (self._sum_inp_pop0*(1/self._count_pop0) - self._sum_inp_pop1*(1/self._count_pop1)).reshape([self._sum_inp_pop0.shape[0], 1])
            mean_out = (self._sum_out_pop0*(1/self._count_pop0) - self._sum_out_pop1*(1/self._count_pop1)).reshape([1, self._sum_out_pop0.shape[0]])
            reg_delta = mean_inp.mm(2*mean_out)

            self.weight1 -= reg_delta * (lrn_rate * reg_str)

            # self.weight1 -= self._regdelta_w * (lrn_rate * reg_str)
        
        self.bias1 -= self._delta_b1 * (lrn_rate / self._count)
        self.weight2 -= self._delta_w2 * (lrn_rate / self._count) # + self.weight2 * wght_decay
        self.bias2 -= self._delta_b2 * (lrn_rate / self._count)

        # reset gradient accumulators and iterations count
        self._delta_w1 = 0
        self._delta_b1 = 0
        self._delta_w2 = 0
        self._delta_b2 = 0
        self._count = 0

        # self._regdelta_w = 0

        self._sum_inp_pop0 = 0
        self._sum_inp_pop1 = 0
        self._sum_out_pop0 = 0
        self._sum_out_pop1 = 0
        self._count_pop0 = 0
        self._count_pop1 = 0

    def save_network(self, lr, reg="no", prefix=""):
        save_dict = {"w1":self.weight1,"b1":self.bias1,"w2":self.weight2,"b2":self.bias2}

        with open(f"data/models/crypt_inference/{prefix}{str(reg).replace('.','d')}Reg_{str(lr).replace('.','d')}LR_{self.weight1.shape[0]}-{self.weight1.shape[1]}.pkl", 'wb') as handler: 
            pickle.dump(save_dict, handler)
    
    def load_network(self, lr, reg="no", prefix=""):
        with open(f"data/models/crypt_inference/{prefix}{str(reg).replace('.','d')}Reg_{str(lr).replace('.','d')}LR_{self.weight1.shape[0]}-{self.weight1.shape[1]}.pkl", 'rb') as handler: 
            load_dict = pickle.load(handler)
        self.weight1 = load_dict["w1"]
        self.bias1 = load_dict["b1"]
        self.weight2 = load_dict["w2"]
        self.bias2 = load_dict["b2"]
        
        return self


class FeatureExtractor(): # for torch dataset transformation
    def __init__(self, features_extractor, torch_device):
        self.features_extractor = features_extractor
        self.torch_device = torch_device

    def __call__(self, tensor):
        return self.features_extractor(tensor.to(self.torch_device).unsqueeze(0)).detach().cpu().numpy().squeeze(0)


def encrypt_data_loader(encryption_context, data_loader, encrypt_label=False):
    return [((CKKSTensor(encryption_context, data if len(data.shape)>1 else data.unsqueeze(0)),
              sensitive),
             CKKSTensor(encryption_context, label) if encrypt_label else label) for (data, sensitive), label in tqdm(data_loader)]
