import numpy as np
import torch
from prettytable import PrettyTable
from tqdm import tqdm

#################################### TRAIN FUNCTIONS ####################################
import scoreUtilities as su

def train_nn(network_model, optimiser, training_criterion, scheduler=None, num_epochs=100, validation_criterion=None, save_res=True, verbose=False):    

    for ep in range(1, num_epochs+1):
        if verbose:
            if scheduler!=None: print(f"\n\n@epoch {ep}, optmising with: {scheduler.get_lr()}")
            else: print("\n\n@epoch {}, optmising with: {}".format(ep, [p_group["lr"] for p_group in optimiser.param_groups]))

        ########### TRAIN PHASE
        train_AVGbatchLoss = training_criterion.perform_epoch(epoch=ep, train_model=network_model, optimiser=optimiser, verbose=verbose)
        ########### VALIDATION PHASE
        valid_AVGbatchLoss = validation_criterion.perform_epoch(epoch=ep, valid_model=network_model, verbose=verbose) if validation_criterion!=None else None
        ########### LR UPDATE
        if scheduler: scheduler.step()
        ########### CHECKPOINT SAVING
        if save_res: save_results(ep, network_model, train_AVGbatchLoss, valid_AVGbatchLoss)


class Training():
    def __init__(self, train_loader, loss_funct, torch_device, reg_params=None):
        self.train_loader = train_loader
        self.loss_funct = loss_funct
        self.torch_device = torch_device
        self.reg_params = reg_params
        self.sens_pop = sorted((next(iter(train_loader))[0][1]).unique()) # assumes that in every batch there's at least an instance of each pop

        if self.reg_params!=None:
            if not isinstance(self.reg_params, dict) or len(self.reg_params)!=3: # criterion, param, layer
                raise ValueError("reg_param must be a dictionary instance with the definition of criterion, regularisation parameter and regularisation layer")
            if not isinstance(self.reg_params["criterion"], (FirstOrderMatching, SinkhornDistance, GaussMaximumMeanDiscrepancy)):
                raise ValueError("criterion must be one of the supported pytorch module")
            if self.reg_params["fair_param"]<0:# or self.reg_params["fair_param"]>1:
                raise ValueError("regularisation parameter must be >= 0")#,1]")
            if not isinstance(self.reg_params["fair_layer"], int) or self.reg_params["fair_layer"]<-1:
                raise ValueError("regularisation layer must be an index (>=-1) referring to a fully-connected classification layer (0=embeddings, -1=output)")
            assert len(self.sens_pop)==2, "non binary sensible population are not supported yet" # because of the fairness metrics

    def perform_epoch(self, epoch, train_model, optimiser, verbose=False):
        train_model.train()

        train_batchLoss = []

        for batch_idx, ((data, sens_attr), label) in tqdm(enumerate(self.train_loader)):
            if len(data)==0: continue

            assert len(sens_attr.unique())==len(self.sens_pop), f"train batch {batch_idx} miss an instance of a single demographic"

            data = data.to(self.torch_device)
            label = label.to(self.torch_device)

            train_batchLoss.append(np.empty((len(self.sens_pop)+1, 3 if self.reg_params!=None else 1)))

            # Task Loss
            if self.reg_params!=None:
                prediction, distribution = train_model(data, reg_layer=self.reg_params["fair_layer"])
            else:
                prediction = train_model(data)                    
            
            loss = self.loss_funct(prediction, label)

            # Fairness Loss (if defined) and Gradients Backpropagation
            if self.reg_params!=None:
                reg_constr = self.reg_params["criterion"](distribution[sens_attr==self.sens_pop[0]], distribution[sens_attr==self.sens_pop[1]])
                # reg_loss = (1-self.reg_params["fair_param"])*loss + self.reg_params["fair_param"]*reg_constr
                reg_loss = loss + self.reg_params["fair_param"]*reg_constr
                reg_loss.backward()
                train_batchLoss[batch_idx][0] = reg_loss.item(), loss.item(), reg_constr.item()        
            else:
                loss.backward()
                train_batchLoss[batch_idx][0] = loss.item()

            # Population Specific Loss
            for p_idx, sens_val in enumerate(self.sens_pop):
                pop_idx = sens_val==sens_attr
                pop_loss = self.loss_funct(prediction[pop_idx], label[pop_idx]).item()
                # train_batchLoss[batch_idx][p_idx+1] = ((1-self.reg_params["fair_param"])*pop_loss, pop_loss, 0) if self.reg_params!=None else pop_loss
                train_batchLoss[batch_idx][p_idx+1] = (0, pop_loss, 0) if self.reg_params!=None else pop_loss

            optimiser.step()
            optimiser.zero_grad()

            del prediction, loss
            if self.reg_params!=None: del distribution, reg_constr, reg_loss

            if verbose:
                print(f"\n@ epoch {epoch}, batch {batch_idx+1}/{len(self.train_loader)} -> train losses:")
                t = PrettyTable(["","Regolarised","Task","Constraint"] if self.reg_params else ["","Task Loss"])
                for i in range(len(train_batchLoss[batch_idx])):
                    t.add_row([f"{'All' if i==0 else f'Pop {i}'}", *np.around(train_batchLoss[batch_idx][i],4)])
                print(t)
            
        return np.around(np.mean(train_batchLoss, axis=0), decimals=4)


class Validation():
    def __init__(self, validation_loader, validation_metrics, torch_device):
        self.validation_loader=validation_loader
        self.torch_device = torch_device
        self.validation_metrics = validation_metrics
        self.sens_pop = sorted(torch.unique(next(iter(validation_loader))[0][1])) # assumes that in every batch there's at least an instance of each pop

        if not isinstance(self.validation_metrics, dict):
            raise ValueError("validation_metrics must be a dictionary with key=string name of the metric and value=the function definition of the metric")


    def perform_epoch(self, epoch, valid_model, verbose=False):
        valid_model.eval()

        valid_batchLoss = []
        
        with torch.no_grad():
            for batch_idx, ((data, sens_attr), label) in enumerate(self.validation_loader):
                if len(data)==0: continue

                assert len(torch.unique(sens_attr))==len(self.sens_pop), f"valid batch {batch_idx} miss an instance of a demographic"

                data = data.to(self.torch_device)
                label = label.to(self.torch_device)
                
                valid_batchLoss.append(np.empty((len(self.sens_pop)+1, len(self.validation_metrics))))

                prediction = valid_model(data) #general
                
                for m_idx, metric in enumerate(self.validation_metrics.values()):
                    if metric in (su.DDP, su.sum_cdfDDP, su.max_cdfDDP):    # binary fairness metrics
                        assert len(self.sens_pop)==2, "binary demographic metric for a non binary population"
                        valid_batchLoss[batch_idx][0, m_idx] = metric(prediction[sens_attr==self.sens_pop[0]], prediction[sens_attr==self.sens_pop[1]])
                    else:                                                   # general metrics
                        valid_batchLoss[batch_idx][0, m_idx] = metric(label, prediction)
                        for pop_idx, sens_val in enumerate(self.sens_pop):  # sub-population
                            idxs = sens_attr==sens_val
                            valid_batchLoss[batch_idx][pop_idx+1, m_idx] = metric(label[idxs], prediction[idxs])
                        
                if verbose:
                    print(f"\n@ epoch {epoch}, batch {batch_idx+1}/{len(self.validation_loader)} -> valid_metric:")
                    t = PrettyTable(["",*self.validation_metrics.keys()])
                    for i in range(len(valid_batchLoss[batch_idx])):
                        t.add_row([f"{'All' if i==0 else f'Pop. {i}'}",*np.around(valid_batchLoss[batch_idx][i],2)])
                    print(t)
        
        return np.around(np.mean(valid_batchLoss, axis=0), decimals=4)



################################## FAIRNESS METRICS ##################################

def distance_mean_loss(d1, d2, reduction="2Norm"):
    loss = d1.mean(dim=0)-d2.mean(dim=0)
    if reduction=="abs": loss = loss.abs()
    # elif reduction=="mean": loss = loss.mean(dim=0) # its not a loss because may be negative
    # elif reduction=="sum": loss = loss.sum(dim=0) # its not a loss because may be negative
    elif reduction=="2norm": loss = loss.norm(p=2)
    elif reduction=="squared2Norm": loss = loss.square().sum()
    # else: raise ValueError("reduction must be on of abs (for 1-dimensional vectors) or 2norm, squared2Norm (for d-dimensional vectors, d>1)")
    return loss

class FirstOrderMatching(torch.nn.modules.loss._Loss):
    __constants__ = ["reduction"]
    def __init__(self, size_average=None, reduce=None, reduction="abs"):
        super(FirstOrderMatching, self).__init__(size_average, reduce, reduction)
        if reduction not in ("abs","2norm","squared2Norm"):
            raise ValueError("Bad argument value for the reduction parameter [must be one of: abs, 2norm, squared2Norm]") #, mean, sum]")

    def __str__(self):
        return "FOM"

    def forward(self, distribution1, distribution2):
        return distance_mean_loss(distribution1, distribution2, reduction=self.reduction)


################################## SAVING & LOADING ##################################
from os.path import isfile

def save_results(epoch, crrnt_model, avg_trnLoss, avg_valLoss):
    torch.save({"epoch": epoch,
                #"model_stateDict": crrnt_model.state_dict(),
                "average_trainLoss": avg_trnLoss,
                "average_validLoss": avg_valLoss},
                f"./checkpoint/{crrnt_model.model_name}_ck{epoch}.tar")