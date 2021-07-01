import neuroNets as mnn

class _BaseCAM_wrapper:

    def __init__(self, model_dict):
        self.model_arch = model_dict['arch']
        if isinstance(self.model_arch.fc[-1], (mnn.checkValidation_sigmoid, mnn.checkValidation_softmax, mnn.checkValidation_logsoftmax)):
            self.model_arch.fc[-1] = nn.Softmax(dim=1)
        self.model_arch.eval()

        module_dict = dict(self.model_arch.named_modules())
        if model_dict['layer_name'] not in module_dict.keys(): raise Exception("Invalid target layer name.")
        self.target_layer = module_dict[model_dict['layer_name']]
        for p in self.target_layer.parameters(): p.requires_grad = True
        
        self.activations = None
        self.gradients = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)

    
    def forward(self, input, class_idx, ret_pred):
        raise NotImplementedError()

    
    def __call__(self, input, class_idx=None, ret_pred=False):
        return self.forward(input, class_idx, ret_pred)


class GradCAM_wrapper(_BaseCAM_wrapper):

    def forward(self, input, class_idx, ret_pred):
        
        logit = self.model_arch(input)
        
        if class_idx is None: class_idx = logit.max(1)[-1]
        
        self.model_arch.zero_grad()
        logit[:, class_idx].backward()

        GAP_grad = self.gradients.mean(dim=(2,3)).squeeze(0)
        conv_embedd = self.activations.squeeze(0)

        gcam = conv_embedd.permute(1,2,0)@GAP_grad

        gcam.relu_()

        gmin, gmax = gcam.min(), gcam.max()
        if gmin != gmax: gcam = (gcam - gmin)/(gmax-gmin)
        elif gmin != 0: gcam /= gmin
        
        gcam = gcam.cpu().detach().numpy()
        return (gcam, class_idx) if ret_pred else gcam