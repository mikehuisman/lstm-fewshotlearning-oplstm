import torch
import torch.nn.functional as F
import numpy as np

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task, get_info,\
                            ParamType
from .modules.similarity import gram_linear, cka


def regularizer_null(device, **kwargs):
    return torch.zeros(1, device=device)

def regularizer_l2(alfas, params, **kwargs):
    summ = torch.sigmoid(alfas[0]) * torch.sum(params[0]**2)
    for i in range(1, len(params)):
        summ = summ + torch.sigmoid(alfas[i//2]) * torch.sum(params[i]**2)
    return summ


# Regularizer for number of parameters
def regularizer_np(alfas, num_params, **kwargs):
    # num_params is a tensor [n1, n2, n3, n4] where ni is 
    # the num of params in layer i
    summ = torch.sigmoid(alfas[0]) * num_params[0]
    for i in range(1, len(alfas)):
        summ = summ + torch.sigmoid(alfas[i]) * num_params[i]
    return summ

def regularizer_l1(alfas, params, free_net=False, **kwargs):
    if free_net:
        i = 0
        loss = None
        for als in alfas:
            if len(als) > 1:
                probs = torch.softmax(als, dim=0)
            else:
                probs = torch.sigmoid(als)
            
            #print("PROBS:", probs)
            
            pls = params[i: i+len(als)]
            for a, p in zip(probs, pls):
                # print("probability:", a)
                # print("param:", p)
                # print("penalt:y", a*torch.norm(p, p=1))
                if loss is None:
                    loss = a * torch.norm(p, p=1)
                else:
                    loss = loss + a*torch.norm(p, p=1)
            i += len(als)
    return loss



def alfa_regularizer_entropy(alfas, params, free_net=False, **kwargs):
    # Punish uncertainty -> sparsify the paths
    if free_net:
        loss = None
        for als in alfas:
            if len(als) > 1:
                probs = torch.softmax(als, dim=0)
                penalty = torch.sum(torch.log(probs) * probs)
            else:
                probs = torch.sigmoid(als)
                penalty = torch.log(probs)*probs + (1-probs)*torch.log(1-probs)
            
            if loss is None:
                loss = penalty
            else:
                loss = loss + penalty
    return loss


def weight_entropy_regularizer(alfas, params, gammas, free_net=False, **kwargs):
    if free_net:
        loss = gammas[0]*regularizer_l1(alfas, params, free_net=free_net) +\
               gammas[1]*alfa_regularizer_entropy(alfas, params, free_net=free_net) 
    return loss

def max_binary_mask(distributions, device, **kwargs):
    max_masks = []
    for distr in distributions:
        max_masks.append( torch.zeros(2,device=device).float() )
        max_index = distr.argmax(dim=0)
        max_masks[-1][max_index] = 1
    return max_masks

def gumbel_binary_mask(distributions, **kwargs):
    binary_masks = []
    for i in range(len(distributions)):
        binary_masks.append( ( F.gumbel_softmax(torch.log(1e-6 + torch.softmax(distributions[i], dim=0)), hard=True) ) )
    return binary_masks

def soft_mask(distributions, temperature, **kwargs):
    # soft mask through a softmax
    masks = []
    for i in range(len(distributions)):
        masks.append( torch.softmax(distributions[i]/temperature + 1e-12, dim=0) )
    return masks

def create_grad_mask(grad_masks, device):
    hard_masks = []
    for p in grad_masks:
        params = torch.stack([p, torch.zeros(p.size(0), device=device)],dim=1)
        logits = torch.log(torch.softmax(params,dim=1)+1e-10)
        mask = F.gumbel_softmax(logits, hard=True)
        real_mask = mask[:,0]
        hard_masks.append(real_mask)
    return hard_masks


def create_deterministic_mask(grad_masks, **kwargs):
    hard_masks = []
    for p in grad_masks:
        real_mask = (p > 0).float()
        hard_masks.append(real_mask)
    return hard_masks

class ReptSAP(Algorithm):
    """Structured Adaptation Prior
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the base-learner
    baselearner_args : dict
        Dictionary of keyword arguments for the base-learner
    opt_fn : constructor function
        Constructor function for the optimizer to use
    T : int
        Number of update steps to parameters per task
    train_batch_size : int
        Indicating the size of minibatches that are sampled from meta-train tasks
    test_batch_size : int
        Size of batches to sample from meta-[val/test] tasks
    lr : float
        Learning rate for the optimizer
    validation : boolean
        Whether this model should use meta-validation
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    episodic : boolean
        Whether to sample tasks or mini batches for training
        
    Methods
    -------
    train(train_x, train_y, test_x, test_y)
        Perform a single training step on a given task
    
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dump the meta-learner state s.t. it can be loaded again later
        
    load_state(state)
        Set meta-learner state to provided @state 
    """
    
    def __init__(self, train_base_lr, gamma, base_lr, second_order, grad_clip=None, transform="interp", reg="num_params", 
                 meta_batch_size=1, sine_constr=None, var_updates=False, pretrain=False, freeze_init=False, freeze_transform=False, 
                 force_nopretrain=False, learn_alfas=False, exp1=False, solid=float("inf"), free_arch=False, relu=False, image=False, 
                 channel_scale=False, free_net=False, unfreeze_init=False, boil=False, svd=False, linear_transform=False, 
                 max_pool_before_transform=False, old=False, discrete_ops=False, trans_before_relu=False, train_iters=None, 
                 anneal_temp=False, soft=False, warm_start=0, tnet=False, avg_grad=False, swap_base_trans=False, 
                 use_grad_mask=False, alfa_lr= 1e-3, arch_steps=float("inf"), meta_lr_architecture=0.6, **kwargs):
        """Initialization of model-agnostic meta-learner
        
        Parameters
        ----------
        T_test : int
            Number of updates to make at test time
        base_lr : float
            Learning rate for the base-learner 
        second_order : boolean
            Whether to use second-order gradient information
        grad_clip : float
            Threshold for gradient value clipping
        meta_batch_size : int
            Number of tasks to compute outer-update
        **kwargs : dict
            Keyword arguments that are ignored
        """
        
        super().__init__(**kwargs)
        self.sine = False
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.grad_clip = grad_clip  
        self.second_order = second_order
        self.gamma = gamma
        self.meta_batch_size = meta_batch_size 
        self.log_test_norm = False
        self.disabled = False
        self.sine_constr = sine_constr
        self.var_updates = var_updates
        self.freeze_init = freeze_init
        self.freeze_transform = freeze_transform
        self.do_pretraining = pretrain
        self.learn_alfas = learn_alfas
        self.solid = solid
        self.free_arch = free_arch
        self.image = image # whether we are doing image classification
        self.channel_scale = channel_scale
        self.free_net = free_net
        self.unfreeze_init = unfreeze_init
        self.boil = boil
        self.svd = svd
        self.soft = soft
        self.avg_grad = avg_grad
        if self.boil: assert self.image, "boil only supported for image"
        self.linear_transform = linear_transform
        self.max_pool_before_transform = max_pool_before_transform
        self.old = old
        self.discrete_ops = discrete_ops
        self.warm_start = warm_start
        self.use_grad_mask = use_grad_mask
        self.swap_base_trans = swap_base_trans
        assert not (self.swap_base_trans and tnet), "tnet is incompatible with swap_base_trans"
        self.training = True
        self.meta_lr = 1
        self.meta_lr_architecture = meta_lr_architecture
        self.alfa_lr = alfa_lr
        self.arch_steps = arch_steps

            
        self.trans_before_relu = trans_before_relu
        assert not (discrete_ops and not self.svd), "Discrete ops only supported when --svd is true"
        assert not (discrete_ops and not self.old), "Discrete ops only supported when --old is true"
        assert not (trans_before_relu and not self.svd), "trans before relu only implemented when --svd is true"
        assert not (trans_before_relu and not self.old), "trans before relu only implemented when --old is true"
        self.tnet = tnet

        self.train_iters = train_iters
        if not self.train_iters is None:
            self.meta_iters = self.train_iters / self.meta_batch_size
        print("Number of train_iters:", self.train_iters)
        self.temperature = 1
        self.anneal_temp = anneal_temp
        assert not (self.anneal_temp and not self.discrete_ops), "Temperature annealing only makes sense when discrete_ops is true"
        if self.anneal_temp:
            self.make_train_masks = soft_mask
        else:
            self.make_train_masks = gumbel_binary_mask

        assert not (self.soft and not self.discrete_ops), "--soft requires --discrete_ops"
        assert not (self.soft and not self.anneal_temp), "--soft requires --anneal_temp"
        if self.soft:
            self.make_test_masks = soft_mask
        else:
            self.make_test_masks = max_binary_mask

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        self.global_counter = 0
        
        # Maintain train loss history
        self.train_losses = []

        # Get random initialization point for baselearner
        self.baselearner_args["free_arch"] = self.free_arch
        self.baselearner_args["relu"] = relu
        self.baselearner_args["channel_scale"] = self.channel_scale
        self.baselearner_args["use_grad_mask"] = self.use_grad_mask
        if self.image and self.svd:
            self.baselearner_args["max_pool_before_transform"] = max_pool_before_transform
            self.baselearner_args["linear_transform"] = linear_transform
            self.baselearner_args["old"] = self.old
            self.baselearner_args["discrete_ops"] = self.discrete_ops
            self.baselearner_args["trans_before_relu"] = self.trans_before_relu
            
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        if self.tnet:
            assert not (self.learn_alfas and self.tnet), "can't learn alfas when tnet"
            assert not (self.svd and self.tnet), "svd doesnt work with tnet"

        self.base_params = [p.clone().detach().to(self.dev) for p in self.baselearner.model.parameters()]
        self.transform_params = [p.clone().detach().to(self.dev) for p in self.baselearner.transform.parameters()]
        self.alfas = [p.clone().detach().to(self.dev) for p in self.baselearner.alfas]
        self.num_real_alfas = len(self.alfas)
        # if we use discrete ops, append the distributions to the alfas as they serve similar purposes
        if self.discrete_ops:
            self.alfas += [p.clone().detach().to(self.dev) for p in self.baselearner.distributions]


        
        if self.use_grad_mask:
            self.grad_masks = [p.clone().detach().to(self.dev) for p in self.baselearner.grad_masks]
            for p in self.grad_masks:
                p.requires_grad=True


        



        self.alfa_mask = set() # will contain indices that will be frozen (activated/de-activated)
        self.counts = np.zeros((len(self.alfas)))
        if not self.free_net and not self.svd:
            self.alfa_history = [ np.array([x.item() for x in self.alfas]) ]
        else:
            asl = []
            for x in self.alfas:
                asl.append([j.item() for j in x])

            self.alfa_history = [ np.array(asl) ]
        
        if self.discrete_ops:
            distribution_history = []
            for x in self.baselearner.distributions:
                distribution_history.append([j.item() for j in x])
            self.distribution_history = [ np.array(distribution_history) ]


        if exp1:
            assert not self.learn_alfas, "alfas were not learnable in experiment 1"
            assert not self.free_arch, "exp1 is incompatible with free_arch"
            self.alfas[0] = torch.Tensor([999]).squeeze()
            self.alfas[1] = torch.Tensor([-999]).squeeze()
            self.alfas[2] = torch.Tensor([-999]).squeeze()
            self.alfas[3] = torch.Tensor([999]).squeeze()
            print("Preset activations to", [torch.sigmoid(x).item() for x in self.alfas])
            self.alfas = [p.clone().detach().to(self.dev) for p in self.alfas]
        

        for b in [self.base_params, self.transform_params, self.alfas]:
            # Enable gradient tracking for the initialization parameters
            for p in b:
                p.requires_grad = True


        self.regmap = {
            "num_params": regularizer_np,
            "l2": regularizer_l2,
            "null": regularizer_null,
            "l1": regularizer_l1,
            "entropy": alfa_regularizer_entropy,
            "we": weight_entropy_regularizer,
        }
        self.regularizer = self.regmap[reg]

        net_params = self.base_params + self.transform_params
        self.optimizer = torch.optim.Adam([
            {'params': net_params, 'lr': self.base_lr},
            {'params': self.alfas, 'lr':self.alfa_lr}
        ], betas=(0,0.999))
        self.opt_dict = self.optimizer.state_dict()



        if not self.free_arch and not self.free_net and not self.svd and not self.tnet:
            if not self.image:
                pls = np.array([p.numel() for p in self.baselearner.transform.parameters()] + [0])
                self.num_params = torch.Tensor(pls.reshape(len(pls)//2,2).sum(axis=1))
            else:
                pls = np.array([p.numel() for p in self.baselearner.transform.parameters()])
                self.num_params = torch.Tensor(pls.reshape(len(pls)//2,2).sum(axis=1))
        else:
            pls = np.array([p.numel() for p in self.baselearner.transform.parameters()])
            self.num_params = torch.Tensor(pls)


        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]  
        
        if pretrain and not force_nopretrain:
            self.pretrain()

        self.current_meta_iter = 0


    def pretrain(self):
        x = np.random.uniform(-5.0-1, 5.0+1, 1028).reshape(-1, 1).astype('float32')
        ampl1, phase1 = 1, 0


        best_loss = 999
        best_init_weights = [p.clone().detach() for p in self.base_params]

        def fn(x, ampl, phase):
            return ampl*np.sin(x + phase)
        import matplotlib.pyplot as plt

        y = fn(x, ampl=ampl1, phase=phase1).reshape(-1, 1).astype('float32')
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        net = self.sine_constr(**self.baselearner_args).to(self.dev)
        optim = self.opt_fn(self.base_params, lr=self.lr)

        for t in range(10000):
            indices = np.random.permutation(len(x))[:128]

            preds = net.forward_weights(x[indices], self.base_params)
            loss = net.criterion(preds, y[indices])
            loss.backward()
            optim.step()
            optim.zero_grad()

            if t % 500 == 0:
                print(f"Loss: {loss.item():.3f}")
                x_plot, y_plot, pred_plot = x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), net.forward_weights(x, self.base_params).detach().numpy().reshape(-1)
                plt.figure()
                plt.scatter(x_plot, y_plot, color='blue', label="ground-truth")
                plt.scatter(x_plot, pred_plot, color='red', label='pred')
                plt.savefig(f"plt{t}.png")
                plt.close()

                with torch.no_grad():
                    full_preds = net.forward_weights(x, self.base_params)
                    full_loss = net.criterion(full_preds, y).item()
                    if full_loss < best_loss:
                        best_loss = full_loss
                        best_init_weights = [p.clone().detach() for p in self.base_params]

        self.base_params = [p.clone().detach() for p in best_init_weights]
        for p in self.base_params:
            p.requires_grad = True

        
        adjustable_params = []
        # Initialize the meta-optimizer
        if not self.freeze_init:
            if self.learn_alfas:
                adjustable_params += (self.base_params + self.alfas)
                print("Alfas are learnable")
            else:
                adjustable_params += self.base_params
        if not self.freeze_transform:
            adjustable_params += self.transform_params
        if len(adjustable_params) > 0:
            self.optimizer = self.opt_fn(adjustable_params, lr=self.lr)


    def _forward(self, x):
        return self.baselearner.transform_forward(x, bweights=self.base_params, 
                                                  weights=self.transform_params, alfas=self.alfas)

    def _mini_batches(self, train_x, train_y, batch_size, num_batches, replacement):
        """
        Generate mini-batches from some data.
        Returns:
        An iterable of sequences of (input, label) pairs,
            where each sequence is a mini-batch.
        """
        if replacement:
            for _ in range(num_batches):
                print("train_x size[0]:", train_x.size()[0])
                ind = np.random.randint(0, train_x.size()[0], batch_size)
                yield train_x[ind], train_y[ind]
            return

        if batch_size >= len(train_x):
            for i in range(num_batches):
                yield train_x, train_y
            return

        batch_count = 0
        while True:
            ind = np.arange(len(train_x))
            np.random.shuffle(ind)
            lb = 0 # lowerbound
            #print("len:", len(train_x), "batch_num:", num_batches)
            while lb + batch_size <= len(train_x) and batch_count != num_batches:
                batch_count += 1
                lb += batch_size
                yield train_x[ind[lb-batch_size:lb]], train_y[ind[lb-batch_size:lb]] #We may exceed the upper bound by 1 but then we simply have a batch of size batch_size - 1
                
            if batch_count == num_batches:
                return


    def _fast_weights(self, params, gradients, train_mode=False, freeze=False, masks=None):
        """Compute task-specific weights using the gradients
        
        Apply a single step of gradient descent using the provided gradients
        to compute task-specific, or equivalently, fast, weights.
        
        Parameters
        ----------
        params : list
            List of parameter tensors
        gradients : list
            List of torch.Tensor variables containing the gradients per layer
        """
        lr = self.base_lr if not train_mode else self.train_base_lr


        # Clip gradient values between (-10, +10)
        if not self.grad_clip is None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients]
        #print(len(masks))
        if masks is None:
            if not self.free_arch and not self.image and not self.free_net:
                fast_weights = [params[0]] + [(params[i] - lr * gradients[i]) if (not freeze or i >= len(gradients) - 2) else params[i]\
                            for i in range(1, len(gradients))] # not start at 1 in regular case
            else:
                fast_weights = [(params[i] - lr * gradients[i]) if (not freeze or i >= len(gradients) - 2) else params[i]\
                                for i in range(len(gradients))] # not start at 1 in regular case
        else:
            fast_weights = []
            param_types = self.baselearner.param_types
            paramid_to_layerid = self.baselearner.pid_to_lid
            # print(len(param_types), len(params))

            # for ptype, lid in zip(param_types, paramid_to_layerid):
            #     print(ptype, lid)
            # print()
            # import sys; sys.exit()
            # print(param_types) 
            for pid, (param, param_type, grad) in enumerate(zip(params, param_types, gradients)):
                #print(pid, paramid_to_layerid[pid], paramid_to_layerid[pid] is None)
                # if paramid_to_layerid[pid] is None:
                #     fast_weights += [param - lr * grad]
                #     continue
                #print(pid, param_type, param.size(), masks[paramid_to_layerid[pid]].size())
                mask = masks[paramid_to_layerid[pid]] # dimension of the current layer
                if param_type == ParamType.Scalar or param_type == ParamType.SVD_U or param_type == ParamType.SVD_S:
                    # Sum the values in the mask. If sum != len(mask), at least one feature is masked so all gradients should be stopped 
                    # otherwise, all features are active, and gradients are allowed to flow
                    summ = torch.sum(mask)
                    # just use summ (which is zero)
                    if summ.item() != len(mask):
                        if summ.item() == 0:
                            mv = summ 
                        else:
                            mv = 1 - summ/(summ.item()) # at least one feature inactive (0) -> zero
                    else:
                        mv = summ/(summ.item()) # all features active -> use 1 
                    #print("paramtype: scalar/svd_u/svd_s --- mask:", mv)
                    fast_weights += [param - lr * grad * mv]
                elif param_type == ParamType.Vector or (param_type == ParamType.Matrix and not self.tnet) or param_type == ParamType.SimpleScale:
                    fast_weights += [param - lr * grad * mask] # element-wise product ~ masks columns of matrix
                    #print("paramtype: vector/matrix --- masked grads:", mask*grad, mask)
                elif param_type == ParamType.SVD_V or (param_type == ParamType.Matrix and self.tnet):
                    # Since V.T is matrix-matrix multiplied with the input, we freeze columns of V.T -> rows of V
                    fast_weights += [param - lr * grad * mask.reshape(-1,1)]
                    #print("paramtype: svd_v --- masked grads:", grad * mask.reshape(-1,1)*grad, mask)
                elif param_type == ParamType.ConvTensor or param_type == ParamType.MTLScale or param_type == ParamType.ConvSVD_U or param_type==ParamType.ConvSVD_V:
                    fast_weights += [param - lr * grad * mask.reshape(-1,1,1,1)]
                    # 3,3,1,1 = (out, in, k, k) -> mask of size [3]
                elif param_type == ParamType.ConvSVD_S:
                    fast_weights += [param -lr * grad * mask.view(-1,1,1)]
                else:
                    print("paramtype is not caught by the net of if-statements")
                    import sys; sys.exit()
        return fast_weights


    def _hessian_vector_product(self, input, target, r=1e-2):
        flat_grads = torch.cat([p.grad.view(-1) for p in self.base_params] + [p.grad.view(-1) for p in self.transform_params])
        # r: the small constant in which we move the params
        # vector: gradients w.r.t. base-learner params
        R = r / flat_grads.norm()
        
        
        # add in gradient direction
        # concat flattens all params
        for p in self.base_params+self.transform_params:
            p.data.add_(R, p.grad) # adds R *v to each parameter 
        preds = self.baselearner.transform_forward(input, bweights=self.base_params, 
                                    weights=self.transform_params, alfas=self.alfas[:self.num_real_alfas], binary_masks=None)
        loss = self.baselearner.criterion(preds, target) # compute new loss
        grads_p = torch.autograd.grad(loss, self.alfas[:self.num_real_alfas]) # loss w.r.t. alphas

        # subtract in gradient direction
        for p in self.base_params+self.transform_params:
            p.data.sub_(2*R, p.grad) # adds R *v to each parameter 
        preds = self.baselearner.transform_forward(input, bweights=self.base_params, 
                                    weights=self.transform_params, alfas=self.alfas[:self.num_real_alfas], binary_masks=None)
        loss = self.baselearner.criterion(preds, target) # compute new loss
        grads_n = torch.autograd.grad(loss, self.alfas[:self.num_real_alfas])

        # restore original params
        for p in self.base_params+self.transform_params:
            p.data.add_(R, p.grad)
        
        for p, left, right in zip(self.alfas[:self.num_real_alfas], grads_p, grads_n):
            p.grad = (left-right)/(2*R)

    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, **kwargs):
        """Run DOSO on a single task to get the loss on the query set
        
        1. Evaluate the base-learner loss and gradients on the support set (train_x, train_y)
        using our initialization point.
        2. Make a single weight update based on this information.
        3. Evaluate and return the loss of the fast weights (initialization + proposed updates)
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        train_mode : boolean
            Whether we are in training mode or test mode

        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """
        
        # if not train mode and self.special, we need to use val_params instead of self.init
        with torch.no_grad():
            fast_transform_params = [p.clone().detach() for p in self.transform_params]
            fast_init_params = [p.clone().detach() for p in self.base_params]
            fast_alfas = [p.clone().detach() for p in self.alfas]
            for b in [fast_transform_params, fast_init_params, fast_alfas]:
                for p in b:
                    p.requires_grad = True

        learner = self.baselearner

        loss_history = None
        if not train_mode: loss_history = []

        binary_masks = None
        # in train mode + discrete ops, we have to sample gumbel masks
        if train_mode and self.discrete_ops:
            binary_masks = self.make_train_masks(self.alfas[self.num_real_alfas:], temperature=self.temperature)
        if not train_mode and self.discrete_ops:
            binary_masks = self.make_test_masks(self.alfas[self.num_real_alfas:], device=self.dev, temperature=self.temperature)
        
        ls = [{'params': fast_transform_params+fast_init_params, 'lr': self.base_lr}]
        if train_mode:
            ls.append({'params': fast_alfas, 'lr':self.alfa_lr})

        optimizer = torch.optim.Adam(ls, betas=(0, 0.999))
        if train_mode:
            optimizer.load_state_dict(self.opt_dict)
            batch_size = self.train_batch_size
            data_x, data_y = torch.cat([train_x, test_x]), torch.cat([train_y, test_y])
        else:
            batch_size = len(train_x) if len(train_x) <= 5 else 15
            data_x, data_y = train_x, train_y



        batches = self._mini_batches(data_x, data_y, batch_size=batch_size, num_batches=T, replacement=False)
        for t, (batch_x, batch_y) in enumerate(batches):
            preds = self.baselearner.transform_forward(batch_x, bweights=fast_init_params, 
                                    weights=fast_transform_params, alfas=fast_alfas[:self.num_real_alfas], binary_masks=binary_masks)
            loss = self.baselearner.criterion(preds, batch_y)
            loss.backward()
            # Mask gradients of architecture if current train iter < warm_up iters
            # or if t > self.arch_steps
            if (train_mode and self.current_meta_iter < self.warm_start) or t > self.arch_steps:
                for p in fast_alfas:
                    p.grad=None
            optimizer.step()
            optimizer.zero_grad()
             
        if train_mode:
            self.opt_dict = optimizer.state_dict()
            ls_left, ls_right = [self.transform_params, self.base_params], [fast_transform_params, fast_init_params]
            if not self.second_order:
                ls_left.append(self.alfas)
                ls_right.append(fast_alfas)

            for orig, new in zip(ls_left, ls_right):
                for pold, new in zip(orig, new):
                    if pold.grad is None:
                        pold.grad = (new.detach() - pold.detach())
                    else:
                        pold.grad += (new.detach() - pold.detach())
            return None, None, loss_history, None

        # Get and return performance on query set
        test_preds = learner.transform_forward(test_x, bweights=fast_init_params, 
                                              weights=fast_transform_params, alfas=fast_alfas[:self.num_real_alfas], binary_masks=binary_masks)
        test_loss = learner.criterion(test_preds, test_y)
        return test_loss, test_preds, loss_history, None
    
    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        Start with the common initialization point and perform a few
        steps of gradient descent from there using the support set
        (rain_x, train_y). Observe the error on the query set and 
        propagate the loss backwards to update the initialization.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """ 
        
        # Put baselearner in training mode
        self.baselearner.train()
        self.task_counter += 1
        self.global_counter += 1
        self.training = True

        # Compute the test loss after a single gradient update on the support set
        if not self.freeze_init or not self.freeze_transform:
            # Put all tensors on right device
            train_x, train_y, test_x, test_y = put_on_device(
                                                self.dev,
                                                [train_x, train_y,
                                                test_x, test_y])
        
        
            test_loss, preds,_,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)

            # Clip gradients
            if not self.grad_clip is None:
                for b in [self.base_params, self.transform_params, self.alfas]:
                    # Enable gradient tracking for the initialization parameters
                    for p in b:
                        p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)
                        if p.grad is None:
                            continue

            if self.task_counter % self.meta_batch_size == 0: 
                # for i, p in enumerate(self.transform_params):
                #     if i < 2 or i>= len(self.transform_params)-1:
                #         print(p.grad)
                if not self.free_arch and not self.image and not self.free_net and not self.svd and not self.tnet:
                    self.transform_params[0].grad = None # zero out grad
                    

                if self.discrete_ops:
                    for b in [self.base_params, self.transform_params, self.alfas]:
                        for p in b:
                            if not torch.all(~torch.isnan(p.grad)):
                                p.grad = None

                if self.second_order:
                    data_x, data_y = torch.cat([train_x, test_x]), torch.cat([train_y, test_y])
                    self._hessian_vector_product(data_x, data_y)


                for b in [self.base_params, self.transform_params]:
                    for p in b:
                        p.data = (p.data + self.meta_lr * p.grad/self.meta_batch_size).detach()
                        p.requires_grad = True
                        p.grad = None

                
                for p in self.alfas:
                    p.data = (p.data + self.meta_lr_architecture * p.grad/self.meta_batch_size).detach()
                    p.requires_grad = True
                    p.grad = None
                # print(self.alfas)

                if self.use_grad_mask:
                    for pid, p in enumerate(self.grad_masks):
                        # print(pid, p.grad)
                        if torch.isnan(p.grad).any():
                            print("Found NAN gradients") 
                            import sys; sys.exit()
                    # import sys;sys.exit(0)
                # for p in self.alfas[self.num_real_alfas:]:
                #     print(p, p.grad)

                #self.optimizer.step()
                self.task_counter = 0
                self.current_meta_iter += 1

                if self.anneal_temp:
                    self.temperature -= 1/self.meta_iters
                self.meta_lr -= 1/self.meta_iters
                self.meta_lr_architecture -= 0.6/self.meta_iters 

                if not self.free_net and not self.svd:
                    self.alfa_history.append( np.array([x.item() for x in self.alfas]) )
                else:
                    asl = []
                    for x in self.alfas:
                        asl.append([j.item() for j in x])

                    self.alfa_history.append(np.array(asl))

                    if self.discrete_ops:
                        distribution_history = []
                        for x in self.alfas[self.num_real_alfas:]:
                            distribution_history.append([j.item() for j in x])
                        self.distribution_history.append( np.array(distribution_history) )


    def evaluate(self, train_x, train_y, test_x, test_y, val=True, compute_cka=False, return_preds=False):
        """Evaluate on a given task
        
        Use the support set (train_x, train_y) and/or 
        the query set (test_x, test_y) to evaluate the performance of 
        the model.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """
        loss_history = []
        # Put baselearner in evaluation mode
        self.baselearner.eval()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        if val:
            T = self.T_val
        else:
            T = self.T_test
        

        if self.training:
            if self.use_grad_mask:
                for p in self.grad_masks:
                    print(p)
            self.training = False
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, loss_history, raw_loss = self._deploy(train_x, train_y, test_x, test_y, False, T)


        if self.operator == min:
            if return_preds:
                return raw_loss, loss_history, preds.detach()
            return raw_loss, loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            test_acc = accuracy(preds, test_y)
            if return_preds:
                return test_acc, loss_history, preds.detach()
            return test_acc, loss_history
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """

        if not self.use_grad_mask:
            return [p.clone().detach() for p in self.base_params],\
                [p.clone().detach() for p in self.transform_params],\
                [p.clone().detach() for p in self.alfas]
        else:
            return [p.clone().detach() for p in self.base_params],\
                [p.clone().detach() for p in self.transform_params],\
                [p.clone().detach() for p in self.alfas],\
                [p.clone().detach() for p in self.grad_masks]

    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        self.base_params = state[0]
        self.transform_params = state[1]
        self.alfas = state[2]

        ls = ["base_params", "transform_params", "alfas"]
        if self.use_grad_mask:
            assert len(state) == 4, "Used old model save - gradmasks are not stored"
            self.grad_masks = state[3]
            ls.append("grad_masks")
        
        for s in ls:
            for p in eval(f"self.{s}"):
                p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.base_params = [p.to(device) for p in self.base_params]
        self.transform_params = [p.to(device) for p in self.transform_params]
        self.alfas = [p.to(device) for p in alfas]