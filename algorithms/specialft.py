import torch
import numpy as np

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task
from .modules.similarity import gram_linear, cka

class InverseSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults={"lr": lr}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        snorm = None
        for gid, group in enumerate(self.param_groups):
            for p in group['params']:
                if p.grad is None:
                    contrinue
                if snorm is None:
                    snorm = torch.sum(p.grad**2)
                else:
                    snorm = snorm + torch.sum(p.grad**2)
            
            norm = torch.sqrt(snorm)
            # print("total norm:", norm)
            for p in group['params']:
                if p.grad is None:
                    contrinue
                
                print("used LR:", group['lr']/norm)
                p = p - (group['lr']/norm) * p.grad 
        return loss

class SpecialFT(Algorithm):
    
    def __init__(self, base_lr, grad_clip=None, meta_batch_size=1, **kwargs):
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
        self.base_lr = base_lr
        self.grad_clip = grad_clip
        self.meta_batch_size = meta_batch_size

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        
        # Maintain train loss history
        self.train_losses = []

        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.vallearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.vallearner.freeze_layers(False)
        self.vallearner.eval()
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True
        
        # Initialize the meta-optimizer
        #self.optimizer = InverseSGD(self.baselearner.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.baselearner.parameters(), lr=self.lr)
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]
            

    def _get_params(self):
        return [p.clone().detach() for p in self.initialization]


    def _fast_weights(self, params, gradients, train_mode=False, freeze=False):
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
        lr = self.base_lr

        # Clip gradient values between (-10, +10)
        if not self.grad_clip is None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients]
        
        if freeze:
            mnt_rng = params[:-2]
            rng = list(range(len(gradients)))[-2:]
        else:
            rng = range(len(gradients))
            mnt_rng = []

        fast_weights = mnt_rng + [params[i] - lr * gradients[i] for i in rng]
        
        return fast_weights
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, compute_cka=False):
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
        
        
        
        loss_history = None
        if not train_mode: loss_history = []

        if not train_mode:
            # if not train mode and self.special, we need to use val_params instead of self.init
            sd = self.baselearner.state_dict()
            learner = self.vallearner
            learner.load_params(sd) 
            learner.freeze_layers(False)

            fast_weights = [p.clone() for p in learner.parameters()]

            for step in range(T):     
                xinp, yinp = train_x, train_y

                # if self.special and not train mode, use val_learner instead
                loss, grads = get_loss_and_grads(learner, xinp, yinp, 
                                            weights=fast_weights, 
                                            create_graph=False,
                                            retain_graph=False,
                                            flat=False)

                loss_history.append(loss)

                fast_weights = self._fast_weights(params=fast_weights, gradients=grads, train_mode=train_mode, freeze=True)

            if compute_cka:
                return fast_weights

            if train_mode and T > 0:
                self.train_losses.append(loss)
        
            xinp, yinp = test_x, test_y
            # Get and return performance on query set
            test_preds = learner.forward_weights(xinp, fast_weights)
            test_loss = learner.criterion(test_preds, yinp)
            loss_history.append(test_loss.item())
        else:
            xinp, yinp = test_x, test_y
            test_preds = self.baselearner(xinp)
            test_loss = self.baselearner.criterion(test_preds, yinp)
        return test_loss, test_preds, loss_history
    
    def setmap(self, class_map):
        self.class_map = class_map
        print("Set up the class map")

    def create_tensors(self, labels):
        converted = [self.class_map[label[0]] for label in labels]
        ts = torch.Tensor(converted).long()
        return ts

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

        # print(test_y)
        # print(type(train_x), type(train_y), type(test_x), type(test_y))
        # print(train_x.size(), train_y.size(), test_x.size(), test_y.size())
        train_y, test_y = self.create_tensors(train_y), self.create_tensors(test_y)
        
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, _,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)

        # Propagate the test loss backwards to update the initialization point
        test_loss.backward()
            
        # Clip gradients
        # if not self.grad_clip is None:
        #     for p in self.baselearner.parameters():
        #         p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)


        if self.task_counter % self.meta_batch_size == 0: 
            norm_type = 2 # Euclidean norm 
            parameters = [p for p in self.baselearner.parameters() if p.grad is not None]
            # rnorm = square root of the Euclidean norm of the gradients 
            div = torch.sqrt(torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)).item()
            for p in self.baselearner.parameters():
                if not p.grad is None:
                    p.grad = p.grad / div
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.task_counter = 0
            
            # snorm = None
            # with torch.no_grad():           
            #     for p in self.baselearner.parameters():
            #         if p.grad is None:
            #             contrinue
            #         if snorm is None:
            #             snorm = torch.sum(p.grad**2)
            #         else:
            #             snorm = snorm + torch.sum(p.grad**2)
                    
            #     norm = torch.sqrt(torch.sqrt(snorm)) # double square root to get the sqrt-norm
            #     lr = self.lr / norm
            #     for group in self.optimizer.param_groups:
            #         group["lr"] = lr 

    def evaluate(self, train_x, train_y, test_x, test_y, val=True, compute_cka=False):
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

        if compute_cka:
            _, initial_features = self.baselearner.forward_weights_get_features(torch.cat((train_x, test_x)), weights=self.initialization)
            fast_weights = self._deploy(train_x, train_y, test_x, test_y, False, T, compute_cka=True)
            _, final_features = self.baselearner.forward_weights_get_features(torch.cat((train_x, test_x)), weights=fast_weights)
            ckas = []
            dists = []
            for features_x, features_y in zip(initial_features, final_features):
                ckas.append( cka(gram_linear(features_x), gram_linear(features_y), debiased=True) )
                dists.append( np.mean(np.sqrt(np.sum((features_y - features_x)**2, axis=1))) )
            # Compute the test loss after a single gradient update on the support set
            test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)
            if self.sine:
                return test_loss.item(), ckas, dists
            
            preds = torch.argmax(preds, dim=1)
            return accuracy(preds, test_y), ckas, dists
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)

        if self.operator == min:
            return test_loss.item(), loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            return accuracy(preds, test_y), loss_history
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach() for p in self.initialization]
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.to(device) for p in self.initialization]