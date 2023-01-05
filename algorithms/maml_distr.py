from enum import unique
import torch
import numpy as np
import os
import psutil
import GPUtil as GPU

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task, get_info
from .modules.similarity import gram_linear, cka


class DISTR(Algorithm):
    """Model-Agnostic Meta-Learning
    
    Meta-learning algorithm that attempts to obtain a good common 
    initialization point (base-learner parameters) across tasks.
    From this initialization point, we want to be able to make quick
    task-specific updates to achieve good performance from just few
    data points.
    Our implementation performs a single step of gradient descent
    
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
    
    def __init__(self, train_base_lr, base_lr, second_order, grad_clip=None, meta_batch_size=1, **kwargs):
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
        print("Using DISTR")
        self.sine = False
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.log_test_norm = False
        self.measure_distances = False
        self.log_interval = 80
        self.num_samples = 50

        self.supp_dists = []
        self.query_dists = []
        

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        self.global_counter = 0

        self.gpu_usage = []
        self.cpu_usage = []
        self.using_gpu = ":" in self.dev
        if self.using_gpu:
            gpu_id = int(self.dev.split(":")[-1]) 
            self.gpu = GPU.getGPUs()[gpu_id]
        
        # Maintain train loss history
        self.train_losses = []
        self.train_scores = []
        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        # Create external memory module
        self.feature_size =  self.baselearner.get_infeatures(torch.rand((1,3,84,84), device=self.baselearner.dev)).size(1)
        self.memory_embedding_mean = torch.zeros(64, self.feature_size, device=self.baselearner.dev, requires_grad=True)
        self.memory_embedding_meansq = torch.zeros(64, self.feature_size, device=self.baselearner.dev, requires_grad=True)
        self.memory_counter = torch.zeros(64, device=self.baselearner.dev)

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True
                
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(self.initialization + [self.memory_embedding_mean] + [self.memory_embedding_meansq], lr=self.lr)

        self.test_losses = []
        self.test_norms = []  
        self.test_perfs = []    
        self.angles = []
        self.distances = []
        self.gangles = []
        self.gdistances = []      

    def _forward(self, x):
        return self.baselearner.forward_weights(x, self.initialization)

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
        lr = self.base_lr if not train_mode else self.train_base_lr

        # Clip gradient values between (-10, +10)
        if not self.grad_clip is None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients]
        
        fast_weights = [(params[i] - lr * gradients[i]) if ((not freeze or i >= len(gradients) - 2) and not gradients[i] is None) else params[i]\
                        for i in range(len(gradients))]
        
        return fast_weights

    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, global_train_y=None, global_test_y=None, **kwargs):
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
        

        fast_weights = [p.clone() for p in self.initialization]   
        learner = self.baselearner
        loss_history = None
        if not train_mode: loss_history = []

        self.fast_memory = self.memory_embedding_mean.clone()
        self.fast_memory_meansq = self.memory_embedding_meansq.clone()

        #with torch.no_grad():
        embeddings = self.baselearner.forward_weights(train_x, weights=fast_weights, flat=True) #[batch_size, hidden_size]

        mean_supp_emb = []
        counts = []
        unique_labels = torch.unique(train_y)
        
        for cix, class_idx in enumerate(unique_labels):
            mask = train_y == class_idx
            mean_emb = embeddings[mask].sum(dim=0)
            meansq_emb = (embeddings[mask]**2).sum(dim=0)
            mean_supp_emb.append(mean_emb)
            counts.append(mask.sum())

            if train_mode:
                unique_global_labels = torch.unique(global_train_y)
                # work on .data directly to prevent autograd issues (in-place op. of view)
                self.memory_counter[unique_global_labels[cix]] += mask.sum()
                self.fast_memory[unique_global_labels[cix]] = self.memory_embedding_mean[unique_global_labels[cix]] + mean_emb
                self.fast_memory_meansq[unique_global_labels[cix]] = self.memory_embedding_meansq[unique_global_labels[cix]] + meansq_emb
        
        mean_supp_emb = torch.stack(mean_supp_emb)
        counts = torch.Tensor(counts).to(self.baselearner.dev)
        #print("mean supp emb size:", mean_supp_emb.size(), "counts size:", counts.size())
        #print("div:", mean_supp_emb/counts.unsqueeze(1))
        #import sys; sys.exit()
        
        
        # all non-zero rows of self.memory
        nz_mask = (self.memory_counter != 0)

        dists = -torch.cdist(mean_supp_emb/counts.unsqueeze(1), self.fast_memory[nz_mask]/self.memory_counter[nz_mask].unsqueeze(1), p=2) #[N, number of seen classes] 
        sims = dists.softmax(dim=1)
        augmented_supp_means = (mean_supp_emb + torch.matmul(sims,self.fast_memory[nz_mask]))/(counts.unsqueeze(1) + torch.matmul(sims, self.memory_counter[nz_mask].unsqueeze(1)))

        query_embeddings = self.baselearner.forward_weights(test_x, weights=fast_weights, flat=True) #[batch_size, hidden_size]
        qdists = -torch.cdist(query_embeddings, self.fast_memory[nz_mask]/self.memory_counter[nz_mask].unsqueeze(1), p=2)
        qsims = qdists.softmax(dim=1)
        augmented_query_means = (query_embeddings + torch.matmul(qsims,self.fast_memory[nz_mask]))/(1 + torch.matmul(qsims, self.memory_counter[nz_mask].unsqueeze(1)))
        test_preds = -torch.cdist(augmented_query_means, augmented_supp_means)
        test_loss = learner.criterion(test_preds, test_y)


        if train_mode and T > 0:
            self.train_losses.append(0)

        if not train_mode: loss_history.append(test_loss.item())
        
        if train_mode:
            unique_labels = torch.unique(test_y)
            for cix, class_idx in enumerate(unique_labels):
                mask = test_y == class_idx
                mean_emb = query_embeddings[mask].sum(dim=0)
                meansq_emb = (query_embeddings[mask]**2).sum(dim=0)

                if train_mode:
                    unique_global_labels = torch.unique(global_test_y)
                    # work on .data directly to prevent autograd issues (in-place op. of view)
                    self.memory_counter[unique_global_labels[cix]] += mask.sum()
                    self.fast_memory[unique_global_labels[cix]] = self.fast_memory[unique_global_labels[cix]] + mean_emb
                    self.fast_memory_meansq[unique_global_labels[cix]] = self.fast_memory_meansq[unique_global_labels[cix]] + meansq_emb
        
        
        
        
        
        # for every class, compute mean and meansq
        # unique_labels = torch.unique(test_y)
        # generated_samples = []
        # generated_classes = []
        # for class_idx in unique_labels:
        #     mask = test_y == class_idx
        #     mean_emb = query_embeddings[mask].sum(dim=0).unsqueeze(dim=0)
        #     meansq_emb = (query_embeddings[mask]**2).sum(dim=0).unsqueeze(dim=0)

        #     # augment mean and meansq with memory module
        #     similarities = torch.cdist(mean_emb, self.memory_embedding_mean[nz_mask], p=2).softmax(dim=1) #[1, number of nonzero classes]
        #     augmented_mean_emb = (mean_emb + similarities*self.memory_embedding_mean[nz_mask])/(mask.sum() + (similarities*self.memory_counter).sum())
        #     augmented_meansq_emb = (meansq_emb + similarities*self.memory_embedding_meansq[nz_mask])/(mask.sum() + (similarities*self.memory_counter).sum())

        #     # Sample examples using the augmented mean and augmented meansq embeddings
        #     for _ in range(self.num_samples):
        #         generated_samples.append(augmented_mean_emb + (augmented_meansq_emb**0.5) * torch.rand_like(augmented_mean_emb))
        #         generated_classes.append(class_idx)

        # compute soft-attention between current query examples and




        #print(self.memory_embedding_mean[unique_labels].requires_grad, self.memory_embedding_mean[unique_labels],)
        #import sys; sys.exit()





        # need to retrieve global class indices



        # for step in range(T):     
        #     # if self.special and not train mode, use val_learner instead
        #     loss, grads = get_loss_and_grads(learner, xinp, yinp, 
        #                                 weights=fast_weights, 
        #                                 create_graph=self.second_order and train_mode,
        #                                 retain_graph=T > 1 or self.second_order,
        #                                 flat=False)
        #     fast_weights = self._fast_weights(params=fast_weights, gradients=grads, train_mode=train_mode)
            
        #     if not train_mode: 
        #         loss_history.append(loss)

            

        # xinp, yinp = test_x, test_y
        # # Get and return performance on query set
        # test_preds = learner.forward_weights(xinp, fast_weights)
        # test_loss = learner.criterion(test_preds, yinp)


        # if train_mode and T > 0:
        #     self.train_losses.append(loss)

        # if not train_mode: loss_history.append(test_loss.item())
        return test_loss, test_preds, loss_history
    
    def train(self, train_x, train_y, test_x, test_y, global_train_y, global_test_y):
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

        # Put all tensors on right device
        train_x, train_y, test_x, test_y, global_train_y, global_test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y,
                                            global_train_y,
                                            global_test_y])
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, _,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T, 
                                      global_train_y=global_train_y, global_test_y=global_test_y)

        # Propagate the test loss backwards to update the initialization point
        test_loss.backward()
            
        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        if self.task_counter % self.meta_batch_size == 0: 
            if self.global_counter % self.log_interval == 0 and self.using_gpu:
                self.gpu_usage.append(self.gpu.memoryUsed)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.task_counter = 0
            with torch.no_grad():
                self.memory_embedding_mean.data = self.fast_memory.data
                self.memory_embedding_meansq.data = self.fast_memory_meansq.data

    def evaluate(self, train_x, train_y, test_x, test_y, val=True):
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

        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T,)

        if self.operator == min:
            return test_loss.item(), loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            test_acc = accuracy(preds, test_y)
            return test_acc, loss_history
    
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