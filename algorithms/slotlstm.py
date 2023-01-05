import torch
import numpy as np
import os
import psutil
import GPUtil as GPU

from .algorithm import Algorithm
from .modules.utils import put_on_device, get_loss_and_grads,\
                           accuracy, deploy_on_task, get_info
from .modules.similarity import gram_linear, cka


class LossType:
    PostAdaptation = 0
    MultiStep = 1

class MetaSlotLSTM(Algorithm):
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
    
    def __init__(self, N, hidden_size, num_layers,  second_order,
                 avg_cell=False, loss_type="post", grad_clip=None, meta_batch_size=1, image=False, 
                 lstm_constructor=None, shuffle=False, learn_hidden=False, num_slots=5, **kwargs):
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
        self.image = image
        self.N = N
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.log_test_norm = False
        self.measure_distances = False
        self.loss_type = LossType.PostAdaptation if loss_type.lower().strip() == "post" else LossType.MultiStep
        self.avg_cell = avg_cell
        self.lstm_constructor = lstm_constructor
        self.shuffle = shuffle
        self.learn_hidden = learn_hidden
        self.num_slots = num_slots
        print("SHUFFLING:", self.shuffle)
        

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
        self.baselearner_args["num_layers"] = num_layers
        self.baselearner_args["hidden_size"] = hidden_size
        self.baselearner_args["learn_hidden"] = learn_hidden

        print("hidden size:", hidden_size, "num_layers:", num_layers)

        self.convnet = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.baselearner_args["input_size"] = self.convnet.in_features + self.convnet.eval_classes 
        self.baselearner_args["output_size"] = self.convnet.eval_classes 
        
        
        self.baselearner_args["slot_size"] = self.convnet.eval_classes 
        self.baselearner_args["num_slots"] = self.num_slots # change later
        self.baselearner_args["input_address_size"] = self.convnet.in_features
        self.baselearner_args["input_content_size"] = self.convnet.eval_classes

        print("NUM SLOTS:", self.num_slots)

        self.lstm = self.lstm_constructor(**self.baselearner_args).to(self.dev)
        
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(list(self.convnet.parameters()) + list(self.lstm.parameters()), lr=self.lr)
        
        self.log_interval = 80
        self.test_losses = []
        self.test_norms = []  
        self.test_perfs = []    
        self.angles = []
        self.distances = []
        self.gangles = []
        self.gdistances = []      


    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T):
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

        # embed the features
        

        train_x, test_x = self.convnet.get_infeatures(train_x), self.convnet.get_infeatures(test_x) # they now have size [batch, infeatures]
        
        # Tanh dot-product attention 
        #ind = torch.argsort(train_y)
        #preds = test_x @ torch.tanh(train_x[ind,:].T)
        # none labels for initial input
        # the one-hot labels for passes 2,...,T

        # COMMENTING BEGINS HERE
        train_onehot_labels = torch.nn.functional.one_hot(train_y).float()
        

        
        # go over the training data --- ingest data into the hidden state
        hn, cn = None, None

        if train_mode and self.loss_type == LossType.MultiStep:
            tr_loss = torch.zeros(1)

        for t in range(T):
            input_address = train_x # the train x
            input_content = train_onehot_labels
            state = None

            output, state  = self.lstm(input_address, input_content, state)
            # if train_mode and self.loss_type == LossType.MultiStep:
            #     init_zeros_supp = torch.zeros((train_x.size(0), self.convnet.eval_classes), device=self.dev)
            #     preds = preds = self.lstm.predict(h=output, x_train=torch.cat([train_x, train_onehot_labels], dim=1),
            #                       y_train=train_onehot_labels.float(), x_query=torch.cat([train_x, init_zeros_supp], dim=1)) 
            #     tr_loss = tr_loss + self.convnet.criterion(preds, train_y[t,:,:].unsqueeze(0))

            # average over hidden states for every layer and repeat along batch dimension
            hn = state[0].mean(dim=0).unsqueeze(0).repeat(input_address.size(0),1,1) # hn shape: [num_layers, batch size, hidden size]
            if self.avg_cell:
                cn = state[1].mean(dim=0).unsqueeze(1).repeat(input_address.size(0),1,1)                
            else:
                cn = state[1]
            state = (hn, cn)

        init_zeros = torch.zeros((test_x.size(0), self.convnet.eval_classes), device=self.dev)
        #test_input = torch.cat([test_x, init_zeros], dim=1).unsqueeze(0) # [1 (seq len), batch size, infeatures]

        # clone and adjust batch dimension to fit query examples
        fast_hn = hn[0,:,:].unsqueeze(0).repeat(test_x.size(0), 1,1) # [seq(1), batch size, infeatures] <- take the 0th batch index because they are all the same since we averaged
        
        if not self.avg_cell:
           fast_cn = None # reset cell state because they are specific to inputs (we have not seen query inputs before so start from scratch) -- 
        else:
           fast_cn = cn[0,:,:].unsqueeze(0).repeat(test_x.size(0), 1,1)  # [seq(1), batch size, infeatures]
    
        output, (fast_hn, fast_cn)  = self.lstm(test_x, init_zeros, (fast_hn, fast_cn))
        # compute predictions from output

        #print(torch.cat([train_x, train_onehot_labels], dim=1).shape, test_input.shape, output.shape)
        preds = self.lstm.predict(output) # test x and not 


        test_loss = self.convnet.criterion(preds, test_y)         
        if train_mode and self.loss_type == LossType.MultiStep:
            test_loss = test_loss + tr_loss

        if train_mode and T > 0:
            self.train_losses.append(test_loss.item())

        if not train_mode: loss_history.append(test_loss.item())
        return test_loss, preds, loss_history
    
            

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
        self.convnet.train()
        self.lstm.train()
        self.task_counter += 1
        self.global_counter += 1

        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])

        # Shuffle input data so the LSTM cannot find any cheating pattern
        if self.shuffle:
            train_indices, test_indices = torch.randperm(train_x.size(0)), torch.randperm(test_x.size(0))
            train_x, train_y, test_x, test_y = train_x[train_indices], train_y[train_indices], test_x[test_indices], test_y[test_indices]
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)
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
            

        preds = torch.argmax(preds, dim=1)
        test_acc = accuracy(preds, test_y)
        return test_loss.item(), test_acc

            
           

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
        self.convnet.train()
        self.lstm.eval()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])

        # Shuffle input data so the LSTM cannot find any cheating pattern
        if self.shuffle:
            train_indices, test_indices = torch.randperm(train_x.size(0)), torch.randperm(test_x.size(0))
            train_x, train_y, test_x, test_y = train_x[train_indices], train_y[train_indices], test_x[test_indices], test_y[test_indices]


        if val:
            T = self.T_val
        else:
            T = self.T_test
        

        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)

        if self.operator == min:
            return test_loss.item(), loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            test_acc = accuracy(preds, test_y)
            if self.log_test_norm:
                self.test_perfs.append(test_acc)
            return test_acc, loss_history
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach() for p in self.lstm.parameters()], [p.clone().detach() for p in self.convnet.parameters()], 
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        lstm_params, conv_params = state
        for p,v in zip(self.lstm.parameters(), lstm_params):
            p.data = v.data
            p.requires_grad = True

        for p,v in zip(self.convnet.parameters(), conv_params):
            p.data = v.data
            p.requires_grad = True

        
    def to(self, device):
        self.lstm = self.lstm.to(device)
        self.convnet = self.convnet.to(device)
