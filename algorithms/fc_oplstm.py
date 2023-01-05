from turtle import update
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

class FCOpLSTM(Algorithm):
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
                 lstm_constructor=None, shuffle=False, learn_hidden=False, avg_hidden=False, learn_init_weight=False,
                 lstm_inputs=None, param_lr=False, elwise=False, layers=None, sine=False, gamma=None, 
                 update_bias=False, no_batchnorm=False, **kwargs):
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
        self.sine = sine
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
        print("SHUFFLING:", self.shuffle)
        self.avg_hidden = avg_hidden
        self.learn_init_weight = learn_init_weight
        self.lstm_inputs = lstm_inputs
        self.param_lr = param_lr
        self.elwise = elwise
        self.use_layers = not layers is None
        self.update_bias = update_bias
        self.no_batchnorm = no_batchnorm
        
        self.gamma = gamma
        if not self.gamma is None:
            self.gamma = torch.nn.Parameter(torch.zeros(1, device=self.dev) + gamma) 

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

        self.convnet = self.baselearner_fn(**self.baselearner_args).to(self.dev)

        # Get random initialization point for baselearner
        self.baselearner_args["num_layers"] = num_layers
        self.baselearner_args["hidden_size"] = hidden_size
        self.baselearner_args["learn_hidden"] = learn_hidden
        self.baselearner_args["learn_init_weight"] = learn_init_weight
        self.baselearner_args["base_input_size"] = self.convnet.in_features
        self.baselearner_args["param_lr"] = param_lr
        self.baselearner_args["layers"] = layers
        self.baselearner_args["update_bias"] = update_bias

        if self.lstm_inputs is None or self.lstm_inputs == "target":
            if self.elwise:
                self.baselearner_args["input_size"] = 1 
                self.baselearner_args["hidden_size"] = 1 # (new_inputsize/old_inputsize) * old_inputsize 
                if self.use_layers:
                    layers[-1] = 1 #layers[-1] * 1 // self.convnet.eval_classes
            else:
                self.baselearner_args["input_size"] = self.convnet.eval_classes  #self.convnet.in_features + 
        elif self.lstm_inputs == "target_pred":
            if self.elwise:
                self.baselearner_args["input_size"] = 2 # pred and target
                self.baselearner_args["hidden_size"] = 1 # (new_inputsize/old_inputsize) * old_inputsize 
                if self.use_layers:
                    layers[-1] = 1 #layers[-1] * 2 // (2*self.convnet.eval_classes)
            else:
                self.baselearner_args["input_size"] = 2*self.convnet.eval_classes
            assert self.learn_init_weight, "Can only feed preds as input when we have initial W learned"

        self.baselearner_args["output_size"] = self.convnet.eval_classes 
        self.baselearner_args["elwise"] = self.elwise



        self.output_lstm = self.lstm_constructor(**self.baselearner_args).to(self.dev)
        self.body_lstm = self.lstm_constructor(**self.baselearner_args).to(self.dev)

        print("hidden size:", hidden_size, "num_layers:", num_layers)

        # Initialize the meta-optimizer
        if not self.gamma is None:
            params = list(self.convnet.parameters()) + list(self.output_lstm.parameters()) + list(self.body_lstm.parameters()) + [self.gamma]
            count_params = 1 + sum([p.numel() for p in self.convnet.parameters()]) + sum([p.numel() for p in self.output_lstm.parameters()]) + sum([p.numel() for p in self.body_lstm.parameters()])
        else:
            params = list(self.convnet.parameters()) + list(self.output_lstm.parameters()) + list(self.body_lstm.parameters())
            count_params = sum([p.numel() for p in self.convnet.parameters()]) + sum([p.numel() for p in self.output_lstm.parameters()]) + sum([p.numel() for p in self.body_lstm.parameters()])
        
        print("Number of parameters:", count_params)
        self.optimizer = self.opt_fn(params, lr=self.lr)
        
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

        # flatten the input images in case of image data
        train_x, test_x = train_x.reshape(train_x.size(0), -1), test_x.reshape(test_x.size(0), -1)
        # embed the features
        #train_x, test_x = self.convnet.get_infeatures(train_x), self.convnet.get_infeatures(test_x) # they now have size [batch, infeatures]
        
        # Tanh dot-product attention 
        #ind = torch.argsort(train_y)
        #preds = test_x @ torch.tanh(train_x[ind,:].T)
        # none labels for initial input
        # the one-hot labels for passes 2,...,T

        # COMMENTING BEGINS HERE
        if self.sine:
            train_onehot_labels = train_y
        else:
            train_onehot_labels = torch.nn.functional.one_hot(train_y, num_classes=self.convnet.eval_classes)

        if train_mode and self.loss_type == LossType.MultiStep:
            tr_loss = torch.zeros(1)

        # initial weights for weight matrices
        Hs = [h.clone() for h in self.convnet.get_fc_weight_params()]

        if self.update_bias:
            Bs = [b.clone() for b in self.convnet.get_fc_bias_params()]
        else:
            Bs = None

        hidden = [None for _ in Hs]
        cell = [None for _ in Hs]

        regularizer_cost = None
        
        for t in range(T):
            newHs = None
            newBs = None

            preds, stats = self.convnet.layer_wise_preds(train_x, Hs, Bs)

            for i in range(len(Hs)):
                idx = len(Hs)-(i+1)
                #print(f"i={i}, idx={idx}")
                hn, cn = hidden[idx], cell[idx]

                # if final layer
                if idx == len(Hs) - 1:
                    lstm = self.output_lstm
                    if self.lstm_inputs is None or self.lstm_inputs == "target":
                        lstm_input = train_onehot_labels.unsqueeze(0).float()
                    else:
                        lstm_input = torch.empty(1, train_onehot_labels.size(0), train_onehot_labels.size(1) + preds[idx].size(1), device=train_onehot_labels.device)
                        lstm_input[0, :, 0::2] = train_onehot_labels.float()
                        if self.second_order:
                            lstm_input[0, :, 1::2] = preds[idx]
                        else:
                            lstm_input[0, :, 1::2] = preds[idx].detach()
                else:
                    lstm = self.body_lstm
                    # print(preds[idx] is None)
                    # print(hidden[idx+1] is None)


                    
                    if not self.use_layers:
                        hinput = hidden[idx+1].reshape(preds[idx].size(0), -1) # batch size, hidden size
                        hinput = hinput @ H_old
                        lstm_input = torch.empty(1, preds[idx].size(0), preds[idx].size(1) + hinput.size(-1), device=train_onehot_labels.device)
                        if self.second_order:
                            lstm_input[0, :, 0::2] = preds[idx]
                            lstm_input[0, :, 1::2] = hinput
                        else:
                            lstm_input[0, :, 0::2] = preds[idx].detach()
                            lstm_input[0, :, 1::2] = hinput.detach()
                    else:
                        hinput = hidden[idx+1][-1].reshape(preds[idx].size(0), -1)
                        hinput = hinput @ H_old
                        lstm_input = torch.empty(1, preds[idx].size(0), preds[idx].size(1) + hinput.size(-1), device=train_onehot_labels.device)
                        #print(preds[idx].size(), hinput.size(), lstm_input.size())
                        if self.second_order:
                            lstm_input[0, :, 0::2] = preds[idx]
                            lstm_input[0, :, 1::2] = hinput
                        else:
                            lstm_input[0, :, 0::2] = preds[idx].detach()
                            lstm_input[0, :, 1::2] = hinput.detach()
                    

                if self.elwise:
                    tsize = lstm_input.size(0)
                    bsize = lstm_input.size(1)
                
                #lstm_input = torch.cat([train_x, train_onehot_labels], dim=1).unsqueeze(0) # [1 (seq len), batch size, num features]
                output, (hn, cn)  = lstm(lstm_input, prevh=hn, prevc=cn)
                

                # average over hidden states for every layer and repeat along batch dimension
                if not self.use_layers:
                    if self.avg_hidden:
                        hn = hn.mean(dim=1).unsqueeze(1).repeat(1,hn.size(1),1) # hn shape: [num_layers, batch size, hidden size]
                    if self.avg_cell:
                        cn = cn.mean(dim=1).unsqueeze(1).repeat(1,cn.size(1),1)  
                else:
                    if self.avg_hidden:
                        for h in hn:
                            h = h.mean(dim=1).unsqueeze(1).repeat(1,h.size(1),1) # hn shape: [num_layers, batch size, hidden size]
                    if self.avg_cell:
                        for c in cn:
                            c = c.mean(dim=1).unsqueeze(1).repeat(1,c.size(1),1)    
                
                hidden[idx] = hn
                cell[idx] = cn

                if self.elwise:
                    output = output.reshape(tsize, bsize, -1)

                # scale to account for BatchNorm in omniglot & feedforward classifier
                if i > 0 and not self.sine and not self.no_batchnorm:
                    gamma, sigma = stats[len(stats)-2*(i-1)-2: len(stats)-2*(i-1)]
                    output = (output.reshape(bsize,-1) * gamma/sigma).unsqueeze(0) 
                    

                if idx == 0:
                    # compute the 2D hidden state H by summing the outer products
                    H_new = torch.einsum("bo,bi->oi", output.squeeze(0), train_x)/output.size(1)
                else:
                    # compute the 2D hidden state H by summing the outer products
                    H_new = torch.einsum("bo,bi->oi", output.squeeze(0), preds[idx-1])/output.size(1)

                    
                    #print(torch.autograd.grad(regularizer_cost, list(self.body_lstm.parameters()) + list(self.output_lstm.parameters()), allow_unused=True) )

                # if self.gamma is not None:
                #     norm = H_new.norm()
                #     H_new = self.gamma*(H_new/norm) # clipped norm to be self.gamma 

                H_old = Hs[idx].clone()
                if self.gamma is None:
                    # add to previous H, multiply update with the learning rate if applicable
                    if self.param_lr:
                        Hs[idx] = Hs[idx] + self.body_lstm.lr * H_new
                    else:
                        Hs[idx] = Hs[idx] + H_new
                else:
                    if newHs is None:
                        newHs = [H_new]
                    else:
                        newHs.append(H_new)
                
                if self.update_bias:
                    if self.gamma is None:
                        # add to previous H, multiply update with the learning rate if applicable
                        if self.param_lr:
                            Bs[idx] = Bs[idx] + self.body_lstm.lr * output.squeeze(0).mean(dim=0)
                        else:
                            Bs[idx] = Bs[idx] + output.squeeze(0).mean(dim=0)
                    else:
                        # add to previous H, multiply update with the learning rate if applicable
                        if self.param_lr:
                            Bs[idx] = Bs[idx] + self.gamma * self.body_lstm.lr * output.squeeze(0).mean(dim=0)
                        else:
                            Bs[idx] = Bs[idx] + self.gamma * output.squeeze(0).mean(dim=0)

                # if train_mode and self.gamma is not None and t == T-1:
                #     if regularizer_cost is None:
                #         regularizer_cost = torch.norm(Hs[idx])**2
                #     else:
                #         regularizer_cost = regularizer_cost + torch.norm(Hs[idx])**2
            

            if self.gamma is not None:
                total_h = torch.cat([p.reshape(-1) for p in newHs]) # compute the norm of the entire parameter vector 
                norm = total_h.norm()

                for i in range(len(Hs)):
                    idx = len(Hs)-(i+1)
                    # add to previous H, multiply update with the learning rate if applicable
                    if self.param_lr:
                        Hs[idx] = Hs[idx] + self.body_lstm.lr * self.gamma*(newHs[i]/norm)
                    else:
                        Hs[idx] = Hs[idx] + self.gamma*(newHs[i]/norm)


        preds, stats = self.convnet.layer_wise_preds(test_x, Hs, Bs)
        preds = preds[-1]

        test_loss = self.convnet.criterion(preds, test_y) 
        
        # if train_mode and self.gamma is not None:
        #     #print(f'test_loss: {test_loss}, regularizer: {regularizer_cost}')
        #     test_loss = test_loss + self.gamma * regularizer_cost
        # print("applying regularization")


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
        self.body_lstm.train()
        self.output_lstm.train()
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
        self.body_lstm.eval()
        self.output_lstm.eval()
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
        if not self.gamma is None:
            return tuple([[p.clone().detach() for p in self.body_lstm.parameters()], [p.clone().detach() for p in self.output_lstm.parameters()]]), [p.clone().detach() for p in self.convnet.parameters()], self.gamma.clone().detach()
        return tuple([[p.clone().detach() for p in self.body_lstm.parameters()], [p.clone().detach() for p in self.output_lstm.parameters()]]), [p.clone().detach() for p in self.convnet.parameters()]
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        if not self.gamma is None:
            lstm_params, conv_params, gamma = state
            self.gamma = gamma.clone().detach()
            self.gamma.requires_grad = True
        else:
            lstm_params, conv_params = state
        
        body_lstm_params, output_lstm_params = lstm_params

        for p,v in zip(self.body_lstm.parameters(), body_lstm_params):
            p.data = v.data
            p.requires_grad = True
        
        for p,v in zip(self.output_lstm.parameters(), output_lstm_params):
            p.data = v.data
            p.requires_grad = True

        for p,v in zip(self.convnet.parameters(), conv_params):
            p.data = v.data
            p.requires_grad = True

        
    def to(self, device):
        self.body_lstm = self.body_lstm.to(device)
        self.output_lstm = self.output_lstm.to(device)
        self.convnet = self.convnet.to(device)
