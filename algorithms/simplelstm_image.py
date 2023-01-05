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

class SimpleImageLSTM(Algorithm):
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
    
    def __init__(self, N, hidden_size, num_layers, train_base_lr, base_lr, second_order, zero_test, 
                 avg_cell=False, loss_type="post", grad_clip=None, meta_batch_size=1, image=False, lstm_constructor=None, 
                 final_linear=True, zero_supp=True, hyper=False, shuffle=False, **kwargs):
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
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.zero_test = zero_test # use all zeros for the previous target when testing
        self.zero_supp = zero_supp
        print("ZERO_SUPP:", self.zero_supp)
        self.log_test_norm = False
        self.measure_distances = False
        self.loss_type = LossType.PostAdaptation if loss_type.lower().strip() == "post" else LossType.MultiStep
        self.avg_cell = avg_cell
        self.lstm_constructor = lstm_constructor
        self.final_linear = final_linear
        self.hyper = hyper
        self.shuffle = shuffle
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
        self.baselearner_args["no_output_layer"] = True

        print("hidden size:", hidden_size, "num_layers:", num_layers)

        self.convnet = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.baselearner_args["input_size"] = self.convnet.in_features + self.convnet.eval_classes 
        self.baselearner_args["output_size"] = self.convnet.eval_classes 
        self.baselearner_args["final_linear"] = self.final_linear
        self.baselearner_args["hyper"] = self.hyper
        self.lstm = self.lstm_constructor(**self.baselearner_args).to(self.dev)
        
        print("SELF.HYPER:", self.hyper)
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(list(self.convnet.parameters()) + list(self.lstm.parameters()), lr=self.lr)
        
        num_params = sum([p.numel() for p in list(self.convnet.parameters()) + list(self.lstm.parameters())])
        print("Number of parameters:", num_params)

        self.log_interval = 80
        self.test_losses = []
        self.test_norms = []  
        self.test_perfs = []    
        self.angles = []
        self.distances = []
        self.gangles = []
        self.gdistances = []      


    # def prepare_data(self, x, y, T, zero=False):
    #     """
    #     Prepares data to be in the right format to be fed into the LSTM. 
    #     Input = x_t, y_t-1 at every timestep. For t=0, y_0 = 0. 

    #     zero: should be True when evaluating performance or when testing (otherwise ground-truth labels will be used)

    #     """
    #     # x: [batch_size, input features]
    #     if len(x.size()) == 2:
    #         x = x.unsqueeze(0).repeat(T,1,1) # repeat input T times [T, batch_size, input features]
    #         y = y.unsqueeze(0).repeat(T,1,1) # repeat output T times [T, batch_size, input features]

    #         # make the previous targets zero in the input sequence
    #         if zero:
    #             wanted_size = list(x.size()[:-1]) + [1] # [T, batch_size, 1]
    #             x = torch.cat([x, torch.zeros(wanted_size, device=self.dev)], dim=2) # concatenate the zeros so that new x is [T, batch_size, input features + 1]
    #             return x, y


    #         # put a zero first: we do not know the real label cuz there was no input yet --> value = 0
    #         init_zeros = torch.zeros(y.size(2)*x.size(1),device=self.dev).reshape(1, x.size(1), x.size(2)) # zero tensor of shape [1, batch_size, out dim (1 for regression)]
    #         shifted_output_batch = torch.cat([init_zeros, y], dim=0) # [seq len + 1, batch size, out dim]
    #         input_batch = torch.cat([x, shifted_output_batch[:x.size(0),:,:]], dim=2) #[seq len, batch_size, infeatures+1]
    #         # print(x)
    #         # print('-'*40)
    #         # print(y)
    #         # print('-'*40)
    #         # print(input_batch)
    #         # import sys; sys.exit()
    #         return input_batch, y


    # def _deploy_sequential(self, train_x, train_y, test_x, test_y, train_mode, T):
    #     """Run DOSO on a single task to get the loss on the query set
        
    #     1. Evaluate the base-learner loss and gradients on the support set (train_x, train_y)
    #     using our initialization point.
    #     2. Make a single weight update based on this information.
    #     3. Evaluate and return the loss of the fast weights (initialization + proposed updates)
        
    #     Parameters
    #     ----------
    #     train_x : torch.Tensor
    #         Inputs of the support set
    #     train_y : torch.Tensor
    #         Outputs of the support set
    #     test_x : torch.Tensor
    #         Inputs of the query set
    #     test_y : torch.Tensor
    #         Outputs of the query set
    #     train_mode : boolean
    #         Whether we are in training mode or test mode

    #     Returns
    #     ----------
    #     test_loss
    #         Loss of the base-learner on the query set after the proposed
    #         one-step update
    #     """
        
    #     loss_history = None
    #     if not train_mode: loss_history = []


    #     # embed the features
    #     train_x, test_x = self.convnet.get_infeatures(train_x), self.convnet.get_infeatures(test_x) # they now have size [batch, infeatures]

    #     train_x = train_x.unsqueeze(1) #[seq len (batch size), 1, infeatures]

    #     # none labels for initial input
    #     init_zeros = torch.zeros((1, self.convnet.eval_classes), device=self.dev).reshape(1, 1, -1) #[1,1,5]
    #     # the one-hot labels for passes 2,...,T
    #     train_onehot_labels = torch.nn.functional.one_hot(train_y).reshape(train_x.size(0), 1, -1) #[ batch size, 1, 5]

    #     appen = torch.cat([init_zeros, train_onehot_labels], dim=0)[:train_x.size(0)]

    #     lstm_input = torch.cat([train_x, appen], dim=2)
    #     output, (hn, cn)  = self.lstm(lstm_input, prevh=None, prevc=None)



    #     init_zeros = torch.zeros((test_x.size(0), self.convnet.eval_classes), device=self.dev)
    #     test_input = torch.cat([test_x, init_zeros], dim=1).unsqueeze(0) # [1 (seq len), batch size, infeatures]

    #     # clone and adjust batch dimension to fit query examples
    #     fast_hn = hn.clone()[:,0,:].unsqueeze(1).repeat(1,test_input.size(1),1) # [seq(1), batch size, infeatures]
    #     if not self.avg_cell:
    #         fast_cn = None # reset cell state -- 
    #     else:
    #         fast_cn = cn.clone()[:,0,:].unsqueeze(1).repeat(1,test_input.size(1),1) # [seq(1), batch size, infeatures]


    #     output, (fast_hn, fast_cn)  = self.lstm(test_input, prevh=fast_hn, prevc=fast_cn)
    #     preds = self.lstm.predict(output.squeeze(0))

    #     test_loss = self.convnet.criterion(preds, test_y)         
    #     if train_mode and self.loss_type == LossType.MultiStep:
    #         test_loss = test_loss + tr_loss

    #     if train_mode and T > 0:
    #         self.train_losses.append(test_loss.item())

    #     if not train_mode: loss_history.append(test_loss.item())
    #     return test_loss, preds, loss_history


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
        # none labels for initial input
        init_zeros = torch.zeros((train_x.size(0), self.convnet.eval_classes), device=self.dev)
        # the one-hot labels for passes 2,...,T
        train_onehot_labels = torch.nn.functional.one_hot(train_y)

        
        # go over the training data --- ingest data into the hidden state
        hn, cn = None, None
        prevh_final, prevc_final = None, None
        if train_mode and self.loss_type == LossType.MultiStep:
            tr_loss = torch.zeros(1)

        for t in range(T):
            if self.final_linear:
                if t == 0 and self.zero_supp:
                    lstm_input = torch.cat([train_x, init_zeros], dim=1).unsqueeze(0) # [1 (seq len), batch size, num features]
                else:
                    lstm_input = torch.cat([train_x, train_onehot_labels], dim=1).unsqueeze(0) # [1 (seq len), batch size, num features]
            
                # output is the hidden state from the last layer of the LSTM, while hn and cn contain for all layers
                output, (hn, cn)  = self.lstm(lstm_input, prevh=hn, prevc=cn) # output: (1, batch size, hidden size), hn and cn: (num_layers, batch size, hidden size) 
                #print(self.convnet.criterion(preds, train_y.unsqueeze(0)))
                if train_mode and self.loss_type == LossType.MultiStep:
                    preds = self.lstm.predict(output)
                    tr_loss = tr_loss + self.convnet.criterion(preds, train_y[t,:,:].unsqueeze(0))

                # average over hidden states for every layer and repeat along batch dimension
                hn = hn.mean(dim=1).unsqueeze(1).repeat(1,lstm_input.size(1),1) # hn shape: [num_layers, batch size, hidden size]
                if self.avg_cell:
                    cn = cn.mean(dim=1).unsqueeze(1).repeat(1,lstm_input.size(1),1)
            else:
                if t == 0:
                    lstm_input = torch.cat([train_x, init_zeros], dim=1).unsqueeze(0) # [1 (seq len), batch size, num features]
                else:
                    lstm_input = torch.cat([train_x, train_onehot_labels], dim=1).unsqueeze(0) # [1 (seq len), batch size, num features]
            
                output, (hn, cn), (prevh_final, prevc_final)  = self.lstm(lstm_input, prevh=hn, prevc=cn, prevh_final=prevh_final, prevc_final=prevc_final)
                #print(self.convnet.criterion(preds, train_y.unsqueeze(0)))
                if train_mode and self.loss_type == LossType.MultiStep:
                    preds = prevh_final.squeeze(0)
                    tr_loss = tr_loss + self.convnet.criterion(preds, train_y[t,:,:].unsqueeze(0))

                # average over hidden states for every layer and repeat along batch dimension
                hn = hn.mean(dim=1).unsqueeze(1).repeat(1,lstm_input.size(1),1) # hn shape: [num_layers, batch size, hidden size]
                prevh_final = prevh_final.mean(dim=1).unsqueeze(1).repeat(1,lstm_input.size(1),1)
                if self.avg_cell:
                    cn = cn.mean(dim=1).unsqueeze(1).repeat(1,lstm_input.size(1),1)
                    prevc_final = prevc_final.mean(dim=1).unsqueeze(1).repeat(1,lstm_input.size(1),1)
                
        

        init_zeros = torch.zeros((test_x.size(0), self.convnet.eval_classes), device=self.dev)
        test_input = torch.cat([test_x, init_zeros], dim=1).unsqueeze(0) # [1 (seq len), batch size, infeatures]

        # clone and adjust batch dimension to fit query examples
        fast_hn = hn[:,0,:].unsqueeze(1).repeat(1,test_input.size(1),1) # [seq(1), batch size, infeatures] <- take the 0th batch index because they are all the same since we averaged
        
        if not self.avg_cell:
            fast_cn = None # reset cell state because they are specific to inputs (we have not seen query inputs before so start from scratch) -- 
        else:
            fast_cn = cn[:,0,:].unsqueeze(1).repeat(1,test_input.size(1),1) # [seq(1), batch size, infeatures]
        
        if self.final_linear:
            weights, bias = None, None
            if self.hyper:
                # embed individual support points, write to weights
                _, (hidden_embeddings, _)  = self.lstm(lstm_input, prevh=hn, prevc=cn)
                hidden_embeddings = hidden_embeddings.squeeze(0)
                mask_net_input = torch.cat([hidden_embeddings, train_onehot_labels], dim=1)
                weight_mask = self.lstm.weight_mask_net(mask_net_input) # shape: (batch size, num_classes)
                bias_mask = self.lstm.bias_mask_net(mask_net_input) # shape: (batch size, num_classes)

                #print(hidden_embeddings.size(), weight_mask.size(), bias_mask.size())
                # compute weight matrix and bias weight matrix
                weights = (hidden_embeddings.T @ weight_mask).T 
                bias = (hidden_embeddings.T @ bias_mask).T.sum(dim=1) # shape: (num_classes, embedding size)


            output, (fast_hn, fast_cn)  = self.lstm(test_input, prevh=fast_hn, prevc=fast_cn)
            preds = self.lstm.predict(output.squeeze(0), weight=weights, bias=bias) # if weights and bias is None, it uses its internal linear output layer
        else:
            fast_hn_final = prevh_final[:,0,:].unsqueeze(1).repeat(1,test_input.size(1),1) # [seq(1), batch size, infeatures]
            if not self.final_linear:
                if not self.avg_cell:
                    fast_cn_final = None # reset cell state -- 
                else:
                    fast_cn_final = prevc_final[:,0,:].unsqueeze(1).repeat(1,test_input.size(1),1) # [seq(1), batch size, infeatures]
            
            output, (fast_hn, fast_cn), (fast_hn_final, fast_cn_final)  = self.lstm(test_input, prevh=fast_hn, prevc=fast_cn, prevh_final=fast_hn_final, prevc_final=fast_cn_final)
            preds = fast_hn_final.squeeze(0)

        

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

            # before_conv = [p.clone().detach() for p in self.convnet.parameters()]
            # before_lstm = [p.clone().detach() for p in self.lstm.parameters()]
            
            
            # print("CONV")
            # for p in self.convnet.parameters():
            #     print(p.grad)
            # print("LSTM")
            # for p in self.lstm.parameters():
            #     print(p.grad)
            # import sys; sys.exit()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.task_counter = 0
            
            # if self.global_counter % 500 == 0: 
            #     lsumm = 0
            #     csumm = 0
            #     print("CONV")
            #     for p in self.convnet.parameters():
            #         print(p.norm())
            #         csumm += p.norm().item()
                
            #     print("LSTM")
            #     for p in self.lstm.parameters():
            #         print(p.norm())
            #         lsumm += p.norm().item()
                
                #print("Conv norm:", csumm)
                #print("LSTM norm:", lsumm)
            



            #after_lstm = [p.clone().detach() for p in self.lstm.parameters()]
            #after_conv = [p.clone().detach() for p in self.convnet.parameters()]

            # print("LSTM")
            # for old_lstm, new_lstm in zip(before_lstm, after_lstm):
            #     print(torch.all(old_lstm == new_lstm))

            # print("CONV")
            # for old_lstm, new_lstm in zip(before_conv, after_conv):
            #     print(torch.all(old_lstm == new_lstm))
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
