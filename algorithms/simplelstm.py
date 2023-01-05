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

class SimpleLSTM(Algorithm):
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
                 avg_cell=False, loss_type="post", grad_clip=None, meta_batch_size=1, layers=None, 
                 lstm_inputs=None, sine=False, **kwargs):
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
        self.N = N
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size
        self.zero_test = zero_test # use all zeros for the previous target when testing
        self.log_test_norm = False
        self.measure_distances = False
        self.loss_type = LossType.PostAdaptation if loss_type.lower().strip() == "post" else LossType.MultiStep
        self.avg_cell = avg_cell
        self.layers=layers
        self.lstm_inputs = lstm_inputs

        print("SELF.SINE:", self.sine)


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
        self.baselearner_args["layers"] = layers
        self.baselearner_args["output_size"] = self.N


        if not self.lstm_inputs is None:
            if self.lstm_inputs == "prevtarget" or self.lstm_inputs == "currtarget":
                self.baselearner_args["input_size"] = 2 if self.sine else 784 + self.baselearner_args["eval_classes"]
            elif self.lstm_inputs == "prev_target_pred" or self.lstm_inputs == "prev_target_err":
                self.baselearner_args["input_size"] = 3
            elif self.lstm_inputs == "prev_target_pred_err":    
                self.baselearner_args["input_size"] = 4
        
        print(self.baselearner_args["input_size"])

        #self.baselearner_args["input_size"] = 2 if self.N == 1 else 28**2 + self.N
        print("hidden size:", hidden_size, "num_layers:", num_layers)
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
                

        count_params = sum([p.numel() for p in self.baselearner.parameters()])
        print("Number of parameters:", count_params)
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(self.baselearner.parameters(), lr=self.lr)
        
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]
        
        self.log_interval = 80
        self.test_losses = []
        self.test_norms = []  
        self.test_perfs = []    
        self.angles = []
        self.distances = []
        self.gangles = []
        self.gdistances = []      


    def prepare_data(self, x, y, T, zero=False):
        """
        Prepares data to be in the right format to be fed into the LSTM. 
        Input = x_t, y_t-1 at every timestep. For t=0, y_0 = 0. 

        zero: should be True when evaluating performance or when testing (otherwise ground-truth labels will be used)

        """

        x = x.reshape(x.size(0), -1)

        # x: [batch_size, input features]
        if len(x.size()) == 2:
            x = x.unsqueeze(0).repeat(T,1,1) # repeat input T times [T, batch_size, input features]
            if self.baselearner.output_size > 1:
                y = torch.nn.functional.one_hot(y)
                y = y.unsqueeze(0).repeat(T,1,1)
                input_batch = torch.cat([x, y], dim=2)

                if zero: # test set should receive no labels
                    wanted_size = list(x.size()[:-1]) + [self.N] # [T, batch_size, 1]
                    x = torch.cat([x, torch.zeros(wanted_size, device=self.dev)], dim=2)
                    return x, y

                return input_batch, y

            
            y = y.unsqueeze(0).repeat(T,1,1) # repeat output T times [T, batch_size, input features]

            # make the previous targets zero in the input sequence
            if zero:
                wanted_size = list(x.size()[:-1]) + [self.baselearner_args["input_size"]-1] # [T, batch_size, 1]
                x = torch.cat([x, torch.zeros(wanted_size, device=self.dev)], dim=2) # concatenate the zeros so that new x is [T, batch_size, input features + self.N]
                return x, y


            # put a zero first: we do not know the real label cuz there was no input yet --> value = 0
            init_zeros = torch.zeros(y.size(2)*x.size(1),device=self.dev).reshape(1, x.size(1), x.size(2)) # zero tensor of shape [1, batch_size, out dim (1 for regression)]
            shifted_output_batch = torch.cat([init_zeros, y], dim=0) # [seq len + 1, batch size, out dim]
            input_batch = torch.cat([x, shifted_output_batch[:x.size(0),:,:]], dim=2) #[seq len, batch_size, infeatures+1]
            # print(x)
            # print('-'*40)
            # print(y)
            # print('-'*40)
            # print(input_batch)
            # import sys; sys.exit()
            return input_batch, y




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

        if self.lstm_inputs is None:
            train_x, train_y = self.prepare_data(train_x, train_y, T) # [seq len, batch_size, infeatures]
            test_x, one_hot_y = self.prepare_data(test_x, test_y, T=1, zero=True) # [seq len, batch_size, infeatures]
        
        
            # go over the training data --- ingest data into the hidden state
            hn, cn = None, None
            predictions = []
            if train_mode and self.loss_type == LossType.MultiStep:
                tr_loss = torch.zeros(1)

            for t in range(T):
                output, (hn, cn)  = self.baselearner(train_x[t,:,:].unsqueeze(0), prevh=hn, prevc=cn)
                preds = self.baselearner.predict(output)
                predictions.append(preds)
                if train_mode and self.loss_type == LossType.MultiStep:
                    tr_loss = tr_loss + self.baselearner.criterion(preds, train_y[t,:,:].unsqueeze(0))
                
                if self.baselearner.output_size == 1:
                    # average over hidden states for every layer and repeat along batch dimension
                    hn = hn.mean(dim=1).unsqueeze(1).repeat(1,train_x.size(1),1) # hn shape: [num_layers, batch size, hidden size]
                    if self.avg_cell:
                        cn = cn.mean(dim=1).unsqueeze(1).repeat(1,train_x.size(1),1)
                else:
                    #print([p.size() for p in hn])
                    hn = [el.mean(dim=1).unsqueeze(1).repeat(1, el.size(1), 1) for el in hn]
                #print([p.size() for p in hn])
        else:
            #train_x, train_y = self.prepare_data(train_x, train_y, T) # [seq len, batch_size, infeatures]
            train_x = train_x.reshape(train_x.size(0), -1)
            train_y = torch.nn.functional.one_hot(train_y) if not self.sine else train_y 
            test_x, one_hot_y = self.prepare_data(test_x, test_y, T=1, zero=True) # [seq len, batch_size, infeatures]
        
        
            # go over the training data --- ingest data into the hidden state
            hn, cn = None, None
            #predictions = []

            for t in range(T):
                # go over every input in the batch + 1 because we also need to see the ground-truth output
                # for the last input
                for idx in range(train_x.size(0)+1):
                    if idx == train_x.size(0):
                        if self.lstm_inputs == "currtarget":
                            break
                        inp = train_x[0].unsqueeze(0)
                        gt = train_y[0]
                    else:
                        inp = train_x[idx].unsqueeze(0)
                        gt = train_y[idx]
                    
                    if idx == 0 and not self.lstm_inputs == "currtarget":
                        if self.sine:
                            z = torch.zeros(self.baselearner_args["input_size"]-1, device=train_x.device)
                        else:
                            z = torch.zeros(self.baselearner_args["eval_classes"], device=train_x.device)
                        inp = torch.cat([inp, z.unsqueeze(0)], dim=1)

                    elif self.lstm_inputs == "prevtarget":
                        inp = torch.cat([inp, train_y[idx-1].unsqueeze(0)], dim=1)
                    elif self.lstm_inputs == "currtarget":
                        inp = torch.cat([inp, train_y[idx].unsqueeze(0)], dim=1)
                    elif self.lstm_inputs == "prev_target_pred":
                        inp = torch.cat([inp, train_y[idx-1].unsqueeze(0), prevpred.unsqueeze(0)], dim=1)

                    elif self.lstm_inputs == "prev_target_err":
                        inp = torch.cat([inp, train_y[idx-1].unsqueeze(0), preverr.unsqueeze(0)], dim=1)
                    elif self.lstm_inputs == "prev_target_pred_err":    
                        inp = torch.cat([inp, train_y[idx-1].unsqueeze(0), prevpred.unsqueeze(0), preverr.unsqueeze(0)], dim=1)

                    inp = inp.unsqueeze(0) # size will be [1, 1, input_size]


                    output, (hn, cn)  = self.baselearner(inp, prevh=hn, prevc=cn)
                    preds = self.baselearner.predict(output)
                    #predictions.append(preds)
                    prevpred = preds.reshape(-1)
                    preverr = (prevpred - gt)**2

                
                # if self.baselearner.output_size == 1:
                #     # average over hidden states for every layer and repeat along batch dimension
                #     hn = hn.mean(dim=1).unsqueeze(1).repeat(1,train_x.size(1),1) # hn shape: [num_layers, batch size, hidden size]
                #     if self.avg_cell:
                #         cn = cn.mean(dim=1).unsqueeze(1).repeat(1,train_x.size(1),1)
                # else:
                #     #print([p.size() for p in hn])
                #     hn = [el.mean(dim=1).unsqueeze(1).repeat(1, el.size(1), 1) for el in hn]
                #print([p.size() for p in hn])



        # if not self.zero_test:
        #     print(test_x[:,:,-self.N:].size())
        #     print(preds[-1,:,-self.N:].clone().detach().unsqueeze(0).repeat(test_x.size(0),1,1).size())
        #     test_x[:,:,-self.N:] = preds[-1,:,-self.N:].clone().detach().unsqueeze(0).repeat(test_x.size(0),1,1)
        #     print("Put previous targets to predicted targets")


        # clone and adjust batch dimension to fit query examples
        if self.baselearner.output_size == 1:
            fast_hn = hn.clone()[:,0,:].unsqueeze(1).repeat(1,test_x.size(1),1) # [seq]
            if not self.avg_cell:
                fast_cn = None # reset cell state -- input specificcn.clone()[:,0,:].unsqueeze(1).repeat(hn.size(0),test_x.size(1),hn.size(2))
            else:
                fast_cn = cn.clone()[:,0,:].unsqueeze(1).repeat(1,test_x.size(1),1)
            output, (fast_hn, fast_cn)  = self.baselearner(test_x, prevh=fast_hn, prevc=fast_cn)
            preds = self.baselearner.predict(output)

            test_loss = self.baselearner.criterion(preds.squeeze(0), test_y)    

        else:
            fast_hn = [el.clone()[:,0,:].unsqueeze(1).repeat(1,test_x.size(1),1) for el in hn]# [seq]
            fast_cn = None # reset cell state -- input specificcn.clone()[:,0,:].unsqueeze(1).repeat(hn.size(0),test_x.size(1),hn.size(2))
            
            output, (fast_hn, fast_cn)  = self.baselearner(test_x, prevh=fast_hn, prevc=fast_cn)
            preds = self.baselearner.predict(output.squeeze(0))
            test_loss = self.baselearner.criterion(preds, test_y)   


             
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
        self.baselearner.train()
        self.task_counter += 1
        self.global_counter += 1

        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, _ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)
        test_loss.backward()
            
        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)


        if self.task_counter % self.meta_batch_size == 0: 
            if self.global_counter % self.log_interval == 0 and self.using_gpu:
                self.gpu_usage.append(self.gpu.memoryUsed)

            # for p in self.baselearner.parameters():
            #     print(p.grad, p.size())
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
        
        return [p.clone().detach() for p in self.baselearner.parameters()]

    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        for p,v in zip(self.baselearner.parameters(), state):
            p.data = v.data
            p.requires_grad = True

        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.to(device) for p in self.initialization]