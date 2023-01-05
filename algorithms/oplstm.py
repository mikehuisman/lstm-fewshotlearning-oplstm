import torch
import numpy as np
import os
import psutil
import GPUtil as GPU
import torch.nn.functional as F

from .algorithm import Algorithm
from .modules.utils import put_on_device, get_loss_and_grads,\
                           accuracy, deploy_on_task, get_info
from .modules.similarity import gram_linear, cka


class LossType:
    PostAdaptation = 0
    MultiStep = 1

class OpLSTM(Algorithm):
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
                 lstm_inputs=None, param_lr=False, elwise=False, layers=None, softmax=False, gamma=None, 
                 update_bias=False, analyze_learned=False, **kwargs):
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
        print("SHUFFLING:", self.shuffle)
        self.avg_hidden = avg_hidden
        self.learn_init_weight = learn_init_weight
        self.lstm_inputs = lstm_inputs
        self.param_lr = param_lr
        self.elwise = elwise
        self.use_layers = not layers is None
        self.softmax = softmax
        self.update_bias = update_bias
        self.analyze_learned = analyze_learned

        if self.analyze_learned:
            self.cosine_stats_grad = [[] for _ in range(self.T+1)] # [ [T=1 list], [T=2 list], ... [T=final compared with T=start]]
            self.euclid_stats_grad = [[] for _ in range(self.T+1)] # [ [T=1 list], [T=2 list], ... [T=final compared with T=start]]

            self.cosine_stats_proto = [[], []] # [ [T=1 list], [T=final compared with T=start]]
            self.euclid_stats_proto = [[], []] # [ [T=1 list], [T=2 list], ... [T=final compared with T=start]]
        
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
        self.baselearner_args["softmax"] = softmax
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
                self.baselearner_args["hidden_size"] = (self.convnet.eval_classes * 1) // self.convnet.eval_classes # (new_inputsize/old_inputsize) * old_inputsize 
                if self.use_layers:
                    layers[-1] = layers[-1] * 1 // self.convnet.eval_classes
            else:
                self.baselearner_args["input_size"] = self.convnet.eval_classes  #self.convnet.in_features + 
        elif self.lstm_inputs == "target_pred":
            if self.elwise:
                self.baselearner_args["input_size"] = 2 # pred and target
                self.baselearner_args["hidden_size"] = (self.convnet.eval_classes * 2) // (2* self.convnet.eval_classes) # (new_inputsize/old_inputsize) * old_inputsize 
                if self.use_layers:
                    layers[-1] = layers[-1] * 2 // (2*self.convnet.eval_classes)
            else:
                self.baselearner_args["input_size"] = 2*self.convnet.eval_classes
            assert self.learn_init_weight, "Can only feed preds as input when we have initial W learned"

        self.baselearner_args["output_size"] = self.convnet.eval_classes 
        self.baselearner_args["elwise"] = self.elwise
        self.lstm = self.lstm_constructor(**self.baselearner_args).to(self.dev)

        print("hidden size:", hidden_size, "num_layers:", num_layers)

        # Initialize the meta-optimizer
        if not self.gamma is None:
            self.optimizer = self.opt_fn(list(self.convnet.parameters()) + list(self.lstm.parameters()) + [self.gamma], lr=self.lr)
            count_params = 1 + sum([p.numel() for p in self.convnet.parameters()]) + sum([p.numel() for p in self.lstm.parameters()])
        else:
            self.optimizer = self.opt_fn(list(self.convnet.parameters()) + list(self.lstm.parameters()), lr=self.lr)
            count_params = sum([p.numel() for p in self.convnet.parameters()]) + sum([p.numel() for p in self.lstm.parameters()])
        
        print("Number of parameters:", count_params)


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

        # and compute the Prototypical network approach where 
        if self.analyze_learned:
            if not train_mode:
                num_classes = self.convnet.eval_classes
            else:
                num_classes = self.convnet.train_classes

            # compute input embeddings
            support_embeddings = train_x.clone().detach().cpu()
            
            # compute prototypes
            prototypes = torch.zeros((num_classes, support_embeddings.size(1)), device="cpu")
            # number of eval_classes = train_classes
            for class_id in range(num_classes):
                mask = train_y == class_id
                prototypes[class_id] = 2* support_embeddings[mask].sum(dim=0) / torch.sum(mask).item()
            pH = torch.cat([p.reshape(-1) for p in prototypes])
        
        # Tanh dot-product attention 
        #ind = torch.argsort(train_y)
        #preds = test_x @ torch.tanh(train_x[ind,:].T)
        # none labels for initial input
        # the one-hot labels for passes 2,...,T

        # COMMENTING BEGINS HERE
        train_onehot_labels = torch.nn.functional.one_hot(train_y, num_classes=self.convnet.eval_classes)
        
        # go over the training data --- ingest data into the hidden state
        hn, cn = None, None

        # if train_mode and self.loss_type == LossType.MultiStep:
        #     tr_loss = torch.zeros(1)

        H = None if not self.learn_init_weight else self.lstm.init_weight.weight.clone()
        B = None if not self.update_bias else self.lstm.b.clone()



        for t in range(T):
            if self.learn_init_weight:
                p = self.lstm.init_predict(train_x, H, B)

            if self.lstm_inputs is None or self.lstm_inputs == "target":
                lstm_input = train_onehot_labels.unsqueeze(0).float()
            else:
                lstm_input = torch.empty(1, train_onehot_labels.size(0), train_onehot_labels.size(1) + p.size(1), device=train_onehot_labels.device)
                lstm_input[0, :, 0::2] = train_onehot_labels.float()
                lstm_input[0, :, 1::2] = p.detach() if not self.second_order else p

            if self.elwise:
                tsize = lstm_input.size(0)
                bsize = lstm_input.size(1)
            #lstm_input = torch.cat([train_x, train_onehot_labels], dim=1).unsqueeze(0) # [1 (seq len), batch size, num features]
            output, (hn, cn)  = self.lstm(lstm_input, prevh=hn, prevc=cn)

            # if train_mode and self.loss_type == LossType.MultiStep:
            #     init_zeros_supp = torch.zeros((train_x.size(0), self.convnet.eval_classes), device=self.dev)
            #     preds = preds = self.lstm.predict(h=output, x_train=torch.cat([train_x, train_onehot_labels], dim=1),
            #                       y_train=train_onehot_labels.float(), x_query=torch.cat([train_x, init_zeros_supp], dim=1)) 
            #     tr_loss = tr_loss + self.convnet.criterion(preds, train_y[t,:,:].unsqueeze(0))

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

            if self.elwise:
                #print("Before output shape:", output.size())
                output = output.reshape(tsize, bsize, -1)
                #print("After output shape:", output.size())
                # for h in hn:
                #     print(h.size())
                # import sys; sys.exit()


            # compute the 2D hidden state H by summing the outer products
            H_new = torch.einsum("bo,bi->oi", output.squeeze(0), train_x)/output.size(1)

            # init weight (to compute the final Euclid/Cosin sim): self.lstm.init_weight.weight.clone()

            # compute gradient updated H_new if analyze_learned is true
            if self.analyze_learned:
                # compute gradient update for H to get H_new_grad
                aloss = self.convnet.criterion(p, train_y)
                agrad = -torch.cat([g.reshape(-1).cpu() for g in torch.autograd.grad(aloss, [H])[0]])   
                Hcopy = H_new.reshape(-1).cpu()

                euclid_diff_grad = (torch.sum((agrad - Hcopy)**2)**0.5).item()
                cosine_sim_grad = F.cosine_similarity(agrad, Hcopy, dim=0).item()
                self.euclid_stats_grad[t].append(euclid_diff_grad)
                self.cosine_stats_grad[t].append(cosine_sim_grad)



            if H is None:
                if not self.gamma is None:
                    norm = H_new.norm()
                    if self.param_lr:
                        H = self.lstm.lr * self.gamma* (H_new/norm)
                    else:
                        H = self.gamma * (H_new/norm)
                else:
                    # set equal to H_new, multiplied with the learning rate if applicable
                    if self.param_lr:
                        H = self.lstm.lr * H_new
                    else:
                        H = H_new
            else:
                if not self.gamma is None:
                    norm = H_new.norm()
                    # add to previous H, multiply update with the learning rate if applicable
                    if self.param_lr:
                        H = H + self.lstm.lr * self.gamma*(H_new/norm)
                    else:
                        H = H + self.gamma*(H_new/norm)
                else:
                    # add to previous H, multiply update with the learning rate if applicable
                    if self.param_lr:
                        H = H + self.lstm.lr * H_new
                    else:
                        H = H + H_new
            
            if self.analyze_learned:
                flat_h = torch.cat([p.clone().detach().cpu().reshape(-1) for p in H])
                if t == 0:
                    euclid_diff_proto = (torch.sum((flat_h - pH)**2)**0.5).item()
                    cosine_sim_proto = F.cosine_similarity(flat_h, pH, dim=0).item()
                    self.euclid_stats_proto[t].append(euclid_diff_proto)
                    self.cosine_stats_proto[t].append(cosine_sim_proto)

            
            if self.update_bias:
                g = self.gamma if not self.gamma is None else 1
                if B is None:
                    if self.param_lr:
                        B = self.lstm.lr * g * output.squeeze(0).mean(dim=0)
                    else:
                        B = g*output.squeeze(0).mean(dim=0)
                else:
                    if self.param_lr:
                        B = B + self.lstm.lr * g * output.squeeze(0).mean(dim=0)
                    else:
                        B = B + g * output.squeeze(0).mean(dim=0)

               
        if self.analyze_learned:
            # compute gradient update for H to get H_new_grad
            aH = self.lstm.init_weight.weight.clone()
            for t in range(self.T):
                p = self.lstm.init_predict(train_x, aH, B)
                aloss = self.convnet.criterion(p, train_y)
                agrad = torch.autograd.grad(aloss, [aH])[0] 
                aH = aH - 0.01*agrad
            aH = (aH.cpu() - self.lstm.init_weight.weight.clone().cpu()).reshape(-1) # update direction
            real_update_direction = (H.cpu() - self.lstm.init_weight.weight.clone().cpu()).reshape(-1)

            euclid_diff_grad = (torch.sum((aH - real_update_direction)**2)**0.5).item()
            cosine_sim_grad = F.cosine_similarity(aH, real_update_direction, dim=0).item()
            self.euclid_stats_grad[-1].append(euclid_diff_grad)
            self.cosine_stats_grad[-1].append(cosine_sim_grad)

            flat_h = torch.cat([p.clone().detach().cpu().reshape(-1) for p in H])
            # prototypical
            euclid_diff_proto = (torch.sum((flat_h - pH)**2)**0.5).item()
            cosine_sim_proto = F.cosine_similarity(flat_h, pH, dim=0).item()
            self.euclid_stats_proto[-1].append(euclid_diff_proto)
            self.cosine_stats_proto[-1].append(cosine_sim_proto)

            # print(euclid_diff_grad, cosine_sim_grad, euclid_diff_proto, cosine_sim_proto)

        preds = self.lstm.predict(H, test_x, B)
        test_loss = self.convnet.criterion(preds, test_y)         
        # if train_mode and self.loss_type == LossType.MultiStep:
        #     test_loss = test_loss + tr_loss

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
        if not self.gamma is None:
            return [p.clone().detach() for p in self.lstm.parameters()], [p.clone().detach() for p in self.convnet.parameters()], self.gamma.clone().detach()
        return [p.clone().detach() for p in self.lstm.parameters()], [p.clone().detach() for p in self.convnet.parameters()], 
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        if self.gamma is None:
            lstm_params, conv_params = state
        else:
            lstm_params, conv_params, gamma = state
            self.gamma = gamma.clone().detach()
            self.gamma.requires_grad = True
        
        
        for p,v in zip(self.lstm.parameters(), lstm_params):
            p.data = v.data
            p.requires_grad = True

        for p,v in zip(self.convnet.parameters(), conv_params):
            p.data = v.data
            p.requires_grad = True
        

        
    def to(self, device):
        self.lstm = self.lstm.to(device)
        self.convnet = self.convnet.to(device)
