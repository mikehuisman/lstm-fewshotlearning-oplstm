# Are LSTMs Good Few-Shot Learners? Outer Product LSTM (OP-LSTM) is!

This is the GitHub repository accompanying the paper titled *Are LSTMs Good Few-Shot Learners? Outer Product LSTM (OP-LSTM) is!*. Below, you can find the instructions to reproduce the results in the paper. The requirements file is called *reqs.txt*. We used Python 3.8.8 for running all experiments. 



## Permutation invariance for the plain LSTM

**Sine wave regression**:

*Sequential:* 
python -u main.py --problem sine --model simplelstm --N 1 --k $1 --k_test 50 --runs 3 --val_after 2500 --cpu --hidden_size 40 --T 4 --validate --loss_type post --meta_batch_size 4 --num_layers 2 --lr 0.004673978383086751 --model_spec bestsimplelstm-sequential --lstm_inputs currtarget 

*Batch:* 
python -u main.py --problem sine --model simplelstm --N 1 --k $1 --k_test 50 --runs 3 --val_after 2500 --cpu --hidden_size 40 --T 4 --validate --loss_type post --meta_batch_size 4 --num_layers 2 --lr 0.004673978383086751 --model_spec bestsimplelstm-batch


**Image classification**:

*Sequential:* 
python -u main.py --problem omniglot --model simplelstm --N 5 --k $2 --k_test 10 --runs 1 --val_after 30000 --hidden_size 1 --T 3 --validate --loss_type post --meta_batch_size 12 --num_layers 1 --lr 8.333333333333333e-05 --model_spec final-simplelstm-sequential --layers 1024,600,400,200,92 --train_iters 960000 --seed $1 --lstm_inputs currtarget --fcnn

*Batch:*
python -u main.py --problem omniglot --model simplelstm --N 5 --k $2 --k_test 10 --runs 1 --val_after 30000 --hidden_size 1 --T 3 --validate --loss_type post --meta_batch_size 12 --num_layers 1 --lr 8.333333333333333e-05 --model_spec final-simplelstm-batch --layers 1024,600,400,200,92 --train_iters 960000 --seed $1 --fcnn


## Performance comparison on few-shot sine wave regression

*MAML*:
python -u main.py --problem sine --model maml --second_order --N 1 --k $1 --k_test 50 --runs 3 --val_after 2500 --cpu --T 14 --validate --meta_batch_size 2 --lr 0.0032937018686863585 --base_lr 0.003769030557807799 --model_spec somaml-best-hsize91 --hdims [91,91,91]

*Plain LSTM*: 
python -u main.py --problem sine --model simplelstm --N 1 --k $1 --k_test 50 --runs 3 --val_after 2500 --cpu --hidden_size 40 --T 4 --validate --loss_type post --meta_batch_size 4 --num_layers 2 --lr 0.004673978383086751 --model_spec bestsimplelstm-batch

*OP-LSTM*:
python -u main.py --problem sine --model oplstm --second_order --N 1 --k $1 --k_test 50 --runs 3 --val_after 2500 --cpu --validate --meta_batch_size 1 --lr 0.0010448688371510757 --model_spec oplstm-70k-best --layers 5,5,1 --hidden_size 1 --T 9 --elwise --learn_init_weight --lstm_inputs target_pred --hdims [91,91,91]  --gamma 0.025



## Performance comparison on few-shot image classification

### Omniglot
*MAML*:
python -u main.py --problem omniglot --k_test 15 --model maml --validate --val_after 80000 --T 1 --model_spec best-maml-best-secondorder --k $2 --N 5 --T_test 3 --T_val 3 --meta_batch_size 32 --runs 1 --train_iters 2560000 --seed $1 --base_lr 0.4 --lr 0.00003125 --fcnn --second_order 

*ProtoNet*:
python -u main.py --problem omniglot --k_test 15 --model protonet --validate --val_after 10000 --model_spec protonet --k $2 --N 5 --meta_batch_size 4 --runs 1 --train_iters 320000 --seed $1 --lr 0.00025 --fcnn 

*Plain LSTM*:
python -u main.py --problem omniglot --model simplelstm --N 5 --k $2 --k_test 15 --runs 1 --val_after 30000 --hidden_size 1 --T 3 --validate --loss_type post --meta_batch_size 12 --num_layers 1 --lr 8.333333333333333e-05 --model_spec final-best-simplelstm-batch --layers 1024,600,400,200,92 --train_iters 960000 --seed $1 --fcnn

*OP-LSTM*:
python -u main.py --problem omniglot --model oplstm --N 5 --k $2 --k_test 15 --runs 1  --loss_type post --model_spec oplstm-elwise-best-secondorder --backbone conv4 --meta_batch_size 4 --layers 5,5,5 --hidden_size 5 --T 1 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --fcnn --second_order 


### MiniImageNet

*MAML:*
python -u main.py --problem min --k_test 15 --model maml --validate --val_after 10000 --T 5 --model_spec best-somaml --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --train_iters 320000 --second_order --seed $1 --cross_eval

python -u main.py --problem min --k_test 15 --model maml --validate --val_after 10000 --T 5 --model_spec best-somaml --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --train_iters 320000 --second_order --seed $1 --cross_eval
 
 
 *ProtoNet*:
python -u main.py --problem min --k_test 15 --model protonet --validate --val_after 10000 --model_spec protonet --k $2 --N 5 --meta_batch_size 4 --runs 1 --train_iters 320000 --seed $1 --lr 0.00025 --fcnn --cross_eval

*SAP*:
python -u main.py --problem $3 --N 5 --k $2 --k_test 15 --model sap --model_spec fsap-best-T1-MBS4 --linear_transform --val_after 2500 --second_order --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 10 --meta_batch_size 4 --T_val 10 --channel_scale --svd --grad_clip 10 --old --base_lr 0.0360774985854036 --seed $1 --single_run --validate --cross_eval  

*WARP-MAML*:
python -u main.py --problem $3 --k_test 15 --backbone conv4 --model sap --val_after 2500 --T 5 --k $2 --k_train 5 --N 5 --T_test 5 --T_val 5 --meta_batch_size 1 --runs 1 --single_run --base_lr 0.1 --model_spec warpgrad-reproduce-final-lr0.1-c64 --second_order --out_channels 64 --transform_out_channels 64 --tnet --gamma 0 --reg null --warpgrad --use_bias --validate --train_iters 60000 --seed $1 --cross_eval  

*Plain LSTM*:
python -u main.py --problem min --model simplelstm --N 5 --k $2 --k_test 15 --runs 1  --loss_type post --model_spec final-best-simplelstm --backbone conv4 --validate --meta_batch_size 32 --lr 0.000069 --num_layers 2 --zero_supp False --final_linear True --hidden_size 1904 --T 3 --val_after 80000 --train_iters 2560000 --seed $1 --cross_eval


*OP-LSTM*:

python -u main.py --problem min --model oplstm --N 5 --k $2 --k_test 15 --runs 1  --loss_type post --model_spec final-best-oplstm --backbone conv4 --meta_batch_size 4 --layers 40,20,5 --hidden_size 5 --T 3 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --gamma 0.025 --cross_eval

### CUB

*MAML*:
python -u main.py --problem cub --k_test 15 --model maml --validate --val_after 10000 --T 5 --model_spec best-somaml --k 1 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --train_iters 320000 --second_order --seed $1 --cross_eval

python -u main.py --problem cub --k_test 15 --model maml --validate --val_after 10000 --T 5 --model_spec best-somaml --k 5 --N 5 --T_test 10 --T_val 10 --meta_batch_size 4 --runs 1 --train_iters 320000 --second_order --seed $1 --cross_eval

 *ProtoNet*:
python -u main.py --problem cub --k_test 15 --model protonet --validate --val_after 10000 --model_spec protonet --k $2 --N 5 --meta_batch_size 4 --runs 1 --train_iters 320000 --seed $1 --lr 0.00025 --fcnn --cross_eval

*SAP*:
python -u main.py --problem $3 --N 5 --k $2 --k_test 15 --model sap --model_spec fsap-best-T1-MBS4 --linear_transform --val_after 2500 --second_order --T 1 --gamma 0 --runs 1 --learn_alfas --reg null --T_test 10 --meta_batch_size 4 --T_val 10 --channel_scale --svd --grad_clip 10 --old --base_lr 0.0360774985854036 --seed $1 --single_run --validate --cross_eval  

*WARP-MAML*:
python -u main.py --problem $3 --k_test 15 --backbone conv4 --model sap --val_after 2500 --T 5 --k $2 --k_train 5 --N 5 --T_test 5 --T_val 5 --meta_batch_size 1 --runs 1 --single_run --base_lr 0.1 --model_spec warpgrad-reproduce-final-lr0.1-c64 --second_order --out_channels 64 --transform_out_channels 64 --tnet --gamma 0 --reg null --warpgrad --use_bias --validate --train_iters 60000 --seed $1 --cross_eval  

*SimpleLSTM*:
python -u main.py --problem cub --model simplelstm --N 5 --k $2 --k_test 15 --runs 1  --loss_type post --model_spec final-best-simplelstm --backbone conv4 --validate --meta_batch_size 32 --lr 0.000069 --num_layers 2 --zero_supp False --final_linear True --hidden_size 1904 --T 3 --val_after 80000 --train_iters 2560000 --seed $1 --cross_eval


*OP-LSTM*:
python -u main.py --problem cub --model oplstm --N 5 --k $2 --k_test 15 --runs 1  --loss_type post --model_spec final-best-oplstm --backbone conv4 --meta_batch_size 4 --layers 40,20,5 --hidden_size 5 --T 3 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --gamma 0.025 --cross_eval



## Analysis of the learned weight updates

### MiniImageNet

python -u main.py --problem min --model oplstm --N 5 --k 1 --k_test 15 --runs 1  --loss_type post --model_spec measure-best-oplstm --backbone conv4 --meta_batch_size 4 --layers 40,20,5 --hidden_size 5 --T 3 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --gamma 0.025 --cross_eval --analyze_learned



python -u main.py --problem min --model oplstm --N 5 --k 5 --k_test 15 --runs 1  --loss_type post --model_spec measure-best-oplstm --backbone conv4 --meta_batch_size 4 --layers 40,20,5 --hidden_size 5 --T 3 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --gamma 0.025 --cross_eval --analyze_learned

### CUB

python -u main.py --problem cub --model oplstm --N 5 --k 5 --k_test 15 --runs 1  --loss_type post --model_spec measure-best-oplstm --backbone conv4 --meta_batch_size 4 --layers 40,20,5 --hidden_size 5 --T 3 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --gamma 0.025 --cross_eval --analyze_learned



python -u main.py --problem cub --model oplstm --N 5 --k 1 --k_test 15 --runs 1  --loss_type post --model_spec measure-best-oplstm --backbone conv4 --meta_batch_size 4 --layers 40,20,5 --hidden_size 5 --T 3 --val_after 10000 --seed $1  --validate --lr 0.00025 --train_iters 320000 --elwise --learn_init_weight --lstm_inputs target_pred --gamma 0.025 --cross_eval --analyze_learned




