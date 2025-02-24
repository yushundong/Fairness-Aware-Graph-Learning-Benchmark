# Fairness-Aware-Graph-Learning-Benchmark


This is the open-source code for the paper A Benchmark for Fairness-Aware Graph Learning.

Step 1: 
For Group Fairness, select a model from [ 'FairGNN', 'FairEdit', 'EDITS', 'NIFTY', 'GNN', 'CrossWalk', 'DeepWalk','FairWalk','FairVGNN'].
For Individual Fairness, select a model from ['InFoRM', 'REDRESS', 'GUIDE', 'GNN_individual'].

Step 2:
Select a dataset from ['german', 'recidivism', 'credit', 'pokec_z', 'pokec_n', 'AMiner-S', 'AMiner-L'].

Step 3: 
Specify the parameter settings in command using --param1 xxx and --param2 xxx. If you want to use the optimal parameters, please set --use_optimal True, and the parameters will be loaded from the param.json file.

Step 4:
Run train.py


Example Command:

Set model_name='FairGNN' and dataset='credit' in train.py.

python train.py --use_optimal True

Output Log:

200526
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:06<00:00, 76.14it/s]
Optimization Finished! Best Epoch: 330

ACC 0.779733
AUCROC 0.546777
F1 0.779733
ACC_sens0 0.782177
AUCROC_sens0 0.544908
F1_sens0 0.782177
ACC_sens1 0.754198
AUCROC_sens1 0.562765
F1_sens1 0.754198
SP 0.017117
EO 0.004074

Set model_name='FairGNN' and dataset='recidivism' in train.py.

403977
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:26<00:00, 18.65it/s]
Optimization Finished! Best Epoch: 475

ACC 0.909091
AUCROC 0.889717
F1 0.909091
ACC_sens0 0.908308
AUCROC_sens0 0.895598
F1_sens0 0.908308
ACC_sens1 0.909850
AUCROC_sens1 0.882804
F1_sens1 0.909850
SP 0.064752
EO 0.026093

Set model_name='NIFTY' and dataset='recidivism' in train.py.

403977
[Train] Epoch 0:train_s_loss: 0.0424 | train_c_loss: 12.7739 | val_s_loss: -0.0894 | val_c_loss: 19.9467 | val_auc_roc: 0.4269
[Train] Epoch 100:train_s_loss: -0.3621 | train_c_loss: 0.7362 | val_s_loss: -0.3574 | val_c_loss: 1.4838 | val_auc_roc: 0.4629
[Train] Epoch 200:train_s_loss: -0.4461 | train_c_loss: 0.2879 | val_s_loss: -0.4317 | val_c_loss: 0.6444 | val_auc_roc: 0.7710
[Train] Epoch 300:train_s_loss: -0.4724 | train_c_loss: 0.2064 | val_s_loss: -0.4542 | val_c_loss: 0.4812 | val_auc_roc: 0.8660

ACC 0.872219
AUCROC 0.851447
F1 0.872219
ACC_sens0 0.866552
AUCROC_sens0 0.852740
F1_sens0 0.866552
ACC_sens1 0.877713
AUCROC_sens1 0.849294
F1_sens1 0.877713
SP 0.059783
EO 0.014832

Set model_name='GNN' and dataset='credit' in train.py.

200526
tensor([0, 0, 1,  ..., 0, 0, 0])
[Train] Epoch 0: train_c_loss: 51.9787 | val_c_loss: 80.3643
[Train] Epoch 100: train_c_loss: 0.9164 | val_c_loss: 2.3468
[Train] Epoch 200: train_c_loss: 0.8849 | val_c_loss: 0.7335
[Train] Epoch 300: train_c_loss: 1.1991 | val_c_loss: 0.7125

ACC 0.727867
AUCROC 0.629570
F1 0.727867
ACC_sens0 0.736158
AUCROC_sens0 0.628480
F1_sens0 0.736158
ACC_sens1 0.641221
AUCROC_sens1 0.624015
F1_sens1 0.641221
SP 0.167765
EO 0.159288
