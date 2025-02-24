from dataloading import load_data
from algorithms.CrossWalk import CrossWalk
from algorithms.UGE import UGE
from algorithms.RawlsGCN import RawlsGCN
from algorithms.FairWalk import FairWalk
from algorithms.FairVGNN import FairVGNN
from algorithms.GEAR import GEAR
from algorithms.FairGNN import FairGNN
from algorithms.FairEdit import FairEdit
from algorithms.EDITS import EDITS
from algorithms.InFoRM_GNN import InFoRM_GNN
from algorithms.GUIDE import GUIDE
from algorithms.REDRESS import REDRESS
from algorithms.NIFTY import NIFTY
from algorithms.NIFTY_cf import NIFTY_cf
from algorithms.GNN import GNN
from algorithms.GNN_individual import GNN_individual
from algorithms.GNN_cf import GNN_cf

import json
import time
import numpy as np
from collections import defaultdict
import torch
import random
import argparse

name = ['ACC', 'AUCROC', 'F1', 'ACC_sens0', 'AUCROC_sens0', 'F1_sens0', 'ACC_sens1', 'AUCROC_sens1', 'F1_sens1', 'SP',
        'EO']


parser = argparse.ArgumentParser(description='')
parser.add_argument('param1', type=float, help='First scalar parameter')
parser.add_argument('param2', type=float, help='Second scalar parameter', default=0.0)
parser.add_argument('use_optimal', type=bool, help='Whether to use optimal parameters', default=False)


args = parser.parse_args()



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


setup_seed(11)

model_name='GNN'
dataset='german'
save_dict = defaultdict(dict)
save_dict['all'] = defaultdict(list)

if args.use_optimal:
    optimal_params= json.load(open('./params.json'))
    args.param1=list(optimal_params[model_name][dataset].values())[0]
    args.param2=list(optimal_params[model_name][dataset].values())[1]

if model_name == 'DeepWalk':
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=True if dataset == 'nba' or dataset == 'german' else False)
    val_loss = 1e10

    num_walks=args.param1
    walk_length=args.param2


    model = CrossWalk()
    time_start = time.time()
    model.fit(adj, feats, labels, idx_train, sens, number_walks=num_walks,
              walk_length=walk_length, window_size=5, deep_walk=True)
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict(
        idx_test)

    temp = {name_one: value_one for name_one, value_one in
            zip(name, [ACC, AUCROC, F1, ACC_sens0,
                       AUCROC_sens0, F1_sens0,
                       ACC_sens1, AUCROC_sens1,
                       F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'num_walks': num_walks, 'walk_length': walk_length}

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val(idx_val)

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})



    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)

    for i, value in enumerate(
            [ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1,
             F1_sens1, SP, EO]):
        print(name[i], '{:.6f}'.format(value))


    json.dump(save_dict, open('./final_save_dict_{}.json'.format(model_name), 'w'))

elif model_name == 'CrossWalk':
    print(dataset)
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=True if dataset == 'nba' or dataset == 'german' else False)
    val_loss = 1e10

    num_walks=args.param1
    walk_length=args.param2


    window_size=5
    model = CrossWalk()
    time_start = time.time()
    model.fit(adj, feats, labels, idx_train, sens, number_walks=num_walks,
              walk_length=walk_length, window_size=window_size)
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict(
        idx_test)

    temp = {name_one: value_one for name_one, value_one in
            zip(name, [ACC, AUCROC, F1, ACC_sens0,
                       AUCROC_sens0, F1_sens0,
                       ACC_sens1, AUCROC_sens1,
                       F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'num_walks': num_walks, 'walk_length': walk_length}

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val(idx_val)

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)

    for i, value in enumerate(
            [ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1,
             F1_sens1, SP, EO]):
        print(name[i], '{:.6f}'.format(value))




elif model_name == 'FairWalk':
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset)
    val_loss = 1e10

    num_walks=args.param1
    walk_length=args.param2


    time_start = time.time()
    model = FairWalk()
    model.fit(adj, labels, idx_train, sens, num_walks=num_walks, walk_length=walk_length)

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict(
        idx_test, idx_val)

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'num_walks': num_walks, 'walk_length': walk_length}
    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)

    for i, value in enumerate(
            [ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP,
             EO]):
        print(name[i], '{:.6f}'.format(value))



elif model_name == 'FairVGNN':
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=True if dataset == 'german' else False)

    top_k=args.param1
    alpha=args.param2

    val_loss = 1e10

    time_start = time.time()

    model = FairVGNN()

    if dataset == 'recidivism':
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, top_k=top_k,
                  alpha=alpha, clip_e=1, g_epochs=10, c_epochs=10, ratio=1, epochs=300)
    elif dataset == 'german':
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, top_k=top_k,
                  clip_e=0.1, d_epochs=5, c_epochs=10, c_lr=0.01, e_lr=0.001, ratio=0,
                  alpha=alpha, prop='scatter', epochs=600)
    elif dataset == 'credit':
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, top_k=top_k,
                  alpha=alpha, clip_e=1, g_epochs=10, c_epochs=5, ratio=0, e_lr=0.01, epochs=200)

    else:
        model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, top_k=top_k,
                  alpha=alpha)

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict()

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'top_k': top_k, 'alpha': alpha}


    SP, EO = model.predict_val()

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(['SP','EO'], [SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)

    for i, value in enumerate(
            [ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1,
             SP, EO]):
        print(name[i], '{:.6f}'.format(value))


        # continue

elif model_name == 'FairGNN':
    temp_accs = {'german': 0.66, 'recidivism': 0.84, 'credit': 0.60, 'pokec_z': 0.6, 'pokec_n': 0.56,
                 'nba': 0.56, 'AMiner-S': 0.8, 'AMiner-L': 0.88, 'facebook': 0.67}

    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=True if dataset == 'nba' or dataset == 'german' else False)

    temp_acc = temp_accs[dataset] - 0.3
    val_loss = 1e10

    alpha=args.param1
    beta=args.param2


    time_start = time.time()
    model = FairGNN(feats.shape[-1], acc=temp_acc, epoch=500, alpha=alpha, beta=beta).cuda()
    model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, idx_train)
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict(
        idx_test)

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'alpha': alpha, 'beta': beta}

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val(idx_val)

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)

    for i, value in enumerate(
            [ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1,
             SP, EO]):
        print(name[i], '{:.6f}'.format(value))

elif model_name == 'FairEdit':
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=True if dataset == 'german' else False)
    val_loss = 1e10

    weight_decay=args.param1
    hidden=args.param2


    time_start = time.time()
    model = FairEdit()
    model.fit(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
              weight_decay=weight_decay, hidden=hidden, dropout=0.2)
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict()

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'weight_decay': weight_decay, 'hidden': hidden}

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val()

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)

    for i, value in enumerate(
            [ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1,
             SP, EO]):
        print(name[i], '{:.6f}'.format(value))



elif model_name == 'EDITS':


    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=False)

    if dataset == 'credit' or dataset == 'german':
        feats = feats / feats.norm(dim=0)


    dropout = args.param1
    threshold_proportions = args.param2

    val_loss = 1e10

    time_start = time.time()

    model = EDITS(feats, dropout=dropout)
    model.fit(adj, feats, sens, idx_train, idx_val, epochs=100 if dataset == 'german' else 500)
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict(
        adj, labels, sens, idx_train, idx_val, idx_test, epochs=100 if dataset == 'german' else 500,
        threshold_proportion=threshold_proportion)

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'dropout': dropout, 'threshold_proportion': threshold_proportion}

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val()

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)



elif model_name == 'NIFTY':

    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=False)
    val_loss = 1e10

    num_hidden = args.param1
    num_proj_hidden = args.param2


    time_start = time.time()

    model = NIFTY(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx,
                  num_hidden=num_hidden, num_proj_hidden=num_proj_hidden)
    model.fit()
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict()

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'num_hidden': num_hidden, 'num_proj_hidden': num_proj_hidden}
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val()

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)




elif model_name == 'GNN':
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=False)
    print(labels)
    val_loss = 1e10

    num_hidden = args.param1
    num_proj_hidden = args.param2

    time_start = time.time()

    model = GNN(adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx, num_hidden=num_hidden,
                num_proj_hidden=num_proj_hidden)
    model.fit()
    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, SP, EO = model.predict()

    temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                      AUCROC_sens0, F1_sens0,
                                                                      ACC_sens1, AUCROC_sens1,
                                                                      F1_sens1, SP, EO])}
    temp['val_loss'] = model.val_loss
    temp['time'] = time.time() - time_start
    temp['parameter'] = {'num_hidden': num_hidden, 'num_proj_hidden': num_proj_hidden}

    ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
    SP, EO, = model.predict_val()

    temp.update({'val_' + name_one: value_one for name_one, value_one in
                 zip(name, [ACC, AUCROC, F1, ACC_sens0,
                            AUCROC_sens0, F1_sens0,
                            ACC_sens1,
                            AUCROC_sens1, F1_sens1,
                            SP, EO])})

    if model.val_loss < val_loss:
        val_loss = model.val_loss
        save_dict['best'][model_name + '+' + dataset] = temp
    save_dict['all'][model_name + '+' + dataset].append(temp)


elif model_name in ['InFoRM', 'REDRESS', 'GUIDE', 'GNN_individual']:
    name = ['ACC', 'AUCROC', 'F1', 'ACC_sens0', 'AUCROC_sens0', 'F1_sens0', 'ACC_sens1', 'AUCROC_sens1',
            'F1_sens1', 'SP', 'EO', 'IF', 'GDIF', 'ndcg_value']
    val_loss = 1e10
    adj, feats, labels, idx_train, idx_val, idx_test, sens, sens_idx = load_data(dataset,
                                                                                 feature_normalize=True)

    lr=args.param2

    if model_name in ['InFoRM', 'GUIDE']:
        alpha=args.param1

        time_start = time.time()
        if model_name == 'InFoRM':
            model = InFoRM_GNN(dataset, adj, feats, idx_train, idx_val, idx_test, labels, sens, lr=lr)
            model.fit(alpha=alpha, epochs=250)
        elif model_name == 'GUIDE':
            model = GUIDE()
            model.fit(dataset, adj, feats, idx_train, idx_val, idx_test, labels, sens, alpha=alpha,
                      lr=lr, initialize_training_epochs=200, epochs=200)

        ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
        SP, EO, IF, GDIF, ndcg_value = model.predict()

        temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                          AUCROC_sens0, F1_sens0,
                                                                          ACC_sens1,
                                                                          AUCROC_sens1, F1_sens1,
                                                                          SP, EO, IF, GDIF,
                                                                          ndcg_value])}
        temp['parameter'] = {'alpha': alpha, 'lr': lr}
        temp['val_loss'] = model.val_loss
        temp['time'] = time.time() - time_start

        ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
        SP, EO, IF, GDIF, ndcg_value = model.predict_val()

        temp.update({'val_' + name_one: value_one for name_one, value_one in
                     zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                AUCROC_sens0, F1_sens0,
                                ACC_sens1,
                                AUCROC_sens1, F1_sens1,
                                SP, EO, IF, GDIF,
                                ndcg_value])})

        if model.val_loss < val_loss:
            val_loss = model.val_loss
            save_dict['best'][model_name + '+' + dataset] = temp
        save_dict['all'][model_name + '+' + dataset].append(temp)
    else:
        num_hidden=args.param1
        time_start = time.time()
        if model_name == 'REDRESS':
            model = REDRESS(dataset, adj, feats, labels, sens, idx_train, idx_val, idx_test,
                            hidden=num_hidden, lr=lr, epochs=25)
            model.fit()
        elif model_name == 'GNN_individual':
            model = GNN_individual(dataset, adj, feats, labels, idx_train, idx_val, idx_test, sens,
                                   sens_idx, lr=lr, num_hidden=num_hidden)  # 0.005
            model.fit()

        ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
        SP, EO, IF, GDIF, ndcg_value = model.predict()

        temp = {name_one: value_one for name_one, value_one in zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                                                          AUCROC_sens0, F1_sens0,
                                                                          ACC_sens1,
                                                                          AUCROC_sens1, F1_sens1,
                                                                          SP, EO, IF, GDIF,
                                                                          ndcg_value])}

        temp['parameter'] = {'num_hidden': num_hidden, 'lr': lr}
        temp['val_loss'] = model.val_loss
        temp['time'] = time.time() - time_start

        ACC, AUCROC, F1, ACC_sens0, AUCROC_sens0, F1_sens0, ACC_sens1, AUCROC_sens1, F1_sens1, \
        SP, EO, IF, GDIF, ndcg_value = model.predict_val()

        temp.update({'val_' + name_one: value_one for name_one, value_one in
                     zip(name, [ACC, AUCROC, F1, ACC_sens0,
                                AUCROC_sens0, F1_sens0,
                                ACC_sens1,
                                AUCROC_sens1, F1_sens1,
                                SP, EO, IF, GDIF,
                                ndcg_value])})

        if model.val_loss < val_loss:
            val_loss = model.val_loss
            save_dict['best'][model_name + '+' + dataset] = temp
        save_dict['all'][model_name + '+' + dataset].append(temp)


json.dump(save_dict, open('./save_dict_{}_avg.json'.format(model_name), 'w'))