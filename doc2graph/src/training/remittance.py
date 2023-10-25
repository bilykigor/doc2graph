import os
from datetime import datetime
from sklearn.model_selection import KFold, ShuffleSplit
import torch
from torch.nn import functional as F
from random import shuffle, seed
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import dgl 
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import time
from statistics import mean
import numpy as np
from PIL import Image
from sklearn.utils import class_weight


from doc2graph.src.data.dataloader import Document2Graph
from doc2graph.src.paths import *
from doc2graph.src.models.graphs import SetModel
from doc2graph.src.utils import get_config
from doc2graph.src.data.preprocessing import unnormalize_box, draw_boxes
from doc2graph.src.training.utils import *
from doc2graph.src.data.graph_builder import GraphBuilder
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch_geometric.loader import DataLoader as GDataLoader


def e2e(args):

    # configs
    start_training = time.time()
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='REMITTANCE TRAIN', src_path=REMITTANCE_TRAIN, device = device, output_dir=TRAIN_SAMPLES)
        data.get_info()
        val_data = Document2Graph(name='REMITTANCE VALIDATION', src_path=REMITTANCE_VAL, device = device, output_dir=VALIDATION_SAMPLES)
        val_data.get_info()
        val_loader = DataLoader(range(len(val_data)), sampler=SequentialSampler(range(len(val_data))), batch_size=cfg_train.batch_size,num_workers=0)
        ################* STEP 1: CREATE MODEL ################
        model = sm.get_model(data.node_num_classes, data.edge_num_classes, data.get_chunks())
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=400, min_lr=1e-3, verbose=True, factor=0.01)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.8)
        e = datetime.now()
        train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
        stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=100)
        writer = SummaryWriter(log_dir=RUNS)
        #convert_imgs = transforms.ToTensor()
            
        ################* STEP 2: TRAINING ################
        print("\n### TRAINING ###")
        print(f"-> Training samples: {len(data)}")
        print(f"-> Validation samples: {len(val_data)}\n")
            
        # TRAIN
        for epoch in range(cfg_train.epochs):
            # TRAINING

            total_train_loss = 0
            total_train_macro = 0
            total_train_micro = 0
            total_train_auc = 0
            
            model.train()

            batch_size = cfg_train.batch_size+int(epoch/cfg_train.epoch_size_step)*cfg_train.batch_size_step
            batch_size = min([batch_size,cfg_train.max_batch_size])
            train_loader = DataLoader(range(len(data)), sampler=RandomSampler(range(len(data))), batch_size = batch_size, num_workers = 0)
            for train_index in train_loader:
                
                train_graphs = [data.graphs[i] for i in train_index]
                tg = dgl.batch(train_graphs)
                tg = tg.int().to(device)
        
                n_scores, e_scores = model(tg, tg.ndata['feat'].to(device))
                n_loss = compute_crossentropy_loss(n_scores.to(device), tg.ndata['label'].to(device),device)
                e_loss = compute_crossentropy_loss(e_scores.to(device), tg.edata['label'].to(device),device)
                tot_loss = n_loss + e_loss
                
                l2_lambda = 0.001
                l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = tot_loss #+ l2_lambda * l2_norm
                
                macro, micro = get_f1(n_scores, tg.ndata['label'].to(device))
                auc = compute_auc_mc(e_scores.to(device), tg.edata['label'].to(device))
                
                total_train_loss += tot_loss
                total_train_macro += macro
                total_train_micro += micro
                total_train_auc += auc
                
                # Backward and optimize
                # ======================================================================
                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

                optimizer.zero_grad()
                model.zero_grad() 
                loss.backward()    
                    
                optimizer.step()

                # ======================================================================
                # if cfg_train.stopper_metric == 'loss_diff':
                #     step_value = 0-tot_loss.item()
                # elif cfg_train.stopper_metric == 'loss':
                #     step_value = tot_loss.item()
                # elif cfg_train.stopper_metric == 'acc':
                #     step_value = 0
                
            avg_train_loss = total_train_loss / len(train_loader)
            avg_train_micro = total_train_micro / len(train_loader)
            avg_train_macro = total_train_macro / len(train_loader)    
            avg_train_auc = total_train_auc / len(train_loader)       
             
            ### VALIDATION    
            total_val_loss = 0
            total_val_macro = 0
            total_val_micro = 0
            total_val_auc = 0
            
            model.eval()
            
            for val_index in val_loader:
                
                val_graphs = [data.graphs[i] for i in val_index]
                vg = dgl.batch(val_graphs)
                vg = vg.int().to(device)
        
                with torch.no_grad():
                    val_n_scores, val_e_scores = model(vg, vg.ndata['feat'].to(device))
                    val_n_loss = compute_crossentropy_loss(val_n_scores.to(device), vg.ndata['label'].to(device),device)
                    val_e_loss = compute_crossentropy_loss(val_e_scores.to(device), vg.edata['label'].to(device),device)
                    val_tot_loss = val_n_loss + val_e_loss
                    val_macro, val_micro = get_f1(val_n_scores, vg.ndata['label'].to(device))
                    val_auc = compute_auc_mc(val_e_scores.to(device), vg.edata['label'].to(device))
                    
                total_val_loss += val_tot_loss
                total_val_macro += val_macro
                total_val_micro += val_micro
                total_val_auc += val_auc
                
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_micro = total_val_macro / len(val_loader)
            avg_val_macro = total_val_micro / len(val_loader)    
            avg_val_auc = total_val_auc / len(val_loader)  
                
            if cfg_train.verbose:
                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | TrainAUC-PR {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} | ValAUC-PR {:.4f} |"
                .format(epoch, avg_train_loss, avg_train_macro, avg_train_auc, avg_val_loss, avg_val_macro, avg_val_auc))
                
            writer.add_scalars('AUC-PR', {'train': avg_train_auc, 'val': avg_val_auc}, epoch)
            writer.add_scalars('LOSS', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
            writer.add_scalars('LOSS_DIFF', {'value': avg_val_loss - avg_train_loss}, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
                
            ss = stopper.step(avg_val_loss)

            if ss == 'stop':
                break
            
            scheduler.step()

    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        models = args.weights
    
    ################* STEP 3: TESTING ################
    print("\n### TESTING ###")
    

    test_data = Document2Graph(name='REMITTANCE TEST', src_path=REMITTANCE_TEST, device = device, output_dir=TEST_SAMPLES)
    test_data.get_info()
    
    model = sm.get_model(test_data.node_num_classes, test_data.edge_num_classes, test_data.get_chunks())
    nodes_micro = []
    #edges_f1 = []
    
    test_graph = dgl.batch(test_data.graphs).to(device)

    m = train_name+'.pt'
    best_model = m
    model.load_state_dict(torch.load(CHECKPOINTS / m))
    model.eval()
    with torch.no_grad():
        n, e = model(test_graph, test_graph.ndata['feat'].to(device))
        # auc = compute_auc_mc(e.to(device), test_graph.edata['label'].to(device))
        #_, preds = torch.max(F.softmax(e, dim=1), dim=1)
        _, npreds_all = torch.max(F.softmax(n, dim=1), dim=1)

        #accuracy, f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'])
        #_, classes_f1 = get_binary_accuracy_and_f1(preds, test_graph.edata['label'], per_class=True)
        #edges_f1.append(classes_f1[1])

        macro, micro = get_f1(n, test_graph.ndata['label'].to(device))
        try:
            f1_per_class = get_f1(n, test_graph.ndata['label'].to(device),per_class=True)
            f1_per_class = {test_data.node_unique_labels[i]:f1_per_class[i] for i in range(len(test_data.node_unique_labels))}
            print(f'f1_per_class: {f1_per_class}')
        except Exception as exception:
            print(f'Error: {exception}')
        nodes_micro.append(micro)
        
        #test_graph.edata['preds'] = preds

    ################* STEP 3.5: VISUALIZATION ##########
    visualize(npreds_all,test_data)
        
    ################* STEP 4: RESULTS ################
    print("\n### RESULTS {} ###".format(m))
    # print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
    # print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    # print(f"\n -> Loading best model {best_model}")
    # model.load_state_dict(torch.load(CHECKPOINTS / best_model))
    # model.eval()
    # with torch.no_grad():

    #     n, e = model(test_graph, test_graph.ndata['feat'].to(device))
    #     auc = compute_auc_mc(e.to(device), test_graph.edata['label'].to(device))
        
    #     _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
    #     _, npreds = torch.max(F.softmax(n, dim=1), dim=1)
    #     test_graph.edata['preds'] = epreds
    #     test_graph.ndata['preds'] = npreds
    #     test_graph.ndata['net'] = n

    #     accuracy, f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'])
    #     _, classes_f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'], per_class=True)
    #     macro, micro = get_f1(n, test_graph.ndata['label'].to(device))

    # ################* STEP 4: RESULTS ################
    #print("\n### BEST RESULTS ###")
    #print("AUC {:.4f}".format(auc))
    #print("Accuracy {:.4f}".format(accuracy))
    #print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
    #print("F1 Edges: None {:.4f} - Pairs {:.4f}".format(classes_f1[0], classes_f1[1]))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    #print("\n### AVG RESULTS ###")
    #print("Semantic Entity Labeling: MEAN ", mean(nodes_micro), " STD: ", np.std(nodes_micro))
    #print("Entity Linking: MEAN ", mean(edges_f1),"STD", np.std(edges_f1))

    if not args.test:
        feat_n, feat_e = get_features(args)
        #? if skipping training, no need to save anything
        model = get_config(CFGM / args.model)
        results = {'MODEL': {
            'name': sm.get_name(),
            'weights': best_model,
            'net-params': sm.get_total_params(), 
            'num-layers': model.num_layers,
            'projector-output': model.hidden_dim,
            'dropout': model.dropout,
            'lastFC': model.hidden_dim
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': {
                'start-lr': cfg_train.lr,
                'weight-decay': cfg_train.weight_decay,
                'seed': cfg_train.seed
            },
            'RESULTS': {
                'val-loss': stopper.best_score.cpu().detach().numpy().tolist(), 
                #'f1-scores': f1,
		        # 'f1-classes': classes_f1,
                'nodes-f1': [macro, micro],
                # 'std-pairs': np.std(edges_f1),
                # 'mean-pairs': mean(edges_f1)
            }}
        
        print(results)
        try:
            save_test_results(train_name, results)
        except Exception as exception:
            print(f'Error: {exception}')
    
        print("END TRAINING:", time.time() - start_training)
    return {}#{'LINKS [MAX, MEAN, STD]': [classes_f1[1], mean(edges_f1), np.std(edges_f1)], 'NODES [MAX, MEAN, STD]': [micro, mean(nodes_micro), np.std(nodes_micro)]}

def gat(args):

    # configs
    start_training = time.time()
    cfg_train = get_config('train')
    seed(cfg_train.seed)
    device = get_device(args.gpu)
    sm = SetModel(name=args.model, device=device)

    if not args.test:
        ################* STEP 0: LOAD DATA ################
        data = Document2Graph(name='REMITTANCE TRAIN', src_path=REMITTANCE_TRAIN, device = device, output_dir=TRAIN_SAMPLES)
        data.get_info()
        labels = sum([list(g.y.cpu().numpy()) for g in data], [])
        w = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(labels), y=labels)
        weights = torch.tensor(w, dtype=torch.float32).to(device)
        
        print('weights:', weights)
        
        val_data = Document2Graph(name='REMITTANCE VALIDATION', src_path=REMITTANCE_VAL, device = device, output_dir=TEST_SAMPLES)
        val_data.get_info()
        val_loader = GDataLoader(val_data, batch_size=cfg_train.batch_size, shuffle=False)
        ################* STEP 1: CREATE MODEL ################
        model = sm.get_model(data.get_nclasses(), data.get_efeatures_size(), data.get_chunks())
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_train.lr), weight_decay=float(cfg_train.weight_decay))
        # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=400, min_lr=1e-3, verbose=True, factor=0.01)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.8)
        e = datetime.now()
        train_name = args.model + f'-{e.strftime("%Y%m%d-%H%M")}'
        stopper = EarlyStopping(model, name=train_name, metric=cfg_train.stopper_metric, patience=100)
        writer = SummaryWriter(log_dir=RUNS)
        #convert_imgs = transforms.ToTensor()
            
        ################* STEP 2: TRAINING ################
        print("\n### TRAINING ###")
        print(f"-> Training samples: {len(data)}")
        print(f"-> Validation samples: {len(val_data)}\n")
            
        # TRAIN
        for epoch in range(cfg_train.epochs):
            # TRAINING

            total_train_loss = 0
            total_train_macro = 0
            total_train_micro = 0
            
            model.train()

            batch_size = cfg_train.batch_size+int(epoch/cfg_train.epoch_size_step)*cfg_train.batch_size_step
            batch_size = min([batch_size,cfg_train.max_batch_size])
            train_loader = GDataLoader(data, batch_size = batch_size, shuffle = True)
            for tg in train_loader:
                tg = tg.to(device)
                n_scores = model(tg)
                
                loss = compute_crossentropy_loss(n_scores, tg.y, weights, device)
                tot_loss = loss
                
                macro, micro = get_f1(n_scores, tg.y)
                
                total_train_loss += tot_loss
                total_train_macro += macro
                total_train_micro += micro
                
                # Backward and optimize
                # ======================================================================
                # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

                optimizer.zero_grad()
                model.zero_grad() 
                loss.backward()    
                    
                optimizer.step()

                # ======================================================================
                # if cfg_train.stopper_metric == 'loss_diff':
                #     step_value = 0-tot_loss.item()
                # elif cfg_train.stopper_metric == 'loss':
                #     step_value = tot_loss.item()
                # elif cfg_train.stopper_metric == 'acc':
                #     step_value = 0
                
            avg_train_loss = total_train_loss / len(train_loader)
            avg_train_micro = total_train_micro / len(train_loader)
            avg_train_macro = total_train_macro / len(train_loader)    
             
            ### VALIDATION    
            total_val_loss = 0
            total_val_macro = 0
            total_val_micro = 0
            
            model.eval()
            
            for vg in val_loader:
                
                vg = vg.to(device)
        
                with torch.no_grad():
                    val_n_scores = model(vg)
                    val_n_loss = compute_crossentropy_loss(val_n_scores, vg.y, weights, device)
                    val_tot_loss = val_n_loss 
                    val_macro, val_micro = get_f1(val_n_scores, vg.y)
                    
                total_val_loss += val_tot_loss
                total_val_macro += val_macro
                total_val_micro += val_micro
                
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_micro = total_val_macro / len(val_loader)
            avg_val_macro = total_val_micro / len(val_loader)    
                
            if cfg_train.verbose:
                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO {:.4f} | ValLoss {:.4f} | ValF1-MACRO {:.4f} |"
                .format(epoch, avg_train_loss, avg_train_macro, avg_val_loss, avg_val_macro))
                
            writer.add_scalars('LOSS', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
            writer.add_scalars('F1_macro', {'train': avg_train_macro, 'val': avg_val_micro}, epoch)
            writer.add_scalars('LOSS_DIFF', {'value': avg_val_loss - avg_train_loss}, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
                
            ss = stopper.step(avg_val_loss)

            if ss == 'stop':
                break
            
            scheduler.step()

    else:
        ################* SKIP TRAINING ################
        print("\n### SKIP TRAINING ###")
        print(f"-> loading {args.weights}")
        models = args.weights
    
    ################* STEP 3: TESTING ################
    print("------------------------------------------------------------------------")
    print("\n### TESTING ###")

    test_data = Document2Graph(name='REMITTANCE TEST', src_path=REMITTANCE_TEST, device = device, output_dir=TEST_SAMPLES)
    test_data.get_info()
    # for g in test_data:
    #     geom = g['geom']
    #     text = g['text']
    #     for i in range(geom.shape[0]):
    #         print(list(geom[i].numpy()), list(g.x[i].numpy()), text[i], g.y[i].numpy())
    #     break
    
    m = train_name+'.pt'
    model = sm.get_model(test_data.get_nclasses(), test_data.get_efeatures_size(), test_data.get_chunks())
    model.load_state_dict(torch.load(CHECKPOINTS / m))
    model.eval()
    
    npreds_all, scores, f1_test = predict(model,test_data)
    visualize(npreds_all,scores, test_data)
    
    ################* STEP 3.5: INFERENCE ##########
    print("------------------------------------------------------------------------")
    print("\n### INFERENCE ###")

    inf_data = Document2Graph(name='REMITTANCE INFERENCE', src_path=REMITTANCE_INFERENCE, device = device, output_dir=INFERENCE_SAMPLES)
    inf_data.get_info()
    # for g in inf_data:
    #     geom = g['geom']
    #     text = g['text']
    #     for i in range(geom.shape[0]):
    #         print(list(geom[i].numpy()), list(g.x[i].numpy()), text[i], g.y[i].numpy() )
    #     break
    
    npreds_all, scores, f1_inference = predict(model,inf_data)
    visualize(npreds_all,scores, inf_data)
    
    print("------------------------------------------------------------------------")
    print(f'f1_per_class: {f1_test}', np.mean([f1_test[k] for k in f1_test if k!='O']))
    print(f'f1_per_class: {f1_inference}', np.mean([f1_inference[k] for k in f1_inference if k!='O']))
        
    ################* STEP 4: RESULTS ################
    print("------------------------------------------------------------------------")
    print("\n### RESULTS {} ###".format(m))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    if not args.test:
        feat_n, feat_e = get_features(args)
        #? if skipping training, no need to save anything
        model_cfg = get_config(CFGM / args.model)
        results = {'MODEL': {
            'name': sm.get_name(),
            'weights': m,
            'net-params': sm.get_total_params(),
            'nclasses': model.nclasses,
            'num_edge_features': model.num_edge_features,
            'in_chunks': model.in_chunks,
            'config': model_cfg
            },
            'FEATURES': {
                'nodes': feat_n, 
                'edges': feat_e
            },
            'PARAMS': cfg_train,
            'RESULTS': {
                'val-loss': stopper.best_score.cpu().detach().numpy().tolist(), 
                #'f1-scores': f1,
		        # 'f1-classes': classes_f1,
                'nodes-f1': [macro, micro],
                # 'std-pairs': np.std(edges_f1),
                # 'mean-pairs': mean(edges_f1)
            }}
        
        print(results)
        try:
            save_test_results(train_name, results)
        except Exception as exception:
            print(f'Error: {exception}')
    
        print("END TRAINING:", time.time() - start_training)
    return {}#{'LINKS [MAX, MEAN, STD]': [classes_f1[1], mean(edges_f1), np.std(edges_f1)], 'NODES [MAX, MEAN, STD]': [micro, mean(nodes_micro), np.std(nodes_micro)]}

def train_remittance(args):

    if args.model == 'e2e':
        e2e(args)
    elif args.model == 'gat':
        gat(args)
    else:
        raise Exception("Model selected does not exists. Choose 'e2e' or 'edge'.")
    return

def predict(model,data):
    test_loader = GDataLoader(data, batch_size=len(data), shuffle=False)
    
    for test_graph in test_loader:
        break

    with torch.no_grad():
        n = model(test_graph)
        scores, npreds_all = torch.max(F.softmax(n, dim=1), dim=1)

        macro, micro = get_f1(n, test_graph.y)
        f1_per_class = get_f1(n, test_graph.y, per_class=True)
        f1_per_class = {data.node_unique_labels[i]:f1_per_class[i] for i in range(len(data.node_unique_labels))}
        #print(f'f1_per_class: {f1_per_class}')
        
    return npreds_all, scores, f1_per_class
    
def visualize(npreds_all, scores_all, data):
    start_ind=0
    npreds_all = npreds_all.cpu().detach()
    for ind in range(len(data)):
        graph = data.graphs[ind]
        n_nodes = graph.ndata['feat'].shape[0]
        u, v = graph.edges()
        
        img_path = data.paths[ind]
        img_name = os.path.basename(img_path)
        #print(img_name)
        
        inference = Image.open(img_path).convert('RGB')
        size = inference.size
        
        npreds = npreds_all[start_ind:start_ind+n_nodes]
        scores = scores_all[start_ind:start_ind+n_nodes]
        start_ind+=n_nodes
        arr = npreds.numpy()
        
        li = list(np.where(arr>0)[0]) 
        sc = [scores[i] for i in li]
        
        labels = [data.node_unique_labels[arr[i]] for i in li]
        
        boxes = list(graph.ndata['geom'].numpy())
        boxes = [unnormalize_box(box, size[0], size[1]) for box in boxes]
        entities =  [boxes[i] for i in li]
        links = {'src': u, 'dst': v}
        
        inference = draw_boxes(inference, boxes, entities, labels=labels, links = links, scores=sc)
        inference.save(data.output_dir / img_name)
        #print(data.output_dir / img_name)
