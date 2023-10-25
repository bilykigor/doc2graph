import argparse

from doc2graph.src.data.download import get_data
from doc2graph.src.inference import inference
from doc2graph.src.training.funsd import train_funsd
from doc2graph.src.training.remittance import train_remittance
from doc2graph.src.utils import create_folder, project_tree, set_preprocessing
from doc2graph.src.training.pau import train_pau

import nltk
nltk.download('punkt')

def seed_everything(seed=10):
    import os
    import random
    import numpy as np
    import torch
    import pytorch_lightning as pl

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #os.environ['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'] = "True"
    #pl.seed_everything(seed)
    #torch.set_default_tensor_type('torch.DoubleTensor')
    
    
def main():
    seed_everything(42)
    
    parser = argparse.ArgumentParser(description='Training')

    # init
    parser.add_argument('--init', action="store_true",
                        help="download data and prepare folders")
    
    # features
    parser.add_argument('--add-geom', '-addG', action="store_true",
                        help="add geometrical features to nodes")
    parser.add_argument('--add-size', '-addS', action="store_true",
                        help="add boxs size features to nodes")
    parser.add_argument('--add-epolar', '-addP', action="store_true",
                        help="add polar relations of nodes")
    parser.add_argument('--add-embs', '-addT', action="store_true",
                        help="add textual embeddings to nodes")
    parser.add_argument('--add-mask', '-addM', action="store_true",
                        help="add textual mask to nodes")
    parser.add_argument('--add-hist', '-addH', action="store_true",
                        help="add histogram of contents to nodes")
    parser.add_argument('--add-visual', '-addV', action="store_true",
                        help="add visual features to nodes")
    parser.add_argument('--add-edist', '-addED', action="store_true",
                        help="add edge distances to graphs")
    parser.add_argument('--add-eshared', '-addES', action="store_true",
                        help="add intersection of boxes to graphs")
    # data
    parser.add_argument("--src-data", type=str, default='FUNSD',
                        help="which data source to use. It can be FUNSD, PAU or CUSTOM")
    parser.add_argument("--data-type", type=str, default='img',
                        help="if src-data is CUSTOM, define the data source type: img or pdf.")
    # graphs
    parser.add_argument("--edge-type", type=str, default='fully',
                        help="choose the kind of connectivity in the graph. It can be: fully or knn.")
    parser.add_argument("--node-granularity", type=str, default='gt',
                        help="choose the granularity of nodes to be used. It can be: gt (if given), ocr (words) or yolo (entities).")
    parser.add_argument("--num-polar-bins", type=int, default=8,
                        help="number of bins into which discretize the space for edge polar features. It must be a power of 2: Default 8.")

    # training
    parser.add_argument("--model", type=str, default='gat',
                        help="which model to use, which yaml file to load: e2e, edge or gcn")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument('--test', action="store_true",
                        help="skip training")
    parser.add_argument('--weights', '-w', nargs='+', type=str, default=None,
                        help="provide a weights file relative path if testing")
    
    # inference
    parser.add_argument('--inference', action="store_true",
                        help="use the model to predict on new, unseen examples")
    parser.add_argument('--docs', nargs='+', type=str, default=None,
                        help="provide documents to do inference on them")
         
    args = parser.parse_args()
    print(args)

    if args.init:
        project_tree()
        get_data()
        print("Initialization completed!")

    else:
        set_preprocessing(args)
        if args.inference:
            create_folder('inference')
            inference(args.weights, args.docs, args.gpu)
        elif args.src_data == 'FUNSD':
            if args.test and args.weights == None:
                raise Exception("Main exception: Provide a weights file relative path! Or train a model first.")
            train_funsd(args)
        elif args.src_data == 'REMITTANCE':
            if args.test and args.weights == None:
                raise Exception("Main exception: Provide a weights file relative path! Or train a model first.")
            train_remittance(args)
        elif args.src_data == 'PAU':
            train_pau(args)
        elif args.src_data == 'CUSTOM':
            #TODO develop custom data preprocessing
            raise Exception('Main exception: "CUSTOM" source data still under development')
        else:
            raise Exception('Main exception: source data invalid. Choose from ["FUNSD", "PAU", "CUSTOM"]')
    
    return

if __name__ == '__main__':
    main()
    