import random
from typing import Tuple
#import spacy
import torch
import torchvision
import numpy as np
from math import sqrt
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as tvF
from typing import List, Tuple,Union
import dgl

#from doc2graph.src.paths import CHECKPOINTS
from doc2graph.src.models.unet import Unet
from doc2graph.src.data.utils import file_to_images
from doc2graph.src.data.utils import polar, get_histogram, polar2, polar3, intersectoin_by_axis, find_dates, find_amounts, find_numbers, find_codes, find_word, find_words
from doc2graph.src.utils import get_config
from doc2graph.src.data.preprocessing import normalize_box


class FeatureBuilder():

    def __init__(self, 
                 add_embs = True, 
                 add_geom=False, 
                 add_epolar = False, 
                 add_size =  False, 
                 add_mask =  False, 
                 add_hist =  False, 
                 add_visual =  False, 
                 add_edist =  False, 
                 add_eshared =  False, 
                 add_fudge =  False, 
                 num_polar_bins = 8, 
                 d : int = 'cpu'):
        """FeatureBuilder constructor

        Args:
            d (int): device number, if any (cpu or cuda:n)
        """
        #self.cfg_preprocessing = get_config('preprocessing')
        self.device = d
        self.add_geom = add_geom #self.cfg_preprocessing.FEATURES.add_geom
        self.add_epolar = add_epolar #self.cfg_preprocessing.FEATURES.add_epolar
        self.add_size = add_size #self.cfg_preprocessing.FEATURES.add_size
        self.add_embs = add_embs #self.cfg_preprocessing.FEATURES.add_embs
        self.add_mask = add_mask #self.cfg_preprocessing.FEATURES.add_mask
        self.add_hist = add_hist #self.cfg_preprocessing.FEATURES.add_hist
        self.add_visual = add_visual #self.cfg_preprocessing.FEATURES.add_visual
        self.add_edist = add_edist #self.cfg_preprocessing.FEATURES.add_edist
        self.add_eshared = add_eshared #self.cfg_preprocessing.FEATURES.add_eshared
        self.add_fudge = add_fudge #self.cfg_preprocessing.FEATURES.add_fudge
        self.num_polar_bins = num_polar_bins #self.cfg_preprocessing.FEATURES.num_polar_bins

        if self.add_embs:
            #text_embedder = spacy.load('en_core_web_lg')
            #self.text_embedder_core = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.text_embedder = self.text_to_embedding
            
        self.mask_embedder = text_to_mask

        # if self.add_visual:
        #     self.visual_embedder = Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=1, classes=4)
        #     self.visual_embedder.load_state_dict(torch.load(CHECKPOINTS / 'backbone_unet.pth', map_location=torch.device(d))['weights'])
        #     self.visual_embedder = self.visual_embedder.encoder
        #     self.visual_embedder.to(d)
        
        self.sg = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
    
    def text_to_embedding(self, text):
        #if find_word(text) or find_words(text):
        return self.text_embedder_core.encode(text, convert_to_tensor=False)
        
        #return np.zeros(384, dtype=np.float32)
    
    def add_features(self, 
                     graphs : List, 
                     features : List):
        """ Add features to provided graphs

        Args:
            graphs (list) : list of DGLGraphs
            features (list) : list of features "sources", like text, positions and images
        
        Returns:
            chunks list and its lenght
        """

        for id, g in enumerate(tqdm(graphs, desc='adding features')):

            # positional features
            size = file_to_images(features['paths'][id])[0].size
            feats = [[] for _ in range(g.num_nodes())]
            efeats = [[] for _ in range(g.num_edges())]
            #geom = [self.sg(box, size) for box in features['boxs'][id]]
            #geom = [normalize_box(box, size[0], size[1]) for box in features['boxs'][id]]
            geom = features['boxs'][id]
            chunks = []
            
            box_height = [box[3]-box[1] for box in geom]
            # m = 1
            med_height = np.median(box_height)
            # m = sqrt((size[0]*size[0] + size[1]*size[1]))

            # 'geometrical' features
            if self.add_geom:
                # TODO add 2d encoding like "LayoutLM*"
                # [feats[idx].extend(self.sg(box, size)) for idx, box in enumerate(features['boxs'][id])]
                _ = [feats[idx].extend(box) for idx, box in enumerate(geom)]
                chunks.append(4)
                
            if self.add_size:
                _ = [feats[idx].extend([
                    (box[2]-box[0])/med_height,
                    (box[3]-box[1])/med_height
                    ]) for idx, box in enumerate(geom)]
                chunks.append(2)
            
            # HISTOGRAM OF TEXT
            if self.add_hist:
                _ = [feats[idx].extend(hist) for idx, hist in enumerate(get_histogram(features['texts'][id]))]
                chunks.append(4)
            
            # textual features
            if self.add_embs:
                words = features['texts'][id]
                for idx, word in enumerate(words):
                    if self.add_mask:
                        mask = self.mask_embedder(word)
                        if mask[0]:
                            word = '<D>'
                        elif mask[1]:
                            word = '<A>'
                        elif mask[2]:
                            word = '<N>'
                    sentence_emb = self.text_embedder(word)
                    feats[idx].extend(sentence_emb)
                    #print(idx,word,sentence_emb[:3])
                    
                #_ = [feats[idx].extend(self.text_embedder(features['texts'][id][idx])) for idx, _ in enumerate(feats)]
                chunks.append(len(self.text_embedder(features['texts'][id][0])))
                
            # if self.add_mask:
            #     _ = [feats[idx].extend(self.mask_embedder(features['texts'][id][idx])) for idx, _ in enumerate(feats)]
            #     chunks.append(len(self.mask_embedder(features['texts'][id][0])))
            
            # visual features
            # https://pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html?highlight=roi
            if self.add_visual:
                img = Image.open(features['paths'][id]).convert('L')
                width, height = img.size
                img = img.resize((min(width,1000), min(1000,height)))
                bboxs = [normalize_box(b, width, height) for b in features['boxs'][id]]
                width, height = img.size
                
                input_tensor = tvF.to_tensor(img).unsqueeze_(0)
                input_tensor = input_tensor.to(self.device)
                visual_emb = self.visual_embedder(input_tensor) # output [batch, channels, dim1, dim2]
                del input_tensor
                
                bboxs = [torch.Tensor(b) for b in bboxs]
                bboxs = [torch.stack(bboxs, dim=0).to(self.device)]
                h = [torchvision.ops.roi_align(input=ve, boxes=bboxs, spatial_scale=1/ min(height / ve.shape[2] , width / ve.shape[3]), output_size=1) for ve in visual_emb[1:]]
                h = torch.cat(h, dim=1)

                # VISUAL FEATURES (RESNET-IMAGENET)
                _ = [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
                chunks.append(len(torch.flatten(h[0]).tolist()))
        
            if self.add_epolar:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                sin1 = []
                cos1 = []
                sin2 = []
                cos2 = []

                for pair in zip(srcs, dsts):
                    s1, c1, s2, c2   = polar2(geom[pair[0]], geom[pair[1]])
                    sin1.append(s1)
                    cos1.append(c1)
                    sin2.append(s2)
                    cos2.append(c2)
                
                _ = [efeats[idx].extend([sin1[idx],cos1[idx],sin2[idx],cos2[idx]]) for idx, _ in enumerate(efeats)]

            if self.add_edist:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                distances1 = []
                distances2 = []

                for pair in zip(srcs, dsts):
                    d1, d2   = polar3(geom[pair[0]], geom[pair[1]])
                    distances1.append(d1/med_height)
                    distances2.append(d2/med_height)
                
                _ = [efeats[idx].extend([distances1[idx],distances2[idx]]) for idx, _ in enumerate(efeats)]
                
            if self.add_eshared:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                x_shared = []
                y_shared = []

                for pair in zip(srcs, dsts):
                    x = intersectoin_by_axis('x',geom[pair[0]], geom[pair[1]])
                    y = intersectoin_by_axis('y',geom[pair[0]], geom[pair[1]])
                    x_shared.append(x)
                    y_shared.append(y)
                
                _ = [efeats[idx].extend([x_shared[idx],y_shared[idx]]) for idx, _ in enumerate(efeats)]

            g.ndata['geom'] = torch.tensor(geom, dtype=torch.float32)
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
            g.edata['feat'] = torch.tensor(efeats,dtype=torch.float32)#.t()

            #=============================================================
            distances = ([0.0 for _ in range(g.number_of_edges())])
            m = 1
            distances = torch.tensor([(1-d/m) for d in distances], dtype=torch.float32)
            tresh_dist = torch.where(distances > 0.9, torch.full_like(distances, 0.1), torch.zeros_like(distances))
            g.edata['weights'] = tresh_dist

            norm = []
            num_nodes = len(features['boxs'][id]) - 1
            for n in range(num_nodes + 1):
                neigs = torch.count_nonzero(tresh_dist[n*num_nodes:(n+1)*num_nodes]).tolist()
                try: norm.append([1. / neigs])
                except: norm.append([1.])
            g.ndata['norm'] = torch.tensor(norm, dtype=torch.float32)

            #! DEBUG PURPOSES TO VISUALIZE RANDOM GRAPH IMAGE FROM DATASET
            # if False:
            #     if id == rand_id and self.add_edist:
            #         print("\n\n### EXAMPLE ###")

            #         img_path = features['paths'][id]
            #         img = Image.open(img_path).convert('RGB')
            #         draw = ImageDraw.Draw(img)

            #         center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
            #         select = [random.randint(0, len(srcs)) for _ in range(10)]
            #         for p, pair in enumerate(zip(srcs, dsts)):
            #             if p in select:
            #                 sc = center(features['boxs'][id][pair[0]])
            #                 ec = center(features['boxs'][id][pair[1]])
            #                 draw.line((sc, ec), fill='grey', width=3)
            #                 middle_point = ((sc[0] + ec[0])/2,(sc[1] + ec[1])/2)
            #                 draw.text(middle_point, str(angles[p]), fill='black')
            #                 draw.rectangle(features['boxs'][id][pair[0]], fill='red')
            #                 draw.rectangle(features['boxs'][id][pair[1]], fill='blue')
                    
            #         img.save(f'esempi/FUNSD/edges.png')

        return chunks, len(chunks)
    
    def get_info(self):
        print(f"-> textual feats: {self.add_embs}\n\
-> text masked: {self.add_mask}\n\
-> visual feats: {self.add_visual}\n\
-> geom feats: {self.add_geom}\n\
-> size feats: {self.add_size}\n\
-> edge epolar: {self.add_epolar}\n\
-> edge edist: {self.add_edist}")

    
