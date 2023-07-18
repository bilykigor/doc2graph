import random
from typing import Tuple
import spacy
import torch
import torchvision
from math import sqrt
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision.transforms.functional as tvF

from src.paths import CHECKPOINTS
from src.models.unet import Unet
from src.data.utils import to_bin, to_bin2
from src.data.utils import polar, get_histogram, polar2
from src.utils import get_config
from src.data.preprocessing import normalize_box

class FeatureBuilder():

    def __init__(self, d : int = 'cpu'):
        """FeatureBuilder constructor

        Args:
            d (int): device number, if any (cpu or cuda:n)
        """
        self.cfg_preprocessing = get_config('preprocessing')
        self.device = d
        self.add_geom = self.cfg_preprocessing.FEATURES.add_geom
        self.add_embs = self.cfg_preprocessing.FEATURES.add_embs
        self.add_hist = self.cfg_preprocessing.FEATURES.add_hist
        self.add_visual = self.cfg_preprocessing.FEATURES.add_visual
        self.add_eweights = self.cfg_preprocessing.FEATURES.add_eweights
        self.add_fudge = self.cfg_preprocessing.FEATURES.add_fudge
        self.num_polar_bins = self.cfg_preprocessing.FEATURES.num_polar_bins

        if self.add_embs:
            self.text_embedder = spacy.load('en_core_web_lg')

        if self.add_visual:
            self.visual_embedder = Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=1, classes=4)
            self.visual_embedder.load_state_dict(torch.load(CHECKPOINTS / 'backbone_unet.pth', map_location=torch.device(d))['weights'])
            self.visual_embedder = self.visual_embedder.encoder
            self.visual_embedder.to(d)
        
        self.sg = lambda rect, s : [rect[0]/s[0], rect[1]/s[1], rect[2]/s[0], rect[3]/s[1]] # scaling by img width and height
    
    def add_features(self, graphs : list, features : list):
        """ Add features to provided graphs

        Args:
            graphs (list) : list of DGLGraphs
            features (list) : list of features "sources", like text, positions and images
        
        Returns:
            chunks list and its lenght
        """

        for id, g in enumerate(tqdm(graphs, desc='adding features')):

            # positional features
            size = Image.open(features['paths'][id]).size
            feats = [[] for _ in range(len(features['boxs'][id]))]
            #geom = [self.sg(box, size) for box in features['boxs'][id]]
            chunks = []

            # 'geometrical' features
            if self.add_geom:
                
                # TODO add 2d encoding like "LayoutLM*"
                # [feats[idx].extend(self.sg(box, size)) for idx, box in enumerate(features['boxs'][id])]
                [feats[idx].extend(normalize_box(box, size[0], size[1])) for idx, box in enumerate(features['boxs'][id])]
                chunks.append(4)
            
            # HISTOGRAM OF TEXT
            if self.add_hist:
                
                [feats[idx].extend(hist) for idx, hist in enumerate(get_histogram(features['texts'][id]))]
                chunks.append(4)
            
            # textual features
            if self.add_embs:
                
                # LANGUAGE MODEL (SPACY)
                [feats[idx].extend(self.text_embedder(features['texts'][id][idx]).vector) for idx, _ in enumerate(feats)]
                chunks.append(len(self.text_embedder(features['texts'][id][0]).vector))
            
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
                [feats[idx].extend(torch.flatten(h[idx]).tolist()) for idx, _ in enumerate(feats)]
                chunks.append(len(torch.flatten(h[0]).tolist()))
        
            if self.add_eweights:
                u, v = g.edges()
                srcs, dsts =  u.tolist(), v.tolist()
                distances1 = []
                sin1 = []
                cos1 = []
                distances2 = []
                sin2 = []
                cos2 = []

                # TODO CHOOSE WHICH DISTANCE NORMALIZATION TO APPLY
                #! with fully connected simply normalized with max distance between distances
                m = sqrt((size[0]*size[0] + size[1]*size[1]))
                # parable = lambda x : (-x+1)**4
                
                for pair in zip(srcs, dsts):
                    d1, s1, c1, d2, s2, c2   = polar2(features['boxs'][id][pair[0]], features['boxs'][id][pair[1]])
                    distances1.append(d1)
                    sin1.append(s1)
                    cos1.append(c1)
                    distances2.append(d2)
                    sin2.append(s2)
                    cos2.append(c2)
                
                #distances = [d1+d2 for d1,d2 in zip(distances1,distances2)]
                #m = max(distances)
                
                distances1 = [d/m for d in distances1]
                distances2 = [d/m for d in distances2]
                polar_coordinates = to_bin2(distances1,sin1,cos1,distances2,sin2,cos2)
                g.edata['feat'] = polar_coordinates

            else:
                distances = ([0.0 for _ in range(g.number_of_edges())])
                m = 1

            g.ndata['geom'] = torch.tensor(geom, dtype=torch.float32)
            g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)

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
            if False:
                if id == rand_id and self.add_eweights:
                    print("\n\n### EXAMPLE ###")

                    img_path = features['paths'][id]
                    img = Image.open(img_path).convert('RGB')
                    draw = ImageDraw.Draw(img)

                    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
                    select = [random.randint(0, len(srcs)) for _ in range(10)]
                    for p, pair in enumerate(zip(srcs, dsts)):
                        if p in select:
                            sc = center(features['boxs'][id][pair[0]])
                            ec = center(features['boxs'][id][pair[1]])
                            draw.line((sc, ec), fill='grey', width=3)
                            middle_point = ((sc[0] + ec[0])/2,(sc[1] + ec[1])/2)
                            draw.text(middle_point, str(angles[p]), fill='black')
                            draw.rectangle(features['boxs'][id][pair[0]], fill='red')
                            draw.rectangle(features['boxs'][id][pair[1]], fill='blue')
                    
                    img.save(f'esempi/FUNSD/edges.png')

        return chunks, len(chunks)
    
    def get_info(self):
        print(f"-> textual feats: {self.add_embs}\n-> visual feats: {self.add_visual}\n-> edge feats: {self.add_eweights}")

    
