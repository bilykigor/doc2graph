import numpy as np
import networkx as nx

from difflib import SequenceMatcher

from doc2graph.src.data.image_utils import intersectoin_by_axis

center_x = lambda rect: (rect[0]+rect[2])/2
center_y = lambda rect: (rect[1]+rect[3])/2
center = lambda rect: ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def box_distance(box_left, 
                 box_right,
                 verbose=False)->float:
    """Distance between two boxes."""
    x_dist = (box_right[2] - box_left[2]) 
    if x_dist<0:
        x_dist = 0
   
    y_dist = (box_right[3] - box_left[3]) 
    if y_dist<0:
        y_dist = 0
    
    if verbose:
        print(x_dist, y_dist)
        
    distance = x_dist + 3*y_dist
    
    return distance


def features_dist(a,b):
    """Custom distance between two convoluted embedding vectors."""
    n = len(a)//2
    res1 = np.linalg.norm(a[:n]-b[:n])
    res2 = np.linalg.norm(a[n:]-b[n:])
    return res1*0.8+res2*0.2

#**********************************************************************
def get_box_to_direction(G, i, direction='right', min_share=0.8):
    edges = [e for e in G.edges(i) if G.edges[e]['direction']==direction]
    
    if edges:
        node = edges[0][1]
        neighbor_box = G.nodes[node]['box']
        box = G.nodes[i]['box']
        
        if intersectoin_by_axis('x',box, neighbor_box)>=min_share:
            return node
        
    return None


def get_line_to_direction(G, i, direction='right', min_share=0.8):
    boxes_to_right_ix = set()
    box_to_right_ix = get_box_to_direction(G, i, direction, min_share)
    orig_box = G.nodes[i]['box']
    while box_to_right_ix:
        boxes_to_right_ix.add(box_to_right_ix)
        box_to_right_ix = get_box_to_direction(G, box_to_right_ix,direction, min_share)
        
        if box_to_right_ix:
            box_to_right = G.nodes[box_to_right_ix]['box']
            if intersectoin_by_axis('x',orig_box, box_to_right)<min_share:
                break
    
    #print(boxes_to_right_ix)
    return boxes_to_right_ix


def get_line(G, i, min_share=0.8):
    res={i}
    res |= get_line_to_direction(G, i, 'right', min_share)
    res |= get_line_to_direction(G, i,  'left', min_share)
    return res


def get_lines(G, min_share=0.8):
    used=[]
    lines=[]
    for i in G.nodes():
        if i in used:
            continue
        
        line =  get_line(G, i, min_share)
        lines.append(line)
            
        used.extend(line)    
        
    return lines    

#**********************************************************************
def get_boxes_to_left_up(ix, boxes, min_share = 0.4):
    bboxs_with_id = [(ix, box) for ix, box in enumerate(boxes)]
    
    box_main = boxes[ix]
            
    boxes_to_left = [x for x in bboxs_with_id if 
                        (
                        (
                            (x[1][0]<box_main[2])# or #(0.5*(x[1][0]+x[1][2])<=box_main[2]) or #center of boxes left to right edge
                            #(intersectoin_by_axis('y',x[1], box_main)>min_share) #intesects on y
                        )
                        and (x[0]!=ix)
                        )]
    
    boxes_to_up =   [x for x in boxes_to_left if 
                        (
                        (
                            (0.5*(x[1][1]+x[1][3])<=box_main[3]) or #center over bottom
                            (intersectoin_by_axis('x',x[1], box_main)>min_share and 
                            (x[1][2]<box_main[0])) #intesects on x
                        )
                        and (x[0]!=ix)
                        )]
    
    boxes_to_up =   [x for x in boxes_to_up if not 
                        ((intersectoin_by_axis('x',x[1], box_main)>min_share) and #not boxes to the right
                        (x[1][0]>box_main[0]))]
    
    
    boxes_to_up = sorted(boxes_to_up,key=lambda x: box_distance(x[1],box_main), reverse=False)
    
    return boxes_to_up


def filter_boxes_left_up(box, box_main, boxes, boxes_x_intersected, boxes_y_intersected):
    if box in boxes_x_intersected:
        boxes = [x for x in boxes if 
                ((x[1][2]+x[1][0]>2*box[1][2]) or 
                (intersectoin_by_axis('y',x[1], box[1])>0.8)) and
                (intersectoin_by_axis('x',x[1], box_main)<0.5)  #avoid 2 connections on x
                ]
        
    elif box in boxes_y_intersected:
        boxes = [x for x in boxes if 
                ((x[1][3]+x[1][1]>2*box[1][3]) or 
                (intersectoin_by_axis('x',x[1], box[1])>0.8)) #and 
                #(intersectoin_by_axis('y',x[1], box_main)<0.5) #avoid 2 connections on y
                ]
        
    else:
        boxes = [x for x in boxes if 
                max(intersectoin_by_axis('x',x[1], box_main), intersectoin_by_axis('y',x[1], box_main))>0.8  #intersected with main
                or 
                (
                    max(intersectoin_by_axis('x',x[1], box[1]), intersectoin_by_axis('y',x[1], box[1]))<0.1 and #not intersected with diag
                    not ((x[1][0]<box[1][2]) and (x[1][1]<box[1][3])) # not in 4th quater
                )
                ]
        
    return boxes 


def get_init_neighbors(ix, box_main, boxes, min_share = 0.4):
    boxes_x_intersected = [x for x in boxes if intersectoin_by_axis('x',x[1], box_main)>max(min_share,intersectoin_by_axis('y',x[1], box_main))]
    boxes_y_intersected = [x for x in boxes if intersectoin_by_axis('y',x[1], box_main)>max(min_share,intersectoin_by_axis('x',x[1], box_main))]
    #===============================================================================
    neighbors=[]
    
    # add left neigbor
    if boxes_x_intersected:
        box = boxes_x_intersected[0]
        neighbors.append(box[0]) 
        boxes = filter_boxes_left_up(box, box_main, boxes, boxes_x_intersected, boxes_y_intersected)
    
    # add other neighbors
    top_added = False
    while boxes:
        box = boxes[0]
        
        if box in boxes_y_intersected:
            top_added = True
        elif top_added and len(neighbors)>=3:
            break
        
        neighbors.append(box[0])   
        
        boxes = boxes[1:]
        boxes = filter_boxes_left_up(box, box_main, boxes, boxes_x_intersected, boxes_y_intersected)
    
    return neighbors
           

def create_graph(words, boxes, min_share = 0.4):
    G = nx.DiGraph()
    for ix,word in enumerate(words):
        G.add_node(
            ix,
            text = word,
            #mask = mask,
            #masked_text = masked_text,
            box = boxes[ix],
            #embedding = text_to_embedding(masked_text)
            )

    horizontal_lines = []
    for ix, box_main in enumerate(boxes):
        #source_word = find_word(words[ix]) or find_words(words[ix])
        
        boxes_to_up = get_boxes_to_left_up(ix, boxes)
        
        neighbors = get_init_neighbors(ix, box_main, boxes_to_up)
        
        # If all neighbors are above and on the same level, keep the left one
        if len(neighbors)>1:
            above_neighbors = [i for i in neighbors if boxes[i][0]>=box_main[0]-2]
            if intersectoin_by_axis('x',boxes[neighbors[0]], boxes[neighbors[1]])>=min_share:
                if intersectoin_by_axis('x',boxes[neighbors[-1]], boxes[neighbors[-2]])>=min_share:
                    if len(neighbors)==len(above_neighbors):
                        above_neighbors = sorted(above_neighbors, key=lambda i: boxes[i][0])
                        neighbors = above_neighbors[:1]
        #----------------------------------------------------------------------     
        #get left_neighbor, top_neighbor
        left_neighbors = [i for i in neighbors if intersectoin_by_axis('x',boxes[ix], boxes[i])>=min_share]
        left_neighbor = left_neighbors[0] if left_neighbors else None
        
        top_neighbor = None
        top_neighbors = [i for i in neighbors if center_y(boxes[i])<box_main[1] and i!=left_neighbor] #above box_main
        top_neighbors = [i for i in top_neighbors if boxes[i][2]>box_main[0]] #has y intersection with box_main
        
        if len(top_neighbors)>1:
            # If on the same level top_neighbors, keep one with largest y share # is the rightest 
            if intersectoin_by_axis('x',boxes[top_neighbors[0]], boxes[top_neighbors[1]])>=min_share:
                if intersectoin_by_axis('x',boxes[top_neighbors[-1]], boxes[top_neighbors[-2]])>=min_share:
                    top_neighbor = sorted(top_neighbors, key=lambda i: (intersectoin_by_axis('y',box_main,boxes[i]),-boxes[i][0]), reverse=True)[0] 
                    for node in top_neighbors:
                        if node!=top_neighbor:
                            neighbors.remove(node)
            
            # or take the lowest one
            if not top_neighbor:            
                top_neighbor = sorted(top_neighbors, key=lambda i: boxes[i][3], reverse=True)[0] 
                #print('diff',G.nodes[ix]['text'],G.nodes[top_neighbor]['text'])
        elif top_neighbors:
            top_neighbor = top_neighbors[0]
        #----------------------------------------------------------------------         
        
        # Only one egde from below
        above_neighbors = [i for i in neighbors if boxes[i][3]<center_y(box_main) and i!=left_neighbor]
        for node in above_neighbors:
            below_neighbors = [e for e in G.edges(node) if G.edges[e]['direction'] in ['down','down_right']]
            #below_neighbors = [i for i in G.neighbors(node) if boxes[node][3]<center_y(boxes[i]) and #below
            #                   intersectoin_by_axis('x',boxes[node], boxes[i])<min_share]            #not left
            if below_neighbors:
                neighbors.remove(node)
        #----------------------------------------------------------------------     
        
        # Only one egde from right
        left_neighbors = [i for i in neighbors if center_x(boxes[i])<=box_main[0]]# and i!=left_neighbor]
        for node in left_neighbors:
            right_neighbors = [e for e in G.edges(node) if G.edges[e]['direction'] in ['right','down_right']]
            if right_neighbors:
                neighbors.remove(node)
                if node==left_neighbor:
                    left_neighbor = None
                    
        # if ix>=4:
        #     print(ix, neighbors, left_neighbors)
        #     break
        #----------------------------------------------------------------------    
        
        # Remove cross
        above_neighbors = [i for i in neighbors if boxes[i][3]<center_y(box_main)]
        above_neighbors = sorted(above_neighbors, key=lambda i: (intersectoin_by_axis('y',box_main,boxes[i]),-boxes[i][0]), reverse=True)
        #above_neighbors = sorted(above_neighbors, key=lambda i: boxes[i][2], reverse=True)
        if top_neighbor:
            if len(above_neighbors)>1:
                diag_neighbor = above_neighbors[1] #first one is top_neighbor
                if center_y(boxes[top_neighbor])>=center_y(boxes[diag_neighbor]):
                    neighbors.remove(diag_neighbor) #diag cant by above top
        #----------------------------------------------------------------------     
          
        # Remove if neighbors are linked, except left and top
        if left_neighbor in neighbors:
            for node in [left_neighbor,top_neighbor]:
                if node:
                    intersection = list(set(G.neighbors(node)) & set(neighbors))
                    for n in intersection:
                        if n not in [left_neighbor,top_neighbor]:
                            neighbors.remove(n)
        #----------------------------------------------------------------------      
        
        # Double check remove cross            
        for n in neighbors:
            skip = False
            
            if n==left_neighbor:
                horizontal_lines.append((min(center_y(boxes[ix]), center_y(boxes[n])), center_x(boxes[n]), center_x(boxes[ix])))
            else:
                for line in horizontal_lines:
                    if center_y(boxes[ix])>line[0]:
                        if center_y(boxes[n])<line[0]:
                            if center_x(boxes[ix])>line[1]:
                                if center_x(boxes[ix])<line[2]:
                                    skip = True
                                    break
            if skip:    
                continue    
            
            if n==top_neighbor:
                direction1 = 'up'
                direction2 = 'down'
            elif n==left_neighbor:
                direction1 = 'left'
                direction2 = 'right'
            else:
                direction1 = 'up_left'
                direction2 = 'down_right'
            
            # if 'Check' in words[ix]:
            #     print(ix,n, direction1)
            #     print(n,ix, direction2)
            G.add_edge(ix,n, direction=direction1)
            G.add_edge(n,ix, direction=direction2)
            
    return G


def split_graph(G):
    boxes = [G.nodes[node]['box'] for node in G]
   
    max_x = max([x[2] for x in boxes])
    max_y = max([x[3] for x in boxes])
    new_graphs, frames = get_frames([G],[get_containing_box(G)])
    
    zones = [graph.nodes() for graph in new_graphs]
    
    edges = list(G.edges())
    for e in edges:
        if find_zone(e[0], zones)!=find_zone(e[1], zones):
            G.remove_edge(e[0],e[1])

    
def Conv(G):
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        sum_emb = None
        if not neighbors:
            sum_emb = np.zeros(len(G.nodes[node]['embedding']))
        else:
            for n in neighbors:
                if sum_emb is None:
                    sum_emb = G.nodes[n]['embedding']
                else:
                    sum_emb += G.nodes[n]['embedding']
            
        G.nodes[node]['conv_embedding'] = np.concatenate((G.nodes[node]['embedding'], sum_emb))
 

def calc_text_embedding(G, text_to_embedding, text_to_mask=None):
    for node in G:
        text  = G.nodes[node]['text']
        if text_to_mask is not None:
            mask = text_to_mask(text)
            if mask[0]:
                masked_text = '<D>'
            elif mask[1]:
                masked_text = '<A>'
            elif mask[2]:
                masked_text = '<N>'
            else:
                masked_text = text
            G.nodes[node]['masked_text'] = masked_text
            G.nodes[node]['embedding'] = text_to_embedding(masked_text)
        else:
            G.nodes[node]['embedding'] = text_to_embedding(text)
            

def get_containing_box(G):
    boxes = [G.nodes[node]['box'] for node in G.nodes()]
    max_x = max([x[2] for x in boxes])
    max_y = max([x[3] for x in boxes])
    min_x = min([x[0] for x in boxes])
    min_y = min([x[1] for x in boxes])
    
    return [min_x,min_y,max_x,max_y]


def get_frames(graphs, frames):
    iters=0
    max_iter = 1000
    while True:
        new_graphs = []
        new_frames = []
        
        G_ix = 0
        for G,frame in zip(graphs,frames):
            zone = [G.nodes[node]['box'] for node in G.nodes()]
            h = np.median([b[3]-b[1] for b in zone])
            
            v_split = get_split(zone, 'x')
            h_split = get_split(zone, 'y')
            
            lines = None
            
            if v_split and v_split[1]-v_split[0]<h:
                v_split = None

            if h_split and h_split[1]-h_split[0]<h:
                h_split = None
                
                # lines = get_lines(G,0.6)
                # lines = [get_containing_box(G.subgraph(line)) for line in lines]

                # prev_line = None
                # for line in lines:
                #     if prev_line:
                #         if center_y(line)-center_y(prev_line)>h/2:
                #             intersection = min([line[2],prev_line[2]])-max([line[0],prev_line[0]])
                #             if intersection<0.5*max(line[2]-line[0],prev_line[2]-prev_line[0]):
                #                 h_split = (prev_line[3], line[1])
                #                 # print(prev_line, line)
                #                 # print(h_split)
                #                 # max_iter = iters
                #                 break
                #     prev_line = line
                
            if not v_split and not h_split: #no splits
                new_graphs.append(G)
                new_frames.append(frame)
                
            if not v_split and h_split: #split on y
                c1,c2 = h_split
                c = 0.5*(c1+c2)
                
                nodes_up = [node for node in G.nodes() if center_y(G.nodes[node]['box'])<c]
                nodes_down = [node for node in G.nodes() if center_y(G.nodes[node]['box'])>c]
                new_graphs.append(G.subgraph(nodes_up))
                new_graphs.append(G.subgraph(nodes_down))
                
                new_frames.append([frame[0],frame[1],frame[2],c])
                new_frames.append([frame[0],c,frame[2],frame[3]])
                
            if v_split and not h_split: #split on x
                lines = get_lines(G)
                
                c1,c2 = v_split
                c = 0.5*(c1+c2)
                nodes_left = {node for node in G.nodes() if G.nodes[node]['box'][2]<c}
                nodes_right = {node for node in G.nodes() if G.nodes[node]['box'][0]>c}
                
                line_breaks = 0
                for line in lines:
                    if line & nodes_left:
                        if line & nodes_right:
                            line_breaks += 1
                
                if True:#line_breaks==len(lines):
                    new_graphs.append(G)
                    new_frames.append(frame)
                else:
                    new_graphs.append(G.subgraph(nodes_left))
                    new_graphs.append(G.subgraph(nodes_right))
                        
                    new_frames.append([frame[0],frame[1],c,frame[3]])
                    new_frames.append([c,frame[1],frame[2],frame[3]])
                    
            if v_split and h_split:  # both splits possible
                v_size = v_split[1]-v_split[0]
                h_size = h_split[1]-h_split[0]
                
                passed = False
                # if v_size>3*h_size:
                #     c1,c2 = v_split
                #     c = 0.5*(c1+c2)
                #     nodes_left = {node for node in G.nodes() if G.nodes[node]['box'][2]<c}
                #     nodes_right = {node for node in G.nodes() if G.nodes[node]['box'][0]>c}
                    
                #     lines = get_lines(G)
                    
                #     line_breaks = 0
                #     for line in lines:
                #         if line & nodes_left:
                #             if line & nodes_right:
                #                 line_breaks += 1
                    
                #     if line_breaks==len(lines):
                #         new_graphs.append(G.subgraph(nodes_left))
                #         new_graphs.append(G.subgraph(nodes_right))
                    
                #         new_frames.append([frame[0],frame[1],c,frame[3]])
                #         new_frames.append([c,frame[1],frame[2],frame[3]])
                        
                #         passed = True

                if not passed:
                    c1,c2 = h_split
                    c = 0.5*(c1+c2)
                    nodes_up = [node for node in G.nodes() if center_y(G.nodes[node]['box'])<c]
                    nodes_down = [node for node in G.nodes() if center_y(G.nodes[node]['box'])>c]
                    new_graphs.append(G.subgraph(nodes_up))
                    new_graphs.append(G.subgraph(nodes_down))
                    
                    new_frames.append([frame[0],frame[1],frame[2],c])
                    new_frames.append([frame[0],c,frame[2],frame[3]])
            
            G_ix+=1  
       
        iters+=1 
        if len(graphs)==len(new_graphs):
            return new_graphs, new_frames
        if iters>max_iter:
            return new_graphs, new_frames
        
        graphs = new_graphs
        frames = new_frames


def merge_intervals(intervals):
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    return merged


def get_split(boxes, direction='x'):
    if direction=='x':
        intervals = [(box[0], box[2]) for box in boxes]
    else:
        intervals = [(box[1], box[3]) for box in boxes]
        
    intervals = sorted(intervals, key=lambda x: x[0])
    intervals = merge_intervals(intervals)
    
    gaps = []
    prev=None
    for interval in intervals:
        if prev is not None:
            gaps.append((interval[0]-prev[1], (prev[1],interval[0])))
        prev = interval
    
    gaps = sorted(gaps, key=lambda x: x[0])
    
    if gaps:
        return gaps[-1][1]
    return None
    

def find_zone(node, zones):
    for ix, zone in enumerate(zones):
        if node in zone:
            return ix
        
    return -1


def find_target_node_by_text(ix,
                            source_G,
                            target_G,
                            pairwise_distance,
                            threshold = 0.5,
                            text_threshold = 0.8):
    
    d = min(pairwise_distance[ix])
    
    if d > threshold:
        return None
    
    source_nodes = list(source_G.nodes())
    target_nodes = list(target_G.nodes())
        
    target_ix = np.argmin(pairwise_distance[ix])  
    
    source_node = source_nodes[ix]
    target_node = target_nodes[target_ix]
        
    source_text = source_G.nodes[source_node]['text']
    target_text = target_G.nodes[target_node]['text']
    
    s = SequenceMatcher(None, source_text, target_text)

    if s.ratio()<text_threshold:
        print(f'Text not match: {source_text} -> {target_text}', s.ratio(), d)
        return None
    
    print('GRAPH', source_text,'--->',target_text, d)
    
    return target_node  