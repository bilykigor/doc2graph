import numpy as np
import networkx as nx
from collections import deque
from networkx.algorithms import isomorphism

from difflib import SequenceMatcher

from doc2graph.src.data.image_utils import points_distance, get_intersection, intersectoin_by_axis, normalize_box

center_x = lambda rect: (rect[0]+rect[2])/2
center_y = lambda rect: (rect[1]+rect[3])/2
box_h = lambda rect: max(rect[1],rect[3])-min(rect[1],rect[3])
box_w = lambda rect: max(rect[0],rect[2])-min(rect[0],rect[2])
center = lambda rect: ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
area = lambda rect: abs(rect[0] - rect[2]) * abs(rect[1] - rect[3])

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


def box_distance_for_split(box_left, 
                 box_right,
                 verbose=False)->float:
    """Distance between two boxes."""
    if intersectoin_by_axis('y',box_left,box_right)>0:
        x_dist = 0
    elif box_left[2]<=box_right[0]:
        x_dist = box_right[0] - box_left[2]
    elif box_right[2]<=box_left[0]:
        x_dist = box_left[0] - box_right[2]
    else:
        raise
        
    if intersectoin_by_axis('x',box_left,box_right)>0:
        y_dist = 0
    else:
        y_dist = abs(center_y(box_left)-center_y(box_right))
    
    if verbose:
        print(x_dist, y_dist)
        
    distance = np.sqrt(x_dist*x_dist + y_dist*y_dist)
    
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


def strict_frame(G):
    lines = get_lines(G)
    
    if len(lines)<2:
        return False
    
    flatten_lines = [n for line in lines for n in line]
    if len(flatten_lines)!=len(G):
        return False
    
    boxes = [get_containing_box(G.subgraph(line)) for line in lines]
    boxes.sort(key = lambda x: center_y(x))
    
    h = list(map(box_h,boxes))
    if np.std(h)/np.mean(h)>0.25:
        return False
    
    d = np.diff(list(map(center_y,boxes)))
    if np.std(d)/np.mean(d)>0.25:
        return False
    
    if max(d)>np.mean(h)+np.std(h):
        return False

    left = list(map(lambda x: x[0],boxes))
    if np.std(left)>np.mean(h):
        return False
    
    right = list(map(lambda x: x[2],boxes))
    if np.std(right)>np.mean(h):
        return False
    
    return True


#**********************************************************************
def get_boxes_inside(boxes, bounding_box):
    return [b for b in boxes if get_intersection(b,bounding_box) is not None]
    

def get_graph_inside(G, bounding_box):
    nodes = [n for n in G if get_intersection(G.nodes[n]['box'],bounding_box) is not None]
    if not nodes:
        return None
    return G.subgraph(nodes)
  
    
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

    horizontal_edges = []
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
        for above_neighbor in above_neighbors:
            current_distance = box_distance(G.nodes[above_neighbor]['box'], box_main)
            below_neighbors = [n for n in G.neighbors(above_neighbor) if G.edges[(above_neighbor,n)]['direction'] in ['down','down_right']]
            for below_neighbor in below_neighbors:
                existed_distance = box_distance(G.nodes[above_neighbor]['box'], G.nodes[below_neighbor]['box'],)
                if current_distance < existed_distance:
                    G.remove_edge(above_neighbor, below_neighbor)
                    G.remove_edge(below_neighbor, above_neighbor)
                else:
                    neighbors.remove(above_neighbor)
                    break
        
        
        
        #----------------------------------------------------------------------     
        # Only one egde from right
        left_neighbors = [i for i in neighbors if center_x(boxes[i])<=box_main[0] and i!=top_neighbor]# and i!=left_neighbor]
        for node in left_neighbors.copy():
            current_distance = box_distance(G.nodes[node]['box'], box_main)
            right_neighbors = [n for n in G.neighbors(node) if G.edges[(node,n)]['direction'] in ['right','down_right']]
            for right_neighbor in right_neighbors:
                neighbors.remove(node)
                left_neighbors.remove(node)
                if node==left_neighbor:
                    left_neighbor = None
                break
                    
                
        #----------------------------------------------------------------------    
        
        # Remove cross
        above_neighbors = [i for i in neighbors if boxes[i][3]<center_y(box_main)]
        above_neighbors = sorted(above_neighbors, key=lambda i: (boxes[i][1], intersectoin_by_axis('y',box_main,boxes[i]),-boxes[i][0]), reverse=True)
        #above_neighbors = sorted(above_neighbors, key=lambda i: boxes[i][2], reverse=True)
        if top_neighbor:
            if len(above_neighbors)>1:
                diag_neighbor = above_neighbors[1] #first one is top_neighbor
                if center_y(boxes[top_neighbor])>=center_y(boxes[diag_neighbor]):
                    neighbors.remove(diag_neighbor) #diag cant by above top
                    
        # if ix>=13:
        #     print(words[ix],ix, neighbors, above_neighbors, left_neighbors,left_neighbor, top_neighbor)
        #     print([words[i] for i in neighbors])
        #     break   
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
          
        # Diag neighbors can be only above left neighbor
        if left_neighbor:
            if len(left_neighbors)>1:
                for node in left_neighbors.copy():
                    if node!=left_neighbor:
                        if center_y(boxes[node])>center_y(boxes[left_neighbor]):
                            neighbors.remove(node)
                            left_neighbors.remove(node)
                            
        
        #----------------------------------------------------------------------      
        
        # Double check remove cross            
        for n in neighbors:
            skip = False
            
            current_edge = (center_x(boxes[n]), center_y(boxes[n]), center_x(box_main),center_y(box_main))
            if n==left_neighbor:
                horizontal_edges.append(current_edge)
            else:
                for edge in horizontal_edges:
                    intersection_point = line_intersection(edge, current_edge)
                    if not intersection_point:
                        continue
                    
                    if points_distance(intersection_point, current_edge[:2])<1:
                        continue
                    
                    if points_distance(intersection_point, current_edge[-2:])<1:
                        continue
                    
                    if points_distance(intersection_point, edge[:2])<1:
                        continue
                    
                    if points_distance(intersection_point, edge[-2:])<1:
                        continue
                    
                    x,y = intersection_point
                    
                    if not get_intersection([
                                        min(current_edge[0],current_edge[2]),
                                        min(current_edge[1],current_edge[3]),
                                        max(current_edge[0],current_edge[2]),
                                        max(current_edge[1],current_edge[3]),
                                        ],(x,y,x,y)):
                        continue
                    
                    if not get_intersection([
                                        min(edge[0],edge[2]),
                                        min(edge[1],edge[3]),
                                        max(edge[0],edge[2]),
                                        max(edge[1],edge[3]),
                                        ],(x,y,x,y)):
                        continue
                        
                    # if ix==17:
                    #     print(edge, current_edge, x,y)
                    skip = True
                    break
            if skip:    
                continue    
            
            # if ix==14:
            #     print(words[n])
            
            if n==top_neighbor:
                direction1 = 'up'
                direction2 = 'down'
            elif n==left_neighbor:
                direction1 = 'left'
                direction2 = 'right'
            else:
                direction1 = 'up_left'
                direction2 = 'down_right'
            
            h = box_main[3]-box_main[1]
            h_n = boxes[n][3] - boxes[n][1]
            mean_h = 0.5*(h+h_n)
            distance = box_distance_for_split(boxes[n], box_main)#/mean_h
            
            G.add_edge(ix,n, direction=direction1, distance = distance)
            G.add_edge(n,ix, direction=direction2, distance = distance)
            
    return G


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate the determinants
    det_line1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    det_line2 = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)

    if det_line1 == 0:
        # Lines are parallel or coincident
        return None

    # Calculate the x and y coordinates
    x = det_line2 / det_line1
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det_line1

    return (x, y)


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
            
     
def split_graph_by_path_vert(G, axis = 1, multiplier = 1.5):
    splitted_any = False
    
    while True:
        zones = [list(nodes) for nodes in nx.connected_components(G.to_undirected())]
        
        splitted = False
        edges_to_remove = []
        for zone in zones:
            removed = None
            debug = False
            
            if len(zone)<4:
                continue
            
            is_strict = strict_frame(G.subgraph(zone))
            if is_strict:
                continue
            
            path = get_shortest_path(G.subgraph(zone),zone[0])
            
            if axis==1:
                edges_vert = [x for x in path if (x[2]['direction'] not in ['left','right'])]
            else:
                edges_vert = [x for x in path if (x[2]['direction'] in ['left','right'])]
            edges_vert.sort(key = lambda x: x[2]['distance'])
            
            distances_vert = [x[2]['distance'] for x in edges_vert]
            
            if multiplier>0 and distances_vert:
                bound_vert = upper_outlier_bound(distances_vert,multiplier)
                
                if max(distances_vert)>bound_vert: #remove largest one
                    e = edges_vert[-1]
                    G.remove_edge(e[0],e[1])
                    G.remove_edge(e[1],e[0])
                    
                    edges_vert = edges_vert[:-1]
                    removed = e
            
            if axis==1:
                edges_horizontal = [x for x in G.subgraph(zone).edges(data = True) if x[2]['direction'] in ['left','right']]
            else:
                edges_horizontal = [x for x in G.subgraph(zone).edges(data = True) if x[2]['direction'] not in ['left','right']]
            
            path = []
            path.extend(edges_vert)
            path.extend(edges_horizontal)
            path_G = nx.from_edgelist(path)
            
            sub_zones = [list(nodes) for nodes in nx.connected_components(path_G)]
            
            if (len(sub_zones)>1) or removed:
                splitted = True
                splitted_any = True
                
                edges = list(G.edges())
                for e in edges:
                    if find_zone(e[0], sub_zones)!=find_zone(e[1], sub_zones):
                        G.remove_edge(e[0],e[1])
            
        if not splitted:
            break
    
    return splitted_any
        
            
def split_graph_by_path(G, vert_multiplier = 1.5, horiz_multiplier = 1.5):
    G_copy = G.copy()
    while True:
        splited = split_graph_by_path_vert(G_copy, axis = 1, multiplier = vert_multiplier)
        if not splited:
            splited = split_graph_by_path_vert(G_copy, axis = 0, multiplier = horiz_multiplier)
            
        if not splited:
            break
        
    sub_zones = [list(nodes) for nodes in nx.connected_components(G_copy.to_undirected())]
    edges = list(G.edges())
    for e in edges:
        if find_zone(e[0], sub_zones)!=find_zone(e[1], sub_zones):
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
 

def calc_text_embedding(G, text_to_embedding=None, text_to_mask=None):
    for node in G:
        text  = G.nodes[node]['text']
        if text_to_mask is not None:
            mask = text_to_mask(text)
            if mask[0]:
                masked_text = '<NW>'#'<D>'
            elif mask[1]:
                masked_text = '<NW>'#'<A>'
            elif mask[2]:
                masked_text = '<NW>'#'<N>'
            elif mask[3]:
                masked_text = '<NW>' #'<C>'
            else:
                masked_text = text
            G.nodes[node]['masked_text'] = masked_text
            if text_to_embedding is not None:
                G.nodes[node]['embedding'] = text_to_embedding(masked_text)
        elif text_to_embedding is not None:
            G.nodes[node]['embedding'] = text_to_embedding(text)
            

def get_containing_box(G):
    if not len(G):
        return None
    
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


def get_all_subgraphs(G, min_length = 3):
    subgraphs = []
    for sg1_nodes in nx.connected_components(G):
        if len(sg1_nodes)<min_length:
            continue
        sg1 = G.subgraph(sg1_nodes)
        
        used=[]
        for i in sg1_nodes:
            #if len(list(G.neighbors(i)))!=1:
            #    continue
            used.append(i)
            others = [ix for ix in sg1_nodes if ix not in used]
            for path in nx.all_simple_paths(sg1, source=i, target=others):
                if len(path)<min_length:
                    continue
                set_path = set(path)
                if set_path not in subgraphs:
                    subgraphs.append(set_path)
    subgraphs = sorted(subgraphs, key=len, reverse=True)
    return subgraphs


def get_all_simple_paths(G, cutoff):
    used=[]
    pathes = []
    for i in G:
        #if len(list(G.neighbors(i)))!=1:
        #    continue
        used.append(i)
        others = [ix for ix in G if ix != i]
        for path in nx.all_simple_paths(G, source=i, target=others, cutoff=cutoff-1):
            if len(path)<cutoff:
                continue
            #set_path = set(path)
            if path not in pathes:
                pathes.append(path)
    return pathes


def get_all_egos(G, radius=1):
    res = []
    for i in G:
        #if len(list(G.neighbors(i)))!=1:
        #    continue
        
        res.append((i,nx.ego_graph(G,i,radius)))
        
    return res


def same_graphs(sg1,sg2,label='text'):
    if sg1.number_of_nodes() != sg2.number_of_nodes():
        return False
    
    if sg1.number_of_edges() != sg2.number_of_edges():
        return False
    
    if {x[1][label] for x in sg1.nodes(data=True)} != {x[1][label] for x in sg2.nodes(data=True)}:
        #print('text')
        return False
        
    GM = isomorphism.GraphMatcher(sg1, sg2, node_match=lambda n1, n2: n1[label] == n2[label], edge_match=lambda e1, e2: e1['direction'] == e2['direction'])
    if GM.is_isomorphic():
        return True
    return False


def collapse_nodes(G, nodes_to_collapse):
    new_node = nodes_to_collapse[0]
    edges = list(G.edges())
    for u,v in edges:
        if u in nodes_to_collapse and v in nodes_to_collapse:
            G.remove_edge(u,v)
        elif u in nodes_to_collapse[1:]:
            G.add_edge(new_node,v, direction = G.edges[(u,v)]['direction'])
            G.remove_edge(u,v)
            
        elif v in nodes_to_collapse[1:]:
            G.add_edge(u,new_node, direction = G.edges[(u,v)]['direction'])    
            G.remove_edge(u,v)   
    
    x = [G.nodes[n]['text'] for n in nodes_to_collapse]
    text = set.union(*x) 
    G.nodes[nodes_to_collapse[0]]['text'] =  text 
    for n in nodes_to_collapse[1:]:
        G.remove_node(n)
        

def match_neighbors(gu, hu, source_G, target_H):
    matched = set()
    
    for gv in source_G.neighbors(gu):
        for hv in target_H.neighbors(hu):
            if source_G.nodes[gv]['text'] == target_H.nodes[hv]['text']:
                if source_G.edges[(gu,gv)]['direction'] == target_H.edges[(hu,hv)]['direction']:
                    d1 = source_G.edges[(gu,gv)]['distance']
                    d2 = target_H.edges[(hu,hv)]['distance']
                    matched.add((gv, hv))
                    # if d1<5 and d2<5:
                    #     matched.add((gv, hv))
                    # else:
                    #     #if abs(d1/d2-1)<0.1 and abs(d1-d2)<4:
                    #     matched.add((gv, hv))
                    #     #if gv==45:
                    #     #print(gu, source_G.nodes[gu]['text'],source_G.nodes[gv]['text'],d1,d2)
    return matched


def cbfs_matching(source_G, target_H, i, j):
    if source_G.nodes[i]['text']!=target_H.nodes[j]['text']:
        return []
    
    #if source_G.nodes[i]['text'] in ('<D>','<A>','<N>','<C>'):
    #    return []
    # Initialize queues for both graphs
    GQ = deque()
    HQ = deque()

    # Initialize colors for all nodes in both graphs
    colors_G = {v: 'WHITE' for v in source_G}
    colors_H = {v: 'WHITE' for v in target_H}

    # Set starting nodes' colors to GRAY
    colors_G[i] = 'GRAY'
    colors_H[j] = 'GRAY'

    # Initialize matching set
    M = [(i, j)]

    # Enqueue starting nodes
    GQ.append(i)
    HQ.append(j)

    while GQ and HQ:
        gu = GQ.popleft()
        hu = HQ.popleft()

        # Find matching neighbors
        N = match_neighbors(gu, hu, source_G, target_H)

        for gv, hv in N:
            if colors_G[gv] == 'WHITE' and colors_H[hv] == 'WHITE':
                M.append((gv, hv))
                GQ.append(gv)
                HQ.append(hv)
                colors_G[gv] = 'GRAY'
                colors_H[hv] = 'GRAY'

    not_mask_exist = False
    for g,h in M:
        if source_G.nodes[g]['text'] not in ('<NW>','<D>','<A>','<N>','<C>'):
            #if len(source_G.nodes[g]['text'])>4:
            not_mask_exist = True
            break
    
    if not not_mask_exist:
        M=[]
    
    return M    


def cbfs_matching_score(M, source_graph_shared, target_graph_shared, source_image_size, target_image_size, source_containing_area, target_containing_area):
    if not M:
        return 0,0
    
    source_graph_matched = source_graph_shared.subgraph([x[0] for x in M])
    target_graph_matched = target_graph_shared.subgraph([x[1] for x in M])
    
    source_containing_area_matched = area(normalize_box(get_containing_box(source_graph_matched), source_image_size[0], source_image_size[1]))
    target_containing_area_matched = area(normalize_box(get_containing_box(target_graph_matched), target_image_size[0], target_image_size[1]))

    score_1 = len(M)+len(M)
    score_1 /= len(source_graph_shared)+len(target_graph_shared)
    
    #print(source_containing_area_matched, target_containing_area_matched)
    score_2 = source_containing_area_matched+target_containing_area_matched
    score_2 /= source_containing_area+target_containing_area
    
    return score_1,score_2


def get_max_graph(source_graph_shared, target_graph_shared, source_image_size, target_image_size, source_containing_area=1, target_containing_area=1, source_node=None):
    max_M = None
    max_score=0
    source_nodes = source_graph_shared.nodes()
    if source_node:
        source_nodes = [source_node]
    for i in source_nodes:
        for j in target_graph_shared:
            M = cbfs_matching(source_graph_shared, target_graph_shared, i, j)
            if len(M)>=2:
                s1,s2 = cbfs_matching_score(M, source_graph_shared, target_graph_shared, source_image_size, target_image_size, source_containing_area, target_containing_area)
                s= s1*s2
                if max_M is None:
                    max_M = M
                    max_score = s
                    (I,J) = (i,j)
                    (S1,S2) = (s1,s2)
                elif s>max_score:
                    max_M = M
                    max_score = s
                    (I,J) = (i,j)
                    (S1,S2) = (s1,s2)
    
    if not source_node:
        if max_M:
            source_nodes_M = [x[0] for x in max_M]
            target_nodes_M = [x[1] for x in max_M]
            
            source_nodes_remaining = [x for x in source_graph_shared if x not in source_nodes_M]
            target_nodes_remaining = [x for x in target_graph_shared if x not in target_nodes_M]
            
            if source_nodes_remaining:
                if target_nodes_remaining:
                    M_sub = get_max_graph(source_graph_shared.subgraph(source_nodes_remaining), \
                                        target_graph_shared.subgraph(target_nodes_remaining), \
                                        source_image_size, target_image_size, source_containing_area, target_containing_area)
                    if M_sub:
                        max_M.extend(M_sub)
                    
    return max_M


def get_shortest_path(G, start_node = None, start_edge= None):
    if start_edge is not None:
        used_nodes=[start_edge[0],start_edge[1]]
        edges = []
        edges.extend(list(G.edges(start_edge[0], data=True)))
        edges.extend(list(G.edges(start_edge[1], data=True)))
        path = [(start_edge[0],start_edge[1],G.get_edge_data(start_edge[0],start_edge[1]))]
        edges = [e for e in edges if e[1] not in used_nodes]
    elif start_node is None:
        start_node = list(G.nodes())[0]
        used_nodes=[start_node]
        edges = list(G.edges(start_node, data=True))
        path = []
    else:
        used_nodes=[start_node]
        edges = list(G.edges(start_node, data=True))
        path = []
    
    while edges:
        edges.sort(key=lambda x: x[2]['distance'])
        #print(edges)
        shortest_edge = edges[0]
        path.append(shortest_edge)
        #print(shortest_edge)
        next_node = shortest_edge[1]
        next_edges = list(G.edges(next_node, data=True))
        #print(next_edges)
        edges.extend(next_edges)
        
        used_nodes.append(next_node)
        edges = [e for e in edges if e[1] not in used_nodes]
        
    return path
 
 
def upper_outlier_bound(data, mult = 2.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    M = np.percentile(data, 50)
    s = np.std(data)
    IQR = Q3 - Q1

    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 +  mult * IQR
    upper_bound = M + mult*s

    #data = [x for x in data if (x < upper_bound)]

    return upper_bound
       

def frames_dist(a,b, source_graph_shared,target_graph_shared, source_image_size, target_image_size):
    if len(a)==1:
        if len(b)==1:
            if source_graph_shared.nodes[a[0]]['text']==target_graph_shared.nodes[b[0]]['text']:
                return 1
    
    sg1 = source_graph_shared.subgraph(a)
    sg2 = target_graph_shared.subgraph(b)
    
    M = get_max_graph(sg1, 
                    sg2, 
                    source_image_size, 
                    target_image_size, 
                    source_node=None)
    
    if M:
        s1, s2 = cbfs_matching_score(M, source_graph_shared, target_graph_shared, source_image_size, target_image_size, 1, 1)
        return s1*s2
    
    return 0