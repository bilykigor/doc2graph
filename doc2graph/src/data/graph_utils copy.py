

def split_graph_by_path_vert(G, axis = 1, multiplier = 1.5):
    splitted_any = False
    ix=0
    while True:
        #ix+=1
        print('---------------------------------')
        zones = [list(nodes) for nodes in nx.connected_components(G.to_undirected())]
        
        splitted = False
        for zone in zones:
            print('---------')
            removed = None
            
            if len(zone)<4:
                continue
            
            G_sub_zone = G.subgraph(zone)
            G_sub_zone_undir = G_sub_zone.to_undirected()
            G_sub_cycles = nx.minimum_cycle_basis(G_sub_zone_undir)
            
            is_strict = strict_frame(G_sub_zone)
            if is_strict:
                continue
            
            #print('Zone', zone)
            
            # Remove not chained nodes
            # chains = list(nx.chain_decomposition(G_sub_zone.to_undirected()))
            # path = [n for line in chains for n in line]
            # path_G = nx.from_edgelist(path)
            # sub_zones = [list(nodes) for nodes in nx.connected_components(path_G)]
            
            # not_chained_nodes = set(zone)-set(path_G.nodes())
            # for node in not_chained_nodes:
            #     edges = list(G_sub_zone.edges(node, data=True))
            #     if len(edges)==1:
            #         continue
            #     edges.sort(key = lambda x: x[2]['distance'])
            #     edges = edges[1:]
            #     for e in edges:
            #         print(f'Remove separate {e}')
            #         G.remove_edge(e[0],e[1])
            #         G.remove_edge(e[1],e[0])
            
            if False:#len(sub_zones)>1:
                #print('Chains before',sub_zones)
                splitted = True
                splitted_any = True
                
                edges = list(G_sub_zone.edges())
                for e in edges:
                    zone_0 = find_zone(e[0], sub_zones)
                    zone_1 = find_zone(e[1], sub_zones)
                    
                    if min(zone_0,zone_1)>-1:
                        if zone_0!=zone_1:
                            print(f'Remove diff zones {e}')
                            G.remove_edge(e[0],e[1])
                            
                sub_zones = [list(nodes) for nodes in nx.connected_components(G_sub_zone.to_undirected())]
                #print('Chains after ',sub_zones,len(sub_zones))
            else:
                mst = nx.minimum_spanning_tree(G_sub_zone.to_undirected(), weight='distance')
                path = list(mst.edges(data=True)) #path = get_shortest_path(G_sub_zone,zone[0])
                
                if axis==1:
                    edges_axis = [x for x in path if (x[2]['direction'] not in ['left','right'])]
                else:
                    edges_axis = [x for x in path if (x[2]['direction'] in ['left','right'])]
                edges_axis.sort(key = lambda x: x[2]['distance'], reverse=True)
                
                if multiplier>0 and edges_axis:
                    distances_vert = [x[2]['distance'] for x in edges_axis]
                    bound_vert = upper_outlier_bound(distances_vert,multiplier)
                    
                    i = 0
                    for i, e in enumerate(edges_axis):
                        if e[2]['distance']<bound_vert:
                            break
                        
                        ec = edge_cycles(e, G_sub_cycles)
                        
                        print(f'Remove long {e}')
                        print(f'Cycles', ec)
                        G.remove_edge(e[0],e[1])
                        G.remove_edge(e[1],e[0])
                        removed = e
                        i += 1
                        break
                    
                    edges_axis = edges_axis[i:]
                
                if axis==1:
                    edges_other_axis = [x for x in G_sub_zone.edges(data = True) if x[2]['direction'] in ['left','right']]
                else:
                    edges_other_axis = [x for x in G_sub_zone.edges(data = True) if x[2]['direction'] not in ['left','right']]
                
                path = []
                path.extend(edges_axis)
                path.extend(edges_other_axis)
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
        if ix >7:
            break
    
    return splitted_any


def split_graph_by_path_vert_v2(G, axis = 1, multiplier = 1.5):
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
                edges_vert_all = [x for x in G.subgraph(zone).edges(data=True) if (x[2]['direction'] not in ['left','right'])]
            else:
                edges_vert = [x for x in path if (x[2]['direction'] in ['left','right'])]
                edges_vert_all = [x for x in G.subgraph(zone).edges(data=True) if (x[2]['direction'] in ['left','right'])]
            edges_vert.sort(key = lambda x: x[2]['distance'])
            edges_vert_all.sort(key = lambda x: x[2]['distance'])
            
            distances_vert = [x[2]['distance'] for x in edges_vert]
            distances_vert_all = [x[2]['distance'] for x in edges_vert_all]
            
            if multiplier>0 and distances_vert_all and distances_vert:
                bound_vert = upper_outlier_bound(distances_vert_all,multiplier)
                
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
        
 