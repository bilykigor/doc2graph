import math
from math import sqrt


def polar(rect_src : list, rect_dst : list):
    """Compute distance and angle from doc2graph.src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    x0_src, y0_src, x1_src, y1_src = rect_src
    x0_dst, y0_dst, x1_dst, y1_dst = rect_dst
    
    # check relative position
    left = (x1_dst - x0_src) <= 0
    bottom = (y1_src - y0_dst) <= 0
    right = (x1_src - x0_dst) <= 0
    top = (y1_dst - y0_src) <= 0
    
    vp_intersect = (x0_src <= x1_dst and x0_dst <= x1_src) # True if two rects "see" each other vertically, above or under
    hp_intersect = (y0_src <= y1_dst and y0_dst <= y1_src) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect 

    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

    # evaluate reciprocal position
    sc = center(rect_src)
    ec = center(rect_dst)
    new_ec = (ec[0] - sc[0], ec[1] - sc[1])
    angle = int(math.degrees(math.atan2(new_ec[1], new_ec[0])) % 360)
    
    if rect_intersect:
        return 0, angle
    elif top and left:
        a, b = (x1_dst - x0_src), (y1_dst - y0_src)
        return int(sqrt(a**2 + b**2)), angle
    elif left and bottom:
        a, b = (x1_dst - x0_src), (y0_dst - y1_src)
        return int(sqrt(a**2 + b**2)), angle
    elif bottom and right:
        a, b = (x0_dst - x1_src), (y0_dst - y1_src)
        return int(sqrt(a**2 + b**2)), angle
    elif right and top:
        a, b = (x0_dst - x1_src), (y1_dst - y0_src)
        return int(sqrt(a**2 + b**2)), angle
    elif left:
        return (x0_src - x1_dst), angle
    elif right:
        return (x0_dst - x1_src), angle
    elif bottom:
        return (y0_dst - y1_src), angle
    elif top:
        return (y0_src - y1_dst), angle
       
       
def polar2(rect_src : list, rect_dst : list):
    """Compute distance and angle from doc2graph.src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    x0_src, y0_src, x1_src, y1_src = rect_src
    x0_dst, y0_dst, x1_dst, y1_dst = rect_dst
    
    a1, b1 = (x0_dst - x0_src), (y0_dst - y0_src)
    d1 = sqrt(a1**2 + b1**2)
    if d1==0:
        sin1  = 0
        cos1  = 0
    else:
        sin1  = b1/d1
        cos1  = a1/d1
    
    a2, b2 = (x1_dst - x1_src), (y1_dst - y1_src)
    d2 = sqrt(a2**2 + b2**2)
    if d2==0:
        sin2  = 0
        cos2  = 0
    else:
        sin2  = b2/d2
        cos2  = a2/d2
    
    return sin1, cos1, sin2, cos2  


def polar3(rect_src : list, rect_dst : list):
    
    x0_src, y0_src, x1_src, y1_src = rect_src
    x0_dst, y0_dst, x1_dst, y1_dst = rect_dst
    
    a1, b1 = (x0_dst - x0_src), (y0_dst - y0_src)
    d1 = sqrt(a1**2 + b1**2)
    
    a2, b2 = (x1_dst - x1_src), (y1_dst - y1_src)
    d2 = sqrt(a2**2 + b2**2)
    
    return d1,d2 


def get_histogram(contents : list):
    """Create histogram of content given a text.

    Args;
        contents (list)

    Returns:
        list of [x, y, z] - 3-dimension list with float values summing up to 1 where:
            - x is the % of literals inside the text
            - y is the % of numbers inside the text
            - z is the % of other symbols i.e. @, #, .., inside the text
    """
    
    c_histograms = list()

    for token in contents:
        num_symbols = 0 # all
        num_literals = 0 # A, B etc.
        num_figures = 0 # 1, 2, etc.
        num_others = 0 # !, @, etc.
        
        histogram = [0.0000, 0.0000, 0.0000, 0.0000]
        
        for symbol in token.replace(" ", ""):
            if symbol.isalpha():
                num_literals += 1
            elif symbol.isdigit():
                num_figures += 1
            else:
                num_others += 1
            num_symbols += 1

        if num_symbols != 0:
            histogram[0] = num_literals / num_symbols
            histogram[1] = num_figures / num_symbols
            histogram[2] = num_others / num_symbols
            
            # keep sum 1 after truncate
            if sum(histogram) != 1.0:
                diff = 1.0 - sum(histogram)
                m = max(histogram) + diff
                histogram[histogram.index(max(histogram))] = m
        
        # if symbols not recognized at all or empty, sum everything at 1 in the last
        if histogram[0:3] == [0.0,0.0,0.0]:
            histogram[3] = 1.0
        
        c_histograms.append(histogram)
        
    return c_histograms
