from math import sqrt
from typing import Tuple
import cv2
import numpy as np
import torch
import math
import fitz
from PIL import Image, ImageDraw
# import re
# from nltk.tokenize import word_tokenize
# from price_parser import Price
# import dateparser
#import torchvision.ops.boxes as bops


def file_to_images(file, gray=False):
    if file[-3:].lower() == 'pdf':
        imgs = []
        
        zoom = 3    # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        
        with fitz.open(file) as pdf:
            for pno in range(pdf.page_count):
                page = pdf.load_page(pno)
                pix = page.get_pixmap(matrix=mat)
                # if width or height > 2000 pixels, don't enlarge the image
                #if pix.width > 2000 or pix.height > 2000:
                #    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1)
                
                mode = "RGBA" if pix.alpha else "RGB"                        
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)                        
                
                if gray:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
                    
                imgs.append(img)
    else:
        if gray:
            img = Image.open(file).convert('L')
        else:
            img = Image.open(file).convert('RGB')
            
        imgs=[img]

    return imgs


def intersectoin_by_axis(axis: str, rect_src : list, rect_dst : list):
        #making same x coordinates
    rect_src = rect_src.copy()
    rect_dst = rect_dst.copy()
    
    if  rect_src[0]==rect_src[2]:
        return 0   
    if  rect_src[1]==rect_src[3]:
        return 0 
    if  rect_dst[0]==rect_dst[2]:
        return 0   
    if  rect_dst[1]==rect_dst[3]:
        return 0   
        
    if axis=='x':
        if min(rect_src[3],rect_dst[3]) <= max(rect_dst[1],rect_src[1]):
            return 0
        
        rect_dst[0]=rect_src[0]
        rect_dst[2]=rect_src[2]
        
        w = rect_dst[2] - rect_dst[0]
        h = min(rect_src[3],rect_dst[3]) - max(rect_dst[1],rect_src[1])
            
        res = w*h
    else:
        if min(rect_src[2],rect_dst[2]) <= max(rect_dst[0],rect_src[0]):
            return 0
        
        rect_dst[1]=rect_src[1]
        rect_dst[3]=rect_src[3]
        
        h = rect_dst[3] - rect_dst[1]
        w = min(rect_src[2],rect_dst[2]) - max(rect_dst[0],rect_src[0])
        res = w*h
        
    area_A = (rect_dst[3]-rect_dst[1])*(rect_dst[2]-rect_dst[0])
    area_B = (rect_src[3]-rect_src[1])*(rect_src[2]-rect_src[0])
    
    # area = bops.box_iou(torch.tensor([rect_dst], dtype=torch.float), torch.tensor([rect_src], dtype=torch.float))
    # area_A = bops.box_area(torch.tensor([rect_dst], dtype=torch.float))
    # area_B = bops.box_area(torch.tensor([rect_src], dtype=torch.float))
    
    #res = area/(1+area)*(area_A+area_B)
    try:
        area = res/min([area_A,area_B])
    except:
        print([rect_src,rect_dst])
        raise
    
    return area

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

def transform_image(img_path : str, scale_image=1.0):
    """ Transform image to torch.Tensor

    Args:
        img_path (str) : where the image is stored
        scale_image (float) : how much scale the image
    """

    np_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    width = int(np_img.shape[1] * scale_image)
    height = int(np_img.shape[0] * scale_image)
    new_size = (width, height)
    np_img = cv2.resize(np_img,new_size)
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    img = img[None,None,:,:]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = 1.0 - img / 128.0
    
    return img

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

def to_bin(dist :int, angle : int, b=8):
    """ Discretize the space into equal "bins": return a distance and angle into a number between 0 and 1.

    Args:
        dist (int): distance in terms of pixel, given by "polar()" util function
        angle (int): angle between 0 and 360, given by "polar()" util function
        b (int): number of bins, MUST be power of 2
    
    Returns:
        torch.Tensor: new distance and angle (binary encoded)

    """
    def isPowerOfTwo(x):
        return (x and (not(x & (x - 1))) )

    # dist
    assert isPowerOfTwo(b)
    m = max(dist) / b
    new_dist = []
    for d in dist:
        bin = int(d / m)
        if bin >= b: bin = b - 1
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_dist.append(bin)
    
    # angle
    amplitude = 360 / b
    new_angle = []
    for a in angle:
        bin = (a - amplitude / 2) 
        bin = int(bin / amplitude)
        bin = [int(x) for x in list('{0:0b}'.format(bin))]
        while len(bin) < sqrt(b): bin.insert(0, 0)
        new_angle.append(bin)

    return torch.cat([torch.tensor(new_dist, dtype=torch.float32), torch.tensor(new_angle, dtype=torch.float32)], dim=1)

def to_bin2(d1,s1,c1,d2,s2,c2):
    return torch.cat([torch.tensor(d1, dtype=torch.float32).unsqueeze(1), 
                      torch.tensor(s1, dtype=torch.float32).unsqueeze(1),
                      torch.tensor(c1, dtype=torch.float32).unsqueeze(1),
                      torch.tensor(d2, dtype=torch.float32).unsqueeze(1),
                      torch.tensor(s2, dtype=torch.float32).unsqueeze(1),
                      torch.tensor(c2, dtype=torch.float32).unsqueeze(1)
                      ], dim=1)

import re

from nltk.tokenize import word_tokenize
from price_parser import Price
import dateparser

def remove_stop_words(text):
    "remove_stop_words from payment description"
    stop_words = [
        "pay",
        r"\w*payment\w*",
        r"\w*payable\w*",
        r"\w*paymnt\w*",
        "ccd",
        "ctx",
        "ppd",
        "ach",
        r"transfer\w*",
        "corp",
        "corporation",
        "inc",
        "llc",
        "pmt",
        r"\w*pymnt\w*",
        ",",
        "\.",
        "\)",
        "\(",
        "-",
        "com",
    ]

    for word in stop_words:
        matched = re.search(rf"\b{word}\W*(\b|$)", text)
        while matched:
            start, stop = matched.span()
            text = text[0:start] + " " + text[stop:]
            matched = re.search(rf"\b{word}\W*(\b|$)", text)

    text = " ".join(text.strip().split())

    return text


def get_mask_tokens(text, mask, min_length=0):
    "find tokens in text that match with mask"
    candidates = list(
        {
                token
                for token in word_tokenize(text.lower())
                if re.match(mask, token)
                and re.match(mask, token).span() == (0, len(token))
        }
    )

    candidates = [x for x in candidates if len(x)>min_length]

    return candidates


def find_amounts(text):
    text = text.lower().strip()
    currencies = r'(?:\$|€|£|¥|eur|usd|gbp|\u20AC|s|S)?'
    mask_1 = r'(([1-9]\d+|0)[,.]\d{1,3})' #0,12123; 0.321233
    mask_2 = r'[1-9]\d{0,2}(?:([,])\d{3})?(?:\1\d{3})*([.]\d{1,3})' #1,321,233.23
    mask_3 = r'[1-9]\d{0,2}(?:([.])\d{3})?(?:\1\d{3})*([,]\d{1,3})' #1,321,233.23
    amount_candidates_1 = set(get_mask_tokens(text, f'{currencies}-?{mask_1}'))
    amount_candidates_2 = set(get_mask_tokens(text, f'{currencies}-?{mask_2}'))
    amount_candidates_3 = set(get_mask_tokens(text, f'{currencies}-?{mask_3}'))
    amount_candidates = list(amount_candidates_1.union(amount_candidates_2).union(amount_candidates_3))
    
    if amount_candidates:
        return True
    return False


def find_codes(text):
    text = text.lower().strip()
    mask = r"#?([a-z]+-?)*\d+[-a-z\d]*"
    mt = re.match(mask, text)
    if mt:
        if mt.span() == (0, len(text)):
            return True
    return False


def find_numbers(text):
    text = text.lower().strip()
    
    return text.isdigit()


def find_word(text):
    text = text.lower().strip()
    
    mask = r'[a-zA-Z.,]+'
    if re.match(mask, text) and re.match(mask, text).span() == (0, len(text)):
        return True
    return False
        
        
def find_words(text):
    text = text.lower().strip()
    
    mask = r'[a-zA-Z :_.,&#/@-]+'
    if re.match(mask, text) and re.match(mask, text).span() == (0, len(text)):
        return True
    return False


def find_dates(s):
    s = s.lower().strip()
    chunks = re.findall(r'\d+|[A-Za-z]+|\W+', s)
    if len(chunks)>5:
        return False
        
    digit_chunks = [x for x in chunks if x.isdigit()]
    if len(digit_chunks)>3:
        return False
    
    word_chunks = [x for x in chunks if x.isalpha()]
    if len(word_chunks)>1:
        return False
    
    other_chunks = [x for x in chunks if not (x.isalpha() or x.isdigit())]
    if len(other_chunks)>2:
        return False
    
    for oc in other_chunks:
        if oc.strip() not in ['.','','/','-',',',' ']:
            return False
    
    if word_chunks:
        if word_chunks[0] not in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','january','february','march','april','may','june','july','august','september','october','november','december']:
            return False
        
        if len(digit_chunks)!=2:
            return False
        
        if int(digit_chunks[0]) not in range(1,32):
            return False
        
        if len(digit_chunks[1]) not in [1,2,4]:
            return False
    else:
        if len(digit_chunks)!=3:
            return False
        
        if int(digit_chunks[1]) not in range(1,32):
            return False
        
        second_day = 32
        if int(digit_chunks[1]) in range(13,32):
            second_day = 13
        
        if len(digit_chunks[0]) not in [1,2,4]:
            return False
        
        if len(digit_chunks[2]) not in [1,2,4]:
            return False
        
        if (len(digit_chunks[0])==4) and (len(digit_chunks[2])==4):
            return False
        
        if len(digit_chunks[0]) in [1,2]:
            if int(digit_chunks[0]) not in range(1,second_day):
                return False
        
        if len(digit_chunks[0])==4:
            if int(digit_chunks[2]) not in range(1,second_day):
                return False
            
    return True
