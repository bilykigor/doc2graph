import io
import requests
import base64
import json
import fitz
import uuid
import awswrangler as wr
import numpy as np
from time import sleep
import boto3
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from difflib import SequenceMatcher

center = lambda rect: ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)

def find_segment_by_image(source_crop, 
                 target_image, 
                 threshold = 0.85,
                 text_threshold = 0.8):
    """Finds source_crop image in target_image."""
    
    source_crop_cv2 = cv2.cvtColor(np.array(source_crop), cv2.COLOR_RGB2BGR)
    target_image_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    
    w, h = source_crop_cv2.shape[:-1]
    
    source_crop_cv2 = source_crop_cv2[:, :, :3]
    target_image_cv2 = target_image_cv2[:, :, :3]
    
    res = cv2.matchTemplate(target_image_cv2, source_crop_cv2, cv2.TM_CCORR_NORMED)
    _,score,_,point = cv2.minMaxLoc(res)
    
    found_box = None
    
    if score<threshold:
        print('Segment not found')
        #plt.imshow(Image.fromarray(source_crop_cv2))
        return None
    
    loc = [point]
    
    if len(loc)==0:
        print('Segment not found')
        #plt.imshow(Image.fromarray(source_crop_cv2))
        return None
    
    if len(loc)>1:
        print('Segment found multiple times')
        #print(loc)
        #plt.imshow(Image.fromarray(source_crop_cv2))
        return None
    
    #print('Segment found')
    pt = loc[0]
    tupleOfTuples = (pt, (pt[0] + h, pt[1] + w))
    found_box = list(sum(tupleOfTuples, ()))
    
    # Compare text in source and target images 
    res = pytesseract.image_to_data(source_crop, lang='eng', config='--psm 7',output_type=pytesseract.Output.DICT)
    source_crop_text = ' '.join(res['text']).strip()
     
    res = pytesseract.image_to_data(target_image.crop(found_box), lang='eng', config='--psm 7',output_type=pytesseract.Output.DICT)
    static_anchor_target_text = ' '.join(res['text']).strip()
    
    s = SequenceMatcher(None, source_crop_text, static_anchor_target_text)
    
    if s.ratio()<text_threshold:
        print(f'Text not match: {source_crop_text} -> {static_anchor_target_text}', s.ratio())
        return None
    
    print('IMAGES', source_crop_text,'--->',static_anchor_target_text)
           
    return found_box


def get_word_boxes(image_path, host, aws=None, n_tries=1):
    pil_image = file_to_images(image_path)[0]
    width, height = pil_image.size
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format='jpeg')
        image_bytes = buffer.getvalue()
        
    data = {
        "image_bytes":base64.b64encode(image_bytes).decode("utf8"),
    }
    
    if aws:
        data['aws'] = aws
        
    boto3_session = boto3.Session(**aws['credentials'])

    request_id = str(uuid.uuid4().hex)
    print(f'request_id: {request_id}| Requesting lambda')  
    
    data['aws']['output_file'] = f'tmp/{request_id}.json'
    
    n_tried=0
    while True:
        try:
            response = requests.post(host,
                            data=json.dumps(data),
                            headers={
                                'content-type':'application/json',
                                'x-amzn-RequestId':request_id
                                },#image/jpg
                            timeout=29
                            )
            msg = response.content.decode("UTF-8")
        except requests.exceptions.ReadTimeout:
            response = requests.Response()
            response.status_code = 504
            msg = 'timeout'
        
        if response.status_code!=200:
            if response.status_code==504:
                #print(msg)
                n_tried+=1
                
                if n_tried>=n_tries:
                    # print(f'Reached {n_tries} n_tries')
                    
                    print(f'Looking to download {data["aws"]["output_file"]}')
                    exception = ''
                    for _ in range(12):
                        sleep(10)
                        
                        try:
                            res = wr.s3.read_json(f's3://{aws["bucket_name"]}/{data["aws"]["output_file"]}', boto3_session=boto3_session)
                            res={
                                'words':res['words'].to_list(),
                                'boxes':res['boxes'].to_list(),
                                }
                            
                            res['boxes'] = [unnormalize_box(bbox, width, height) for bbox in res['boxes']]
    
                            return res
                        except Exception as exp:
                            exception = str(exp)
                            pass
                        
                    print(f'get_words_boxes_paddle| Failed to read results from s3 {exception}')
                    return None
                    
                sleep(90)
                continue
            
            print(f'get_words_boxes_paddle| {msg}')
            
            return None
        
        break
    
    ict_str = response.content.decode("UTF-8")
    res = json.loads(ict_str)
    
    res['boxes'] = [unnormalize_box(bbox, width, height) for bbox in res['boxes']]

    return res 


def get_intersection_area(box1, box2):
    """Returns intersection of two rectangles."""
    if box1[3]<=box2[1]:
        return 0.0
    
    if box2[3]<=box1[1]:
        return 0.0
    
    if box1[2]<=box2[0]:
        return 0.0
    
    if box2[2]<=box1[0]:
        return 0.0
    
    result = [max(box1[0],box2[0]),max(box1[1],box2[1]),min(box1[2],box2[2]),min(box1[3],box2[3])]
    
    area = float((result[2]-result[0])*(result[3]-result[1]))
    
    return area


def find_intersected(target_box, boxes, threadhold=0.0):
    """looks for intersected boxes with target_box"""
    matched_boxes=[]
    
    try:
        for box in boxes:
            area_a = (box[2]-box[0])*(box[3]-box[1])
            area_b = (target_box[2]-target_box[0])*(target_box[3]-target_box[1])
            area = get_intersection_area(box,target_box)/min([area_a,area_b])
            if area>threadhold:
                matched_boxes.append((box,area))
    except Exception as exception:
        print(f'Error find_intersected: {exception}')
        print(target_box, boxes)
    
    matched_boxes = sorted(matched_boxes, key=lambda tup: tup[1])
    
    return matched_boxes


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
        if min(rect_src[3], rect_dst[3]) <= max(rect_dst[1], rect_src[1]):
            return 0
        
        rect_dst[0]=rect_src[0]
        rect_dst[2]=rect_src[2]
        
        w = rect_dst[2] - rect_dst[0]
        h = min(rect_src[3], rect_dst[3]) - max(rect_dst[1], rect_src[1])
            
        res = w*h
    else:
        if min(rect_src[2], rect_dst[2]) <= max(rect_dst[0], rect_src[0]):
            return 0
        
        rect_dst[1]=rect_src[1]
        rect_dst[3]=rect_src[3]
        
        h = rect_dst[3] - rect_dst[1]
        w = min(rect_src[2], rect_dst[2]) - max(rect_dst[0], rect_src[0])
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
        print('Fail intersectoin_by_axis:',[rect_src,rect_dst])
        raise
    
    return area


def draw_boxes(image, boxes, color='green', width=2):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box,outline=color,width=2)#outline=color
    return image


# def draw_boxes(image, boxes_all, boxes, labels=None, links = None, scores = None, color='green', width=2):
#     draw = ImageDraw.Draw(image, "RGBA")
#     font = ImageFont.load_default()
    
#     if links:
#         for idx in range(len(links['src'])):
#             key_center = center(boxes_all[links['src'][idx]])
#             value_center = center(boxes_all[links['dst'][idx]])
#             draw.line((key_center, value_center), fill='violet', width=2)
            
#     if labels:
#         for box,label in zip(boxes,labels):
#             if color=='green':
#                 fill=(0, 255, 0, 127)
#             else:
#                 fill=(255, 0, 0, 127)
#             draw.rectangle(box, outline=(color), width=width,fill=fill)
#             text_position = (box[0]+10, box[1]-10)
#             text = str(label)
#             draw.text(text_position, text=text, font=font, fill=(255,0, 0)) 
#         if scores:
#             for box,label, score in zip(boxes,labels, scores):
#                 if color=='green':
#                     fill=(0, 255, 0, 127)
#                 else:
#                     fill=(255, 0, 0, 127)
#                 draw.rectangle(box, outline=(color), width=width,fill=fill)
#                 text_position = (box[0]+10, box[1]-10)
#                 text = '%s-%6.2f' % (label, score)
#                 draw.text(text_position, text=text, font=font, fill=(255,0, 0)) 
#     else:
#         for box in boxes:
#             if color=='green':
#                 fill=(0, 255, 0, 127)
#             else:
#                 fill=(255, 0, 0, 127)
#             draw.rectangle(box, outline=(color), width=width,fill=fill)
        
#     return image


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]
  
    
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]
    
    
def draw_graph(img, G, c = 'blue', nodes = None, blank = False, boxes = True):
    if blank:
        img = Image.new("RGB", img.size, "white")

    draw = ImageDraw.Draw(img)

    if boxes:
        for node in G.nodes():
            if nodes:
                if node not in nodes:
                    continue
            box = G.nodes[node]['box']
            draw.rectangle(box, outline=c, width=3)
    
    for edge in G.edges():
        if nodes:
            if edge[0] not in nodes:
                continue
            if edge[1] not in nodes:
                continue
            
        key_center = center(G.nodes[edge[0]]['box'])
        value_center = center(G.nodes[edge[1]]['box'])
        draw.ellipse((tuple(x-4 for x in key_center) + tuple(x+4 for x in key_center)), fill = 'red')
        draw.ellipse((tuple(x-4 for x in value_center) + tuple(x+4 for x in value_center)), fill = 'red')
        draw.line((key_center, value_center), fill='red', width=3)
        
    return img


def get_intersection(box1,box2):
    "returns intersection of two rectangles"
    if box1[3]<=box2[1]:
        return None
    
    if box2[3]<=box1[1]:
        return None
    
    if box1[2]<=box2[0]:
        return None
    
    if box2[2]<=box1[0]:
        return None
    
    result = [max(box1[0],box2[0]),max(box1[1],box2[1]),min(box1[2],box2[2]),min(box1[3],box2[3])]
    
    return result


def points_distance(p1,p2):
    delta_x = abs(p1[0]-p2[0])
    delta_y = abs(p1[1]-p2[1])
    return np.sqrt(delta_x*delta_x+delta_y+delta_y)