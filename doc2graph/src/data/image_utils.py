import io
import requests
import base64
import json
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

center = lambda rect: ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)

def get_word_boxes(image_path, host):
    pil_image = file_to_images(image_path)[0]
    width, height = pil_image.size
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format='jpeg')
        image_bytes = buffer.getvalue()
        
    data = {
        "image_bytes":base64.b64encode(image_bytes).decode("utf8")
    }

    response = requests.post(f"{host}",
                    data=json.dumps(data),
                    headers={'content-type':'application/json',
                            'x-amzn-RequestId': '84cad557-a68f-45db-9c01-79449f0aeecb'},#image/jpg
                    timeout=29
                    )
    
    ict_str = response.content.decode("UTF-8")
    res = json.loads(ict_str)
    
    res['boxes'] = [unnormalize_box(bbox, width, height) for bbox in res['boxes']]
    
    return res


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


def draw_boxes(image, boxes_all, boxes, labels=None, links = None, scores = None, color='green', width=2):
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    
    if links:
        for idx in range(len(links['src'])):
            key_center = center(boxes_all[links['src'][idx]])
            value_center = center(boxes_all[links['dst'][idx]])
            draw.line((key_center, value_center), fill='violet', width=2)
            
    if labels:
        for box,label in zip(boxes,labels):
            if color=='green':
                fill=(0, 255, 0, 127)
            else:
                fill=(255, 0, 0, 127)
            draw.rectangle(box, outline=(color), width=width,fill=fill)
            text_position = (box[0]+10, box[1]-10)
            text = str(label)
            draw.text(text_position, text=text, font=font, fill=(255,0, 0)) 
        if scores:
            for box,label, score in zip(boxes,labels, scores):
                if color=='green':
                    fill=(0, 255, 0, 127)
                else:
                    fill=(255, 0, 0, 127)
                draw.rectangle(box, outline=(color), width=width,fill=fill)
                text_position = (box[0]+10, box[1]-10)
                text = '%s-%6.2f' % (label, score)
                draw.text(text_position, text=text, font=font, fill=(255,0, 0)) 
    else:
        for box in boxes:
            if color=='green':
                fill=(0, 255, 0, 127)
            else:
                fill=(255, 0, 0, 127)
            draw.rectangle(box, outline=(color), width=width,fill=fill)
        
    return image


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
    
    
def draw_graph(img, G, c = 'blue', nodes = None):
    draw = ImageDraw.Draw(img)

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