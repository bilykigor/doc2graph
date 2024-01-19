from typing import List, Tuple,Union


def relative_box(anchor_point: Tuple[int, int, int, int], 
                 box: Tuple[int, int, int, int], 
                 is_relative=True)-> Tuple[int, int, int, int]:
    """Calculates relative coordinates of box with respect to anchor_point."""
    if is_relative:
        return [
            box[0]-anchor_point[0],
            box[1]-anchor_point[1],
            box[2]-anchor_point[0],
            box[3]-anchor_point[1]
                ]
    
    return [
        box[0]+anchor_point[0],
        box[1]+anchor_point[1],
        box[2]+anchor_point[0],
        box[3]+anchor_point[1]
        ]