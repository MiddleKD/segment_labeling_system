import cv2
import numpy as np
import json


"""
Related to json
"""

def to_polygon(two_dim_np):
    contours, _ = cv2.findContours(two_dim_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_list = []

    for cnt in contours:
        for point in cnt:
            x, y = point[0]
            polygon_list.extend([float(x), float(y)])

    return polygon_list

def to_json_like(labeled_tensor, image_file_name):
    json_like = {image_file_name:{"annotations":[]}}


    for label in range(labeled_tensor.shape[-1]):
        polygon_list = to_polygon(labeled_tensor[:,:,label])
        if len(polygon_list) == 0: continue

        json_like[image_file_name]["annotations"].append(
                {"segmentation":polygon_list, "category_id":label}
            )

    return json_like

def write_line_json(json_like, json_file_name):
    try:
        with open(json_file_name, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []
    except json.JSONDecodeError:
        existing_data = []
    
    existing_data.append(json_like)

    with open(json_file_name, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)



"""
Related to image
"""

# 인덱싱이 작을 수록 우선순위가 높은 레이블 (BGR)
colors_dominance = [
                [0, 0, 255],    # 0:icon 빨강색
                [0, 255, 255],  # 1:title 노란색
                [0, 255, 0],    # 2:text 초록색
                [255, 0, 0],    # 3:table 파랑색
                [255, 0, 255],  # 4:shape 자주색
                [255, 255, 0],  # 5:image 하늘색
                                # no_label:bg 하얀색
            ]


def to_label_image(labeled_tensor):
    background = np.ones((*labeled_tensor.shape[0:2], 3)) * 255
    
    for label in reversed(range(labeled_tensor.shape[-1])):
        label_mask = labeled_tensor[:,:,label]
        mask_indices = label_mask == 1
        background[mask_indices] = colors_dominance[label]

    return background

def write_label_image(labeled_image, file_name):
    cv2.imwrite(file_name+".jpg", labeled_image)

    