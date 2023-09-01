import cv2
from labeler import Labeler
from utils import *
import os

def load_image(image_path):
    img = cv2.imread(image_path)
    return img

def preprocess_img(img):
    img_origin = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a, binary_image = cv2.threshold(img_gray, 140, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_image = cv2.dilate(binary_image, kernel, iterations=3)

    num_labels, image_components = cv2.connectedComponents(dilated_image)
    return img_origin, image_components

def preprocess_img2(img):
    img_origin = img.copy()
    edges = cv2.Canny(img, 150, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     # Draw the largest contour
    #     edges = cv2.drawContours(edges, contour, -1, (0, 255, 0), 3)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # dilated_image = cv2.dilate(edges, kernel, iterations=3)

    # num_labels, image_components = cv2.connectedComponents(dilated_image)

    return img_origin, contours


def process(labeler, image_path, args):
    img = load_image(image_path)
    image_origin, image_components = preprocess_img(img)

    key_state = labeler.process_image_labeling(image_origin, image_components, window_width=1200, window_height=1010)
    if key_state == "quit":
        return key_state
    
    labeled_pixels_array = labeler.export_pixels_label()
    
    if args.save_mode == "image":
        labeled_image = to_label_image(labeled_pixels_array)
        write_label_image(labeled_image, file_name=os.path.join(args.save_path, os.path.basename(image_path)))

    elif args.save_mode == "json":
        json_like = to_json_like(labeled_pixels_array, image_file_name=os.path.basename(image_path))
        write_line_json(json_like, json_file_name=os.path.join(args.save_path, "label.json"))
    
    elif args.save_mode == "both":
        labeled_image = to_label_image(labeled_pixels_array)
        json_like = to_json_like(labeled_pixels_array, image_file_name=os.path.basename(image_path))

        write_label_image(labeled_image, file_name=os.path.join(args.save_path, os.path.basename(image_path).split('.jpg')[0]))
        write_line_json(json_like, json_file_name=os.path.join(args.save_path, f"{os.path.basename(image_path).split('.jpg')[0]}.json"))

    return key_state    # "pass"

import re
def main(args):
    def sorting_key(filename):
        match = re.match(r'(\d+)_(\d+)\.jpg', filename)
        if match:
            return tuple(map(int, match.groups()))
        return (0, 0)

    fns = sorted(os.listdir(args.data_path),key=sorting_key)

    labeler = Labeler()

    if "first start" == args.resume_fn:
        resume_idx = 0
    else:
        resume_idx = -1

    for idx, fn in enumerate(fns):
        if fn == args.resume_fn:
            resume_idx = idx
            continue
        if resume_idx == -1:
            continue
        
        image_path = os.path.join(args.data_path, fn)

        key_state = process(labeler, image_path, args)

        if key_state == "quit":
            print(f"Successfully Quit! {idx} / file name: {fn} doesn't saved")
            break
        elif key_state == "pass":
            with open("last_labeld.txt", mode="w", encoding="utf-8") as f:
                f.write(f"{args.data_path}, {idx}, {fn}")
            pass


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labeling system for document layout segmentation")
    
    parser.add_argument('--data_path', default='./data/image', type=str, help='Image data directory path')
    parser.add_argument('--save_path', default='./data/label', type=str, help='The direcory path which is you want to save results')
    parser.add_argument('--save_mode', default='both', type=str, help='Choose save file type (image, json, both)')
    parser.add_argument('--resume_fn', default="first start", type=str, help='Resume image file name')
    
    args = parser.parse_args()

    main(args)