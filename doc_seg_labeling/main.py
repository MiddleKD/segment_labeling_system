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

        write_label_image(labeled_image, file_name=os.path.join(args.save_path, os.path.basename(image_path)))
        write_line_json(json_like, json_file_name=os.path.join(args.save_path, "label.json"))

    return key_state


def main(args):
    fns = sorted(os.listdir(args.data_path), key= lambda x:int(x.split(".")[0]))

    labeler = Labeler()

    resume_idx = -1
    for idx, fn in enumerate(fns):
        if fn == args.resume_fn:
            resume_idx = idx
        
        if resume_idx == -1:
            continue
        
        image_path = os.path.join(args.data_path, fn)

        key_state = process(labeler, image_path, args)

        with open("last_labeld.txt", mode="w", encoding="utf-8") as f:
            f.write(f"{idx}, {fn}")

        if key_state == "quit":
            print(f"last labeld index: {idx} / file name: {fn}")
            break
        elif key_state == "pass":
            pass


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labeling system for document layout segmentation")
    
    parser.add_argument('--data_path', default='./data/image', type=str, help='Image data directory path')
    parser.add_argument('--save_path', default='./data/label', type=str, help='The direcory path which is you want to save results')
    parser.add_argument('--save_mode', default='both', type=str, help='Choose save file type (image, json, both)')
    parser.add_argument('--resume_fn', default="0.jpg", type=str, help='Resume image file name')
    
    args = parser.parse_args()

    main(args)