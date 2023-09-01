import cv2
import numpy as np

class Labeler:
    # Related initialize
    def __init__(self, color_list = None):
        self.initialize_state()

        if color_list == None:
            self.colors = [
                    [0, 0, 255],    # 0:icon 빨강색
                    [0, 255, 255],  # 1:title 노란색
                    [0, 255, 0],    # 2:text 초록색
                    [255, 0, 0],    # 3:table 파랑색
                    [255, 0, 255],  # 4:shape 자주색
                    [255, 255, 0],  # 5:image 하늘색
                                    # no_label:bg 하얀색
                ]
        else:
            self.colors = color_list

    def initialize_state(self):
        self.window_width = self.window_height = self.image_origin = self.image_components = self.image_processing = None
        self.LB_drawing = self.Shift_LB_drawing = self.Ctrl_LB_drawing = self.is_esc_pressed = False
        self.top_left_pt = self.bottom_right_pt = (-1,-1)
        self.selected_region = self.selected_components = []

        # target / label_map: {0:"icon", 1:"title", 2:"text", 3:"tabel", 4:"shape", 5:"image"}
        self.current_pixels_label = None


    # Main process
    def process_image_labeling(self, image_origin, image_components, window_width, window_height):
        self.initialize_state()

        self.image_origin = image_origin
        self.image_processing = image_origin.copy()
        self.image_components = image_components

        self.window_width, self.window_height = window_width, window_height

        self.current_pixels_label = np.zeros([*image_origin.shape[:2], len(self.colors)], dtype=np.int8)

        cv2.namedWindow('ImageViewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ImageViewer', self.window_width, self.window_height)
        cv2.setMouseCallback('ImageViewer', self.drag_event)
        cv2.imshow('ImageViewer', image_origin)

        while True:
            key = cv2.waitKey(1)

            if key == 27: # ESC
                self.is_esc_pressed = True
                self.turn_off_highlight(too_drag_box=False)

            elif key == 112: # p
                cv2.destroyAllWindows()
                return "pass"

            elif key == 113: # q
                cv2.destroyAllWindows()
                return "quit"

            else:
                if key == 48:   # 0 icon
                    self.label_pexels(0)
                elif key == 49: # 1 title
                    self.label_pexels(1)
                elif key == 50: # 2 text
                    self.label_pexels(2)
                elif key == 51: # 3 tabel
                    self.label_pexels(3)
                elif key == 52: # 4 shape
                    self.label_pexels(4)
                elif key == 53: # 5 image
                    self.label_pexels(5)
                elif key == 54: # 6 bg
                    self.label_pexels(6)

    # Related to mouse event
    def drag_event(self, event, x, y, flags, param):
        
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            if event == cv2.EVENT_LBUTTONDOWN and self.Shift_LB_drawing==False:
                self.Shift_LB_drawing = True
                self.top_left_pt = (x,y)
                self.turn_off_highlight(too_drag_box=True)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.Shift_LB_drawing==True:
                overlay = self.image_processing.copy()
                cv2.rectangle(overlay, self.top_left_pt, (x,y), (0,0,0), 2)
                image_while_draggin = self.apply_transparency(overlay,  alpha=0.5)
                cv2.imshow("ImageViewer", image_while_draggin)

            elif event == cv2.EVENT_LBUTTONUP and self.Shift_LB_drawing==True:
                self.Shift_LB_drawing = False
                self.is_esc_pressed = False
                self.bottom_right_pt = (x,y)

                self.top_left_pt = (min(self.top_left_pt[0], self.bottom_right_pt[0]), min(self.top_left_pt[1], self.bottom_right_pt[1]))
                self.bottom_right_pt = (max(self.top_left_pt[0], self.bottom_right_pt[0]), max(self.top_left_pt[1], self.bottom_right_pt[1]))

                self.visualize_pixels_label()
                self.turn_on_custom_highlight(inverse_thresh_binary=True)

        elif flags & cv2.EVENT_FLAG_CTRLKEY:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.Ctrl_LB_drawing = True
                self.top_left_pt = (x,y)
                self.turn_off_highlight(too_drag_box=True)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.Ctrl_LB_drawing==True:
                overlay = self.image_processing.copy()
                cv2.rectangle(overlay, self.top_left_pt, (x,y), (0,0,0), 2)
                image_while_draggin = self.apply_transparency(overlay,  alpha=0.5)
                cv2.imshow("ImageViewer", image_while_draggin)

            elif event == cv2.EVENT_LBUTTONUP and self.Ctrl_LB_drawing:
                self.Ctrl_LB_drawing = False
                self.is_esc_pressed = False
                self.bottom_right_pt = (x,y)

                self.top_left_pt = (min(self.top_left_pt[0], self.bottom_right_pt[0]), min(self.top_left_pt[1], self.bottom_right_pt[1]))
                self.bottom_right_pt = (max(self.top_left_pt[0], self.bottom_right_pt[0]), max(self.top_left_pt[1], self.bottom_right_pt[1]))

                self.visualize_pixels_label()
                self.turn_on_custom_highlight(inverse_thresh_binary=False)

        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.LB_drawing = True
                self.top_left_pt = (x,y)
                self.turn_off_highlight(too_drag_box=True)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.LB_drawing==True:
                overlay = self.image_processing.copy()
                cv2.rectangle(overlay, self.top_left_pt, (x,y), (0,0,0), 2)
                image_while_draggin = self.apply_transparency(overlay,  alpha=0.5)
                cv2.imshow("ImageViewer", image_while_draggin)

            elif event == cv2.EVENT_LBUTTONUP and self.LB_drawing:
                self.LB_drawing = False
                self.is_esc_pressed = False
                self.bottom_right_pt = (x,y)

                self.top_left_pt = (min(self.top_left_pt[0], self.bottom_right_pt[0]), min(self.top_left_pt[1], self.bottom_right_pt[1]))
                self.bottom_right_pt = (max(self.top_left_pt[0], self.bottom_right_pt[0]), max(self.top_left_pt[1], self.bottom_right_pt[1]))

                self.visualize_pixels_label()
                self.turn_on_component_highlight()


    # Related visualization
    def turn_on_component_highlight(self):

        overlay = self.image_processing.copy()
        cv2.rectangle(overlay, self.top_left_pt, self.bottom_right_pt, (0,0,0), 2)
        self.image_processing = self.apply_transparency(overlay,  alpha=0.5)

        cv2.imshow('ImageViewer', self.image_processing)

        selected_area = self.image_components[self.top_left_pt[1]:self.bottom_right_pt[1], self.top_left_pt[0]:self.bottom_right_pt[0]]
        unique_labels = np.unique(selected_area)
        
        drag_rect_mask = np.zeros(self.image_processing.shape[:2], dtype=np.int8)
        drag_rect_mask[self.top_left_pt[1]:self.bottom_right_pt[1], self.top_left_pt[0]:self.bottom_right_pt[0]] = 1
        self.selected_region = [drag_rect_mask]

        for label in unique_labels:
            if label != 0:
                overlay = self.image_processing.copy()
                overlay[self.image_components == label] = [255,127,0]
                self.image_processing = self.apply_transparency(overlay, alpha=0.5)

                self.selected_components.append(np.where(self.image_components==label,1,0))

        cv2.imshow('ImageViewer', self.image_processing)

    def turn_on_custom_highlight(self, inverse_thresh_binary=False):

        overlay = self.image_processing.copy()
        cv2.rectangle(overlay, self.top_left_pt, self.bottom_right_pt, (0,0,0), 2)
        self.image_processing = self.apply_transparency(overlay,  alpha=0.5)

        cv2.imshow('ImageViewer', self.image_processing)

        selected_area_contour_mask = self.apply_components_algorithm(self.image_origin, self.top_left_pt, self.bottom_right_pt, inverse_thresh_binary)
        # selected_area_contour_mask = self.apply_contour_algorithm(self.image_origin, self.top_left_pt, self.bottom_right_pt)

        drag_rect_mask = np.zeros(self.image_processing.shape[:2], dtype=np.int8)
        drag_rect_mask[self.top_left_pt[1]:self.bottom_right_pt[1], self.top_left_pt[0]:self.bottom_right_pt[0]] = 1
        self.selected_region = [drag_rect_mask]

        overlay = self.image_processing.copy()
        overlay[selected_area_contour_mask == 1] = [255,127,0]
        self.image_processing = self.apply_transparency(overlay, alpha=0.5)

        self.selected_components.append(selected_area_contour_mask.astype(np.uint8))
        cv2.imshow('ImageViewer', self.image_processing)
    

    def turn_off_highlight(self, too_drag_box = False):
        self.image_processing[:] = self.image_origin[:]

        self.selected_components = []
        if too_drag_box == False:
            cv2.rectangle(self.image_processing, self.top_left_pt, self.bottom_right_pt, (0,0,0), 2)
        else:
            self.selected_region = []
        
        cv2.imshow('ImageViewer', self.image_processing)

    def visualize_pixels_label(self):
        colors = self.colors
        for i in reversed(range(self.current_pixels_label.shape[-1])):
            overlay = self.image_processing.copy()
            overlay[self.current_pixels_label[:,:,i] == 1] = colors[i]
            self.image_processing = self.apply_transparency(overlay, alpha=0.5)

        cv2.imshow('ImageViewer', self.image_processing)

    def apply_transparency(self, overlay, alpha=0.5):
        return (self.image_processing * (1 - alpha) + overlay * alpha).astype(np.uint8)


    # Related update pixels label array
    def label_pexels(self, label):
        target_pixels = []

        if self.is_esc_pressed == True:
            target_pixels = self.selected_region
        else:
            target_pixels = self.selected_components

        if label == len(self.colors):
            for target_pixel in target_pixels:
                self.current_pixels_label -= np.repeat(target_pixel[:,:,np.newaxis], len(self.colors), axis=2)
                self.current_pixels_label = np.where(self.current_pixels_label<1, 0, 1)
        else:
            for target_pixel in target_pixels:
                self.current_pixels_label[:,:,label] += target_pixel
                self.current_pixels_label[:,:,label] = np.where(self.current_pixels_label[:,:,label]>0, 1, 0)

        self.is_esc_pressed = False
        self.top_left_pt = self.bottom_right_pt = (-1,-1)
        self.turn_off_highlight(too_drag_box=True)
        self.visualize_pixels_label()
    
    # Export pixels label array
    def export_pixels_label(self):
        return self.current_pixels_label
    

    # Algorithm
    def apply_components_algorithm(self, img, box_top_left=None, box_bottom_right=None, inverse_thresh_binary=True):    # bbox = [y1:y2,x1:x2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if (box_top_left != None) and (box_bottom_right != None):
            img_cropped = img_gray[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
        else:
            img_cropped = img_gray

        _, img_binary = cv2.threshold(img_cropped, 240, 255, cv2.THRESH_BINARY_INV if inverse_thresh_binary else cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated_image = cv2.dilate(img_binary, kernel, iterations=1)
        num_labels, image_components = cv2.connectedComponents(dilated_image)

        mask_cropped = np.zeros(img_cropped.shape)
        for label in range(num_labels):
            if label != 0:
                mask_cropped[image_components==label] = 1
        
        mask_total = np.zeros(img_gray.shape)
        mask_total[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]] = mask_cropped
        
        return mask_total

    def apply_contour_algorithm(self, img, box_top_left=None, box_bottom_right=None):    # bbox = [y1:y2,x1:x2]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if (box_top_left != None) and (box_bottom_right != None):
            img_cropped = img_gray[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
        else:
            img_cropped = img_gray

        _, img_binary = cv2.threshold(img_cropped, 140, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        
        mask_cropped = np.zeros(img_cropped.shape)
        cv2.drawContours(mask_cropped, contours, 0, 1, thickness=cv2.FILLED)
        
        mask_total = np.zeros(img_gray.shape)
        mask_total[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]] = mask_cropped

        return mask_total
