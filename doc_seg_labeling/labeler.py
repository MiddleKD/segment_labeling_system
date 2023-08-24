import cv2
import numpy as np

class Labeler:
    # Related initialize
    def __init__(self, color_list = None):
        self.initialize_state()

        if color_list == None:
            self.colors = [
                    [0, 0, 255],    # 0:title 빨강색
                    [0, 255, 255],  # 1:icon 노란색
                    [0, 255, 0],    # 2:text 초록색
                    [255, 0, 0],    # 3:table 파랑색
                    [255, 0, 255],  # 4:image 자주색
                                    # no_label:bg 하얀색
                ]
        else:
            self.colors = color_list

    def initialize_state(self):
        self.window_width = self.window_height = self.image_origin = self.image_components = self.image_processing = None
        self.drawing = self.is_esc_pressed = False
        self.top_left_pt = self.bottom_right_pt = (-1,-1)
        self.selected_region = self.selected_components = []

        # target / label_map: {0:"title",1:"icon",2:"text",3:"tabel",4:"image"}
        self.current_pixels_label = None


    # Main process
    def process_image_labeling(self, image_origin, image_components, window_width, window_height):
        self.initialize_state()

        self.image_origin = image_origin
        self.image_processing = image_origin.copy()
        self.image_components = image_components

        self.window_width, self.window_height = window_width, window_height

        self.current_pixels_label = np.zeros([*image_origin.shape[:2], 5], dtype=np.int8)

        cv2.namedWindow('ImageViewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ImageViewer', self.window_width, self.window_height)
        cv2.setMouseCallback('ImageViewer', self.drag_event)
        cv2.imshow('ImageViewer', image_origin)

        while True:
            key = cv2.waitKey(1)

            if key == 27: # ESC
                self.is_esc_pressed = True
                self.turn_off_component_highlight(too_drag_box=False)

            elif key == 112: # p
                cv2.destroyAllWindows()
                return "pass"

            elif key == 113: # q
                cv2.destroyAllWindows()
                return "quit"

            else:
                if key == 48: # 0 title
                    self.label_pexels(0)
                elif key == 49: # 1 icon
                    self.label_pexels(1)
                elif key == 50: # 2 text
                    self.label_pexels(2)
                elif key == 51: # 3 tabel
                    self.label_pexels(3)
                elif key == 52: # 4 image
                    self.label_pexels(4)
                elif key == 53: # 5 bg
                    self.label_pexels(5)

    # Related to mouse event
    def drag_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.top_left_pt = (x,y)
            self.turn_off_component_highlight(too_drag_box=True)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
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
    
    def turn_off_component_highlight(self, too_drag_box = False):
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

        if label == 5:
            for target_pixel in target_pixels:
                self.current_pixels_label -= np.repeat(target_pixel[:,:,np.newaxis], 5, axis=2)
                self.current_pixels_label = np.where(self.current_pixels_label<1, 0, 1)
        else:
            for target_pixel in target_pixels:
                self.current_pixels_label[:,:,label] += target_pixel
                self.current_pixels_label[:,:,label] = np.where(self.current_pixels_label[:,:,label]>0, 1, 0)

        self.is_esc_pressed = False
        self.top_left_pt = self.bottom_right_pt = (-1,-1)
        self.turn_off_component_highlight(too_drag_box=True)
        self.visualize_pixels_label()


    
    # Export pixels label array
    def export_pixels_label(self):
        return self.current_pixels_label
    
