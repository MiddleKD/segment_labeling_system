import cv2

img = cv2.imread("./data/image/2.jpg")
cv2.namedWindow("temp", cv2.WINDOW_NORMAL)
cv2.resizeWindow("temp", 1200,1010)
cv2.imshow("temp", img)
while True:
    key = cv2.waitKey(1)
    if key == -1:
        continue
    print(key)
    if key == 27:  # ESC key
        cv2.destroyAllWindows()
        break
