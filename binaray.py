import cv2

img = cv2.imread('original'
                 '.png', 0)
ret0, thresh0 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

opened2 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
ret1, thresh1 = cv2.threshold(opened2, 200, 255, cv2.THRESH_BINARY)
closed1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
ret2, thresh2 = cv2.threshold(closed1, 200, 255, cv2.THRESH_BINARY)

# cv2.imshow('image', img)
# cv2.imshow('image', thresh1)

cv2.imshow('image', thresh0)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()

