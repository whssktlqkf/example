import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
'''
x = np.array([10,20,15])
y = np.array([5,10,15])
plt.plot( x , y , 'b-o',label = 'Data 1' )
plt.ylabel('Y axls')
plt.xlabel('X axls')
plt.legend()
plt.show()
'''
'''
x = np.linspace(0, 2, 100)
y1 = 0.5 * x
y2 = 0.5 * x**2
y3 = 0.5 * x**3

plt.plot(x , y1, label = "linear")
plt.plot(x , y2, label = "quadratic")
plt.plot(x , y3, label = "cubic")

plt.legend()
plt.show()
'''
image = cv.imread('C:/Users/BigData/.vscode/example/sexy.png')

plt.figure()
plt.imshow(image)
plt.title("Original")
# plt.show()

rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.figure()
plt.imshow(image)
plt.title("RGB")
plt.show()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(gray, cmap = 'gray')
plt.title("GRAY")
plt.show()

blur = cv.blur(image,(50,50))
blur = cv.cvtColor(blur, cv.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(rgb)
plt.title("RGB")
plt.subplot(122)
plt.imshow(blur)
plt.title("BLur")

# plt.show()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(gray, cmap='gray')
plt.title("Gray")

edges = cv.Canny(gray, 100, 200)
plt.subplot(121)
plt.imshow(gray, cmap='Gray')
plt.title("Gray")
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")

plt.show()