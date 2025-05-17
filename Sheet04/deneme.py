import cv2
import matplotlib.pyplot as plt

ball = cv2.imread("./ball.png")
coffee = cv2.imread("./coffee.png")

ballGray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY) #converting bgr to rgb
coffeeGray = cv2.cvtColor(coffee, cv2.COLOR_BGR2GRAY)

cv2.imshow("sd", ballGray)
cv2.waitKey(0)