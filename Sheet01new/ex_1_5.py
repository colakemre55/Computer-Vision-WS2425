import cv2
import numpy as np

def display_image(window_name, img):
    """
    Displays image with the given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    display_image("Grayscale Image", gray_img)
    

    # sigma2 convolution
    img1 = cv2.GaussianBlur(gray_img, (5,5),2)
    img1New = cv2.GaussianBlur(img1, (5,5),2)
    
    # sigma 2sqr2 convolution
    img2 = cv2.GaussianBlur(gray_img, (5, 5), 2*np.sqrt(2))
    
    display_image("Convolution with sigma=2", img1New)
    display_image("Convolution with sigma=2√2", img2)
    
    # pixelwise diff
    difference = cv2.absdiff(img1New, img2)  
    display_image("pixelwise difference", difference)   
    error = np.max(difference)
    print(f"Maximum Pixel Error: {error}")

if __name__ == "__main__":
    main()
