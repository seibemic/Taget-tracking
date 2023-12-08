from PIL import Image
import matplotlib.pyplot as plt

# reading png image  file
img = Image.open('plane.png')

# resizing the image
img.thumbnail((50, 50))
#img=img.crop((50,50,50,50))
imgplot = plt.imshow(img)
plt.show()