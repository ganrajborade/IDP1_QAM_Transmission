import numpy as np
from matplotlib import pyplot as plt
img_array = np.load('binary_image.npy')

e1,a1 = 3, np.load('binary_image.npy')
plt.imshow(a1,'gray')
plt.xlabel("No.of pixels that are wrongly pointed at Eb/No="+str(-10)+" is "+str(e1))
plt.title("Received Image at Eb/No = "+str(-10)+"dB")
plt.show()