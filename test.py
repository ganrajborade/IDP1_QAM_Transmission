import numpy as np
from matplotlib import pyplot as plt

img_array = np.load('binary_image.npy')

b = img_array.reshape(-1)  #Converting 110x100 array to 1D-array of size 11000.
#print(b)



T = 10**(-6)  #T = 1 microseconds
fc = 2*(10**(6)) # Carrier frequency = 2MHz

fs = 50*(10**(6)) # Sampling rate of fs =  4MHz.
total_samples = fs*(0.0055) #The total number of samples of r(t) will be fs × 0.0055 = 22000, since the communication duration is 5.5 m sec.

t = np.linspace(0,5500*T,5501)
c = np.zeros((5500,50))
for i in range(0,5500):
	c[i] = np.linspace((i)*T,(i+0.999)*T,50) #each c[i] representing time range : i*T <= t <=(i+1)*T 
#print(c[0])


x = np.zeros(11000)  
for j in range(0,11000):
	if(b[j] == 0):
		x[j] = 1
	if (b[j] == 1):
		x[j] = -1


#print(x)

s = np.zeros((5500,50))
for i in range(0,5500):
	for j in range(0,50):
		s[i][j] = x[2*i]*np.cos(2*np.pi*fc*c[i][j])  + x[(2*i) + 1]*np.sin(2*np.pi*fc*c[i][j])
#print(s)
#print(s)
redefined_s = s.reshape(-1)
print("Transmitted Bit Sequence is",redefined_s)
#Take Eg = 1,therefore the value of Eb = Eg/2 i.e. Eb = 1/2
Eb = (10**(-6))/2
No = Eb/(10**(0.5)) #Considering last case in Eb /No : −10, −5, 0, 5 dB
#Producing AWGN channel :
#Noise variance = No/2

 

variance = fs*(No/2)

mu, sigma = 0, np.sqrt(variance) # mean and standard deviation
noise = np.random.normal(mu, sigma, int(total_samples))
noise_reformed = noise.reshape(5500,50)
r_reformed = s + noise_reformed
#print(r_reformed)  #	In this r_reformed[0] indicates sequence from r[0] to r[49].
#print(noise.shape)

r = redefined_s + noise
print("Received Bit Sequence is",r)

#print(r.shape)

# fs x T = 50 in this case . i.e., samples r[0], . . . , r[49] correspond to bits b 1 and b 2 ; the next 50 samples correspond to bits b 3 and b 4 , and so on
def distances(Array1,Array2):
	c = Array1 - Array2
	s = 0
	for i in range(Array1.size):
		s = s +(c[i])**2
	return np.sqrt(s)

sub_array = np.zeros((5500,5500))
Received_s = np.zeros(5500)
for i in range(0,5500):
	for j in range(0,5500):
		sub_array[i][j] = distances(r_reformed[i] ,s[j])
	Received_s[i] = np.argmin(sub_array[i])
print(Received_s)



#To be continued...