import numpy as np
from matplotlib import pyplot as plt
from pylab import *
img_array = np.load('binary_image.npy')

b = img_array.reshape(-1)  #Converting 110x100 array to 1D-array of size 11000.
#print("Transmitted bit sequence:",b)
T = 10**(-6)  #T = 1 microseconds
fc = 2*(10**(6)) # Carrier frequency = 2MHz

fs = 50*(10**(6)) # Sampling rate of fs =  4MHz.
total_samples = fs*(0.0055) #The total number of samples of r(t) will be fs × 0.0055 = 22000, since the communication duration is 5.5 m sec.


c = np.zeros((5500,50))  #c is the matrix of size(5500x50) i.e total number of elements = 275000
for i in range(0,5500):
    c[i] = np.linspace((i)*T,(i+0.999)*T,50) #each c[i] representing time range : i*T <= t <=(i+1)*T .i.e 1st row represents the array of time between 0 to T,2nd row represents array of time from T to 2T,and so on.
#print(c[0])


x = np.zeros(11000)   #(Am).g(t) for 4-QAM . Since (Am).g(t) is constant and Am(REAL AND IMAGINARY PART VALUES) values for 4-QAM are {-1,1}. So g(t) = 1. Basically Am values form rectangle with coordinates (1+j),(1-j),(-1-j),(-1+j).
for j in range(0,11000):
    if(b[j] == 0):
        x[j] = 1
    if (b[j] == 1):
        x[j] = -1

#print(x)

s = np.zeros((5500,50))   
for i in range(0,5500):
    for j in range(0,50):
        s[i][j] = x[2*i]*np.cos(2*np.pi*fc*c[i][j])  + x[(2*i) + 1]*np.sin(2*np.pi*fc*c[i][j])  #1st row is representing the transmitted waveform s(t) from time 0<=t<T,2nd row is representing the transmitted waveform s(t) from time T<=t<2T,and so on (In terms of discrete values.)
#print(s)

redefined_s = s.reshape(-1)  #Reshaping s (5500x50) array to show 275000 elements in a single row.
#print("Transmitted Waveform (in terms of discrete values) is given by :",redefined_s)
def MainFunction(b,s,redefined_s,EbNo):  # EbNo = Eb/No in dB

    #Take Eg = 1,therefore the value of Eb = Eg/2 i.e. Eb = 1/2
    Eb = (10**(-6))/2   # Eb = T/2
    No = Eb/(10**(EbNo/10)) #Because 10log(Eb/No) = -10,-5,0,5dB(Given to us).
    

    #Producing AWGN channel :
    #Noise variance = No/2
    variance = fs*(No/2)
    mu, sigma = 0, np.sqrt(variance) # mean and standard deviation
    noise = np.random.normal(mu, sigma, int(total_samples))
    noise_reformed = noise.reshape(5500,50)
    r_reformed = s + noise_reformed
    #print(r_reformed)  #	In this r_reformed[0] indicates sequence from r[0] to r[49].,r_reformed[1] indicates sequence from r[50] to r[99],and so on.
    #print(noise.shape)

    r = redefined_s + noise #Reshaping r_reformed (5500x50) array to show 275000 elements in a single row in which r[0] to r[49] are used to demodulate b0 and b1(because,we are starting index from 0),r[50] to r[99] are used to demodulate b2 and b3,and so on.
    #print("Received Waveform at Eb/No="+str(EbNo)+" (in terms of discrete values) is given by :",r)

    #print(r.shape)

    # fs x T = 50 in this case . i.e., samples r[0], . . . , r[49] correspond to bits b0 and b1 ; the next 50 samples correspond to bits b2 and b3 , and so on
    

    #Developing Minimum Distance Neighbour Decoding(ML DETECTION RULE):

    #Below function is used to calculate ||r_reformed - Sk||
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
        Received_s[i] = np.argmin(sub_array[i])   #print the index of the element which gives minimum ||r_reformed - Sj || (j varying from 0 to 5500)
    #print("Received_s :",Received_s)
    
    #Demodulating on the basis of ML detection rule:

    b_received = np.zeros(11000)
    j = 0
    for i in Received_s:
            b_received[2*j] = b[2*int(i)]
            b_received[(2*j)+1] = b[(2*int(i))+1]
            j= j+1

    #print("Received Bit Sequence at Eb/No="+str(EbNo)+":",b_received)
    image_array_received = b_received.reshape(110,100)  #For showing Image.
    #plt.imshow(image_array_received,'gray')
    #plt.title("Received Image at Eb/No = "+str(EbNo)+"dB")
    error_bits = b_received - b
    total_error_bits = 0
    for i in range(11000):
        if(error_bits[i] != 0):
            total_error_bits +=1
    print("No.of pixels that are wrongly pointed at Eb/No="+str(EbNo)+":",total_error_bits)
    return total_error_bits, image_array_received

    
subplot(2,2,1)
e1,a1 = MainFunction(b,s,redefined_s,-10)

print("--------------------------------------------------------------------------------------------")
plt.imshow(a1,'gray')
plt.xlabel("No.of pixels that are wrongly pointed at Eb/No="+str(-10)+" is "+str(e1))
plt.title("Received Image at Eb/No = "+str(-10)+"dB")

subplot(2,2,2)
e2,a2 = MainFunction(b,s,redefined_s,-5)
print("--------------------------------------------------------------------------------------------")
plt.imshow(a2,'gray')
plt.xlabel("No.of pixels that are wrongly pointed at Eb/No="+str(-5)+" is "+str(e2))
plt.title("Received Image at Eb/No = "+str(-5)+"dB")

subplot(2,2,3)
e3,a3 = MainFunction(b,s,redefined_s,0)
print("--------------------------------------------------------------------------------------------")
plt.imshow(a3,'gray')
plt.xlabel("No.of pixels that are wrongly pointed at Eb/No="+str(0)+" is "+str(e3))
plt.title("Received Image at Eb/No = "+str(0)+"dB")

subplot(2,2,4)
e4,a4 = MainFunction(b,s,redefined_s,5)
print("--------------------------------------------------------------------------------------------")
plt.imshow(a4,'gray')
plt.xlabel("No.of pixels that are wrongly pointed at Eb/No="+str(5)+" is "+str(e4))
plt.title("Received Image at Eb/No = "+str(5)+"dB")


plt.show()
