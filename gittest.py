import numpy as np 
img_array = np.load('binary_image.npy')

def modulator(data, M, code='gray', data_type='binary'):
	# Constants
	sqrt_M = np.sqrt(M).astype(int)
	k = np.log2(M).astype(int)

	# Binary to Gray code constelation convertor
	vect = np.array(range(sqrt_M))
	gray_constallation = np.bitwise_xor(vect, np.floor(vect/2).astype(int))

	# Gray code constelation to symbols convertor
	vect = np.arange(1, np.sqrt(M), 2)
	symbols = np.concatenate((np.flip(-vect, axis=0), vect)).astype(int)

	# Modulation
	if data_type == 'binary':
		# Data handling
		data_input = data.reshape((-1,k))
		I = np.zeros((data_input.shape[0],))
		Q = np.zeros((data_input.shape[0],))
		for n in range(int(data_input.shape[1] / 2)):
			I = I + data_input[:,n] * 2 ** n
		for n in range(int(data_input.shape[1]/2),int(data_input.shape[1])):
			Q = Q + data_input[:,n] * 2 ** (n - int(data_input.shape[1]/2))
	elif data_type == 'numbers':
		tmp = data / sqrt_M
		Q = np.floor(tmp)
		I = (tmp - Q) * 4
	else:
		return 0

	I = I.astype(int)
	Q = Q.astype(int)

	if code == 'gray':
		I = gray_constallation[I]
		Q = gray_constallation[Q]

	I = symbols[I]
	Q = symbols[Q]

	S = I + 1j * Q

	return S

Sm = modulator(img_array,4)
print(Sm.shape)