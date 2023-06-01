#Normal Data test

import numpy as np
import tensorflow as tf
import random

from matplotlib import pyplot as plt
from PIL import Image
from socketcan import CanRawSocket, CanFrame

interface_1 = "vcan0"
interface_2 = "vcan1"
s_1 = CanRawSocket(interface = interface_1)
s_2 = CanRawSocket(interface = interface_2)

#Discriminator load
model = tf.keras.models.load_model('HyperModel_wdsc.h5')

#Functions
def initializer():
	global first_arr, second_arr, third_arr, total
	first_arr, second_arr, third_arr, total = [], [], [], []

def transmitter():
	#Fuzzy
	#can_id = random.randrange(0,2018)

	#DoS
	can_id = 0x000
	data = bytes(range(0, 0x88, 0x11))
	frame = CanFrame(can_id = can_id, data = data)
	s_1.send(frame)
	
# make matrix 100x16x3
def make_img_data():
	Result = []
	for i in range(100):
		initializer()
		'''
		#DoS
		if(i%3!=0):
			transmitter()
			
		#Fuzzy
		transmitter()
		'''

		#Vectorization
		frame = s_2.recv()
		#print(hex(frame.can_id))
		if(len(str(hex(frame.can_id)))==4):
			data_1 = '0'
			data_2 = (str(hex(frame.can_id)))[2]
			data_3 = (str(hex(frame.can_id)))[3]
			
		elif(len(str(hex(frame.can_id)))==3):
			data_1 = '0'
			data_2 = '0'
			data_3 = (str(hex(frame.can_id)))[2]	
		else:
			data_1 = (str(hex(frame.can_id)))[2]
			data_2 = (str(hex(frame.can_id)))[3]
			data_3 = (str(hex(frame.can_id)))[4]			
		first_arr.append(tf.one_hot(int(data_1,16),16))		#3rd val of '0x---' = 1st val of hex
		second_arr.append(tf.one_hot(int(data_2,16),16))	#4th val of '0x---' = 2nd val of hex
		third_arr.append(tf.one_hot(int(data_3,16),16))		#5th val of '0x---' = 3rd val of hex
		
		
		total.append(first_arr)
		total.append(second_arr)
		total.append(third_arr) #list
		Result.append(np.array(total).T.reshape(16,3)) #Result: list
		
	
	Result_img = np.array(Result)
	Result = np.array(Result) * 255
	Result = np.expand_dims(Result, 0)
	print("Result shape: ", Result.shape)	
	return Result_img, Result

def Discriminator(img_data):
	condition = {"0" : "DoS", "1" : "Fuzzy", "2" : "Impersonation", "3" : "Normal"}
	pred = model.predict(img_data)
	index = np.argmax(pred)
	
	print("Class num: ", index,", Predict : ", condition.get(str(index)))
	return index
	
	
def main():
	scores = 0
	iteration = 0
	for i in range(100):
		iteration += 1
		img, img_data = make_img_data()
		if (Discriminator(img_data) == 3):
			scores += 1
		print("iteration: ", iteration)
		print("Accuracy: {:.2f}".format(float(scores / iteration) * 100),"(%)")
		#plt.imsave('img.png',img)
		#img = Image.open('img.png')
		#img.show()

if __name__ == "__main__":
	main()



















