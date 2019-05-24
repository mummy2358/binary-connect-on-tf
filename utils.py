import tensorflow as tf
import numpy as np

def shuffle_together(arr1,arr2):
	c=list(zip(arr1,arr2))
	np.random.shuffle(c)
	arr_1,arr_2=zip(*c)
	return arr_1,arr_2

def acc(label,pred):
	# input predictions and labels and calculate accuracy
	correct=0
	for i in range(len(label)):
		if np.argmax(label[i])==np.argmax(pred[i]):
			correct+=1
	res=correct/len(label)
	return res

