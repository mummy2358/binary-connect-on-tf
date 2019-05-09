#######################
# binary connect implementation on tensorflow
# by: mym
# time :
# environments:
# python3, tensorflow 1.12.0
# dirty code version
#######################


import tensorflow as tf
import numpy as np
import re


# "_bi" means with binarized kernel
def binarize_zero(inputs,name='bi0'):
	# binarize given variables by sign; >=0 and <0
	logical=tf.cast(tf.greater(-inputs,0),tf.float32)
	return tf.add(1-logical,-logical,name=name)

def binarize_stocastic(inputs,name='bi0'):
	# binarize given variables by hard-sigmoid clip((1+x)/2,0,1) in stacastic way
	randp=tf.random.uniform(inputs.get_shape(),0,1)
	p=tf.clip_by_value((inputs+1)/2,0,1)
	logical=tf.cast(tf.greater(randp,p),tf.float32)
	return tf.add(-logical,1-logical,name=name)
	
	
def fc_layer_bi(inputs,bi_function=binarize_zero,units=100,W_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="fc0"):
	# save the first rank and flatten all the rest
	# be sure that inputs have at least rank of 2
	rest=1
	for m in inputs.get_shape()[1:]:
		rest=rest*m.value
	flat=tf.reshape(inputs,shape=[-1,rest])
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		if W_init:
			W=tf.get_variable(initializer=W_init,dtype=tf.float32,name="W")
		else:
			W=tf.get_variable(initializer=tf.random.truncated_normal(shape=[rest,units],stddev=stddev),dtype=tf.float32,name="W")
		if use_bias:
			if bias_init:
				b=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="b")
			else:
				b=tf.get_variable(initializer=tf.random.truncated_normal(shape=[units],stddev=stddev),dtype=tf.float32,name="b")
		
		# added binarization layer for W
		W_bi=bi_function(W,name="W_bi")
		res=tf.matmul(flat,W_bi)
		res=tf.nn.bias_add(res,b)
		if activation:
			res=activation(res)
		return res

def conv2d_layer_bi(inputs,bi_function=binarize_zero,kernel_size=3,filters=64,strides=[2,2],kernel_init=None,stddev=1e-3,use_bias=True,bias_init=None,activation=None,name="conv0"):
	# kernel_size and filters are both integers
	# inputs is a tensor of [batch,height,width,channels]
	with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
		# auto_reuse also reuse scope
		if not kernel_init:
			kernels=tf.get_variable(initializer=tf.random.truncated_normal(shape=[kernel_size,kernel_size,inputs.get_shape()[-1].value,filters],stddev=stddev),dtype=tf.float32,name="kernels")
		else:
			kernels=tf.get_variable(initializer=kernel_init,dtype=tf.float32,name="kernels")
		if use_bias:
			if not bias_init:
				bias=tf.get_variable(initializer=tf.random.truncated_normal(shape=[filters],stddev=stddev),dtype=tf.float32,name="bias")
			else:
				bias=tf.get_variable(initializer=bias_init,dtype=tf.float32,name="bias")
		# added binarization
		kernels_bi=bi_function(kernels,name="kernels_bi")
		res=tf.nn.convolution(inputs,kernels_bi,padding="SAME",strides=strides,name=name)
		res=tf.nn.bias_add(res,bias,data_format="NHWC")
		if activation:
			res=activation(res)
		return res

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

# [dataset]
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=np.reshape(x_train,newshape=[-1,28,28,1])
x_test=np.reshape(x_test,newshape=[-1,28,28,1])
x_train,x_test=x_train,x_test

classnum=10
identity=np.eye(classnum)
y_hinge_train,y_hinge_test=2*identity[y_train]-1,2*identity[y_test]-1

#x_train, x_test = x_train / 255.0, x_test / 255.0


## typical SGD manual implementation
with tf.variable_scope("mlp0") as scope:
	x=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1],name="x")
	y=tf.placeholder(dtype=tf.float32,shape=[None,classnum],name="y")
	
	#conv1=conv2d_layer_bi(x,bi_function=binarize_zero,kernel_size=3,filters=64,strides=[2,2],activation=None,name="conv1")
	#bn1=tf.layers.batch_normalization(conv1,axis=-1,center=False,scale=False,name="bn1")
	
	#conv2=conv2d_layer_bi(bn1,bi_function=binarize_zero,kernel_size=3,filters=64,strides=[2,2],activation=None,name="conv2")
	#bn2=tf.layers.batch_normalization(conv2,axis=-1,center=False,scale=False,name="bn2")
	
	#conv3=conv2d_layer_bi(bn2,bi_function=binarize_zero,kernel_size=3,filters=64,strides=[2,2],activation=None,name="conv3")
	#bn3=tf.layers.batch_normalization(conv3,axis=-1,center=False,scale=False,name="bn3")
	h1=fc_layer_bi(x,bi_function=binarize_stocastic,units=2048,activation=tf.nn.relu,name="h1")
	
	h2=fc_layer_bi(h1,bi_function=binarize_stocastic,units=2048,activation=tf.nn.relu,name="h2")
	
	h3=fc_layer_bi(h2,bi_function=binarize_stocastic,units=2048,activation=tf.nn.relu,name="h3")
	
	fc2=fc_layer_bi(h3,bi_function=binarize_zero,units=10,name="fc2")
	#loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=fc2))
	loss=tf.losses.hinge_loss(labels=y,logits=fc2)

varlist=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="mlp0")
#p1=re.compile("conv\d+/kernels")
p1=re.compile("h\d+/W")
p2=re.compile("fc\d+/W")

# varlist_bi is a varlist with some variables substituted by their binarized version
varlist_bi=[]
for i,v in enumerate(varlist):
	if p1.findall(v.name) or p2.findall(v.name):
		varlist_bi.append(tf.get_default_graph().get_tensor_by_name(v.name[:v.name.find(':')]+'_bi'+v.name[v.name.find(':'):]))
	else:
		varlist_bi.append(v)

grads=tuple(tf.gradients(loss,varlist_bi,name="gradients_bi"))
lr_buffer=1e-4
decay_factor=0.5
decay_iter=10
lr=tf.placeholder_with_default(lr_buffer,shape=None)

print("varlist_bi:")
for var_bi in varlist_bi:
	print(var_bi.name)

print("varlist")
updatesteps=[]
for i,var in enumerate(varlist):
	print(var.name)
	if "bi" in var.name:
		updatesteps.append(tf.assign(var,tf.clip_by_value(var-lr*grads[i],-1,1)))
		#updatesteps.append(tf.assign(var,var-lr*grads[i]))
	else:
		updatesteps.append(tf.assign(var,var-lr*grads[i]))
# end of model op definition

print('ok')
saver=tf.train.Saver()
sess=tf.Session()

init=tf.global_variables_initializer()

max_epoch=5000
batch_size=200
save_iter=10
test_iter=1
# training
sess.run(init)

restore_epoch=0
if restore_epoch>0:
	print("restoring from epoch "+str(restore_epoch)+"...")
	saver.restore(sess,"epoch_"+str(restore_epoch)+".ckpt")

for e in range(restore_epoch,max_epoch):
	shuffle_together(x_train,y_hinge_train)
	epoch_loss=0
	for b in range(np.shape(x_train)[0]//batch_size+1):
		if b!=np.shape(x_train)[0]//batch_size:
			batch_x=x_train[b*batch_size:(b+1)*batch_size]
			batch_y=y_hinge_train[b*batch_size:(b+1)*batch_size]
		else:
			batch_x=x_train[b*batch_size:-1]
			batch_y=y_hinge_train[b*batch_size:-1]
		# calculate grads(model not changed)
		grad_values=sess.run(grads,feed_dict={x:batch_x,y:batch_y})
		# weights updating
		sess.run(updatesteps,feed_dict={grads:grad_values,lr:lr_buffer})
		# calculate loss value
		batch_loss=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
		epoch_loss+=batch_loss
	epoch_loss=epoch_loss/np.shape(x_train)[0]
	print("epoch "+str(e+1)+": "+str(epoch_loss))
	if not (e+1)%decay_iter:
		lr_buffer=lr_buffer*decay_factor
		print("updated lr: "+str(lr_buffer))
	if not (e+1)%save_iter:
		print("saving model...")
		saver.save(sess,"./epoch_"+str(e+1)+".ckpt")
	if not (e+1)%test_iter:
		print("testing model...")
		pred=sess.run(fc2,feed_dict={x:x_test,y:y_hinge_test})
		accuracy=acc(pred=pred,label=y_hinge_test)
		print("test accuracy: "+str(accuracy))
