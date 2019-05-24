import tensorflow as tf
import numpy as np
import re
import basic_layers
import utils

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
	h1=basic_layers.fc_layer_bi(x,bi_function=basic_layers.binarize_zero,units=2048,activation=tf.nn.leaky_relu,name="h1")
	
	h2=basic_layers.fc_layer_bi(h1,bi_function=basic_layers.binarize_zero,units=2048,activation=tf.nn.leaky_relu,name="h2")
	
	h3=basic_layers.fc_layer_bi(h2,bi_function=basic_layers.binarize_zero,units=2048,activation=tf.nn.leaky_relu,name="h3")
	
	fc2=basic_layers.fc_layer_bi(h3,bi_function=basic_layers.binarize_zero,units=10,name="fc2")
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
test_iter=5
# training
sess.run(init)

test_acc_save=[]

restore_epoch=0
if restore_epoch>0:
	print("restoring from epoch "+str(restore_epoch)+"...")
	saver.restore(sess,"epoch_"+str(restore_epoch)+".ckpt")

for e in range(restore_epoch,max_epoch):
	utils.shuffle_together(x_train,y_hinge_train)
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
		accuracy=utils.acc(pred=pred,label=y_hinge_test)
		test_acc_save.append(accuracy)
		print("val accuracy: "+str(accuracy))
		np.save("./val_acc.npy",np.array(test_acc_save))
