# 2D Semantic Segmentation
# License John Lambert
# Stanford University

from scipy.misc import imread, imresize
import numpy as np
#from '/Applications/MATLAB_R2016a.app/extern/engines/python/build/lib/matlab/engine' 
import matlab.engine

import os.path
import time
import tensorflow as tf

import os, sys
import csv
import os.path

from sys import argv
from os.path import exists
import matplotlib.pyplot as plt
from datetime import datetime

import scipy.io
import h5py
import matplotlib


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def getImages( maxZSlice):
	#maxZSlice = 39
	dir_path = 'June_6_2011_415-5_Series008/'
	images = np.zeros( (maxZSlice , 1024, 1024, 3) )
	for idx in xrange( maxZSlice ):
		zSliceNumber = str( idx )
		if len( zSliceNumber ) == 1:
			zSliceNumber = '0' + zSliceNumber
		imageName = 'June_6_2011__415-5__Series008_z0%s_ch00.tif' % ( zSliceNumber )
		if os.path.isfile( os.path.join(dir_path, imageName ) ):
			imagePath = os.path.join( dir_path, imageName )
			testIm = imread( imagePath )
			#print testIm.shape
			images[idx] = imread( imagePath )

	images = images.astype( np.float32 )
	return images


def getDendriticSpineData():
	print 'Data will be retrieved:'
	Z_SLICE_DEPTH = 39
	IMAGE_SIZE = (1024,1024,3)
	NUM_CLASSES = 2
	#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 25
	#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 25 # I DUNNO
	#eng = matlab.engine.start_matlab()
	#labelsObject = eng.GenerateDendriticSpineVoxels() # should specify folders here, etc.
	# for property, value in vars(labelsObject).iteritems():
	# 	print property #, ": ", value
	#labelsPythonArray = getattr( labelsObject, '_data' )
	#python_type = getattr( labelsObject, '_python_type' )
	#sizeOfArray = getattr( labelsObject, '_size' )
	#startOfArray = getattr( labelsObject, '_start' )
	#labels = np.asarray( labelsPythonArray , dtype=np.float64 )
	#labels = np.frombuffer( labelsPythonArray, dtype=np.float64 ) 
	#print 'Buffered Labels shape', labels.shape
	#labels = np.reshape( labels, (1024,1024, Z_SLICE_DEPTH )) # labels is of shape 1024 x 1024 x 47
	f = h5py.File('/Users/johnlambert/Documents/Stanford_2015-2016/SpringQuarter_2015-2016/CS_231A/FinalProject/dendriticSpines.mat','r')
	data = f.get('voxels') # Get a certain dataset
	h5pyLabels = np.array(data)
	print 'h5pylabels shape', h5pyLabels.shape
	#rint 'Previous Labels shape', labels.shape
	h5pyLabels = np.transpose( h5pyLabels , ( 0, 2 , 1) ) # or should it be ( 2 , 1, 0 ) ? IS X OR Y FIRST?
	print 'h5pylabels shape', h5pyLabels.shape
	#labelsToPlot = labels[:,:,8]
	# labelsToPlot = h5pyLabels[:,:,8]
	# plt.subplot(1, 1, 1)
	# plt.imshow( labelsToPlot.astype('uint8') , cmap='Greys_r')
	# plt.axis('off')
	# plt.gcf().set_size_inches(10, 10)
	# plt.title( 'Segmentation GT')
	# plt.show()
	labels = h5pyLabels
	labels = np.reshape( labels, (Z_SLICE_DEPTH, 1024,1024, 1) ) # MAX_POOL needs batch, height, width, channels
	print 'New Labels shape', labels.shape
	images = getImages( Z_SLICE_DEPTH )
	labels = 1 - labels
	return images, labels



#_activation_summary(h_conv1)
# pool1
#bias = tf.nn.bias_add(conv, biases)
# norm1
#norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
# conv2
#biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
#_activation_summary(conv2)
# norm2
#norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
# pool2

# two dimensions are the patch size, the next is the number of input channels, 
# and the last is the number of output channels.
def inference( images ):
	print 'Running network forward, performing inference'
	with tf.variable_scope('conv1') as scope:
		weight_conv1 = weight_variable([7, 7, 3, 64]) # I use 32 Filters of size (3 x 3). Later try ( 5 x 5 )
		conv1_output = tf.nn.conv2d( images, weight_conv1 , strides=[1, 1, 1, 1], padding='SAME')
		
		bias_conv1 = bias_variable([64])
		conv1_output_wBias = tf.nn.bias_add( conv1_output , bias_conv1 )

		conv1_activation = tf.nn.relu( conv1_output_wBias )
	pool1 = tf.nn.max_pool( conv1_activation , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

	with tf.variable_scope('conv2') as scope:
		weight_conv2 = weight_variable([7, 7, 64, 2]) # kernel_size: 2 , stride: 2 #stddev=1e-4, wd=0.0)
		conv2_output = tf.nn.conv2d( pool1 , weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
		
		bias_conv2 = bias_variable([2])
		conv2_output_wBias = tf.nn.bias_add( conv2_output, bias_conv2 )

		conv2_activation = tf.nn.relu( conv2_output_wBias )
	pool2 = tf.nn.max_pool( conv2_activation , ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	return pool2


# try putting on fc layer at the end?
# or end with 2 filters as 2 class buckets?

# For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size] 
# and the dtype int64. Each label is an int in range [0, num_classes).
def computeLoss(logits, labels, batch_size ):
	batch_size = 256 * 256
	print 'computing loss'

	# NEED A 4-D TENSOR IN ORDER TO DO MAX-POOLING. (Z_SLICE_DEPTH, 1024,1024, 1)
	# `logits` must have the shape `[batch_size, num_classes]` and dtype `float32` or `float64`.
	logits = tf.reshape( logits, [batch_size, -1])
	labels = tf.cast(labels, tf.float32)
	firstPooledLabels = tf.nn.max_pool( labels, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME') # equivalent to 1,4,4,1 ?
	#pooledLabels2 = tf.nn.max_pool( pooledLabels1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# Sparse means we write the index of the class ( not one-hot)
	# Calculate the average cross entropy loss across the batch.
	# `labels` must have the shape `[batch_size]` and dtype `int32` or `int64`
	pooledLabels = tf.reshape( firstPooledLabels, [ batch_size ] ) # [ -1 ]
	pooledLabels = tf.cast( pooledLabels , tf.int64)

	# USE WEIGHTS
	# tf.nn.weighted_cross_entropy_with_logits(logits, targets, pos_weight, name=None)
	predictions = tf.nn.softmax( tf.cast( logits, 'float64') ) # logits is float32, float64. 2-D with shape [batch_size, num_classes].

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits, pooledLabels, name='cross_entropy_per_pixel_example')
	
	cross_entropy_sum = tf.reduce_sum(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_sum)
	#cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	#tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss'), predictions, firstPooledLabels



# def computeLoss( logits, labels, batch_size ): # weighted version
# 	batch_size = 256 * 256
# 	print 'computing loss'

# 	#NEED A 4-D TENSOR IN ORDER TO DO MAX-POOLING. (Z_SLICE_DEPTH, 1024,1024, 1)
# 	#`logits` must have the shape `[batch_size, num_classes]` and dtype `float32` or `float64`.
# 	logits = tf.reshape( logits, [batch_size, -1])
# 	labels = tf.cast(labels, tf.float32)
# 	firstPooledLabels = tf.nn.max_pool( labels, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME') # equivalent to 1,4,4,1 ?
# 	#pooledLabels2 = tf.nn.max_pool( pooledLabels1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 	#Sparse means we write the index of the class ( not one-hot)
# 	#Calculate the average cross entropy loss across the batch.
# 	#`labels` must have the shape `[batch_size]` and dtype `int32` or `int64`
# 	pooledLabels = tf.reshape( firstPooledLabels, [ batch_size ] ) # [ -1 ]
# 	pooledLabels = tf.cast( pooledLabels , tf.int64)

# 	#USE WEIGHTS
# 	predictions = tf.nn.softmax( tf.cast( logits, 'float64') ) # logits is float32, float64. 2-D with shape [batch_size, num_classes].

# 	num_labels = 2

# 	label_batch = tf.cast( pooledLabels , tf.int32 )
# 	sparse_labels = tf.reshape( label_batch, [-1, 1])
# 	derived_size = tf.shape(label_batch)[0]
# 	indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
# 	concated = tf.concat(1, [indices, sparse_labels])
# 	outshape = tf.pack([derived_size, num_labels])
# 	oneHotLabels = tf.sparse_to_dense( concated, outshape, 1.0, 0.0)


# 	cross_entropy = tf.nn.weighted_cross_entropy_with_logits( logits, oneHotLabels, 200, name='weighted_CE_per_pixel') # pos_weight = 2
# 	cross_entropy_sum = tf.reduce_sum(cross_entropy, name='cross_entropy')
# 	tf.add_to_collection('losses', cross_entropy_sum)

# 	# cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
# 	# tf.add_to_collection('losses', cross_entropy_mean)
# 	# The total loss is defined as the cross entropy loss plus all of the weight
# 	# decay terms (L2 loss).
# 	return tf.add_n(tf.get_collection('losses'), name='total_loss'), predictions, firstPooledLabels









def trainNetwork( total_loss, global_step ):
	print 'Training Network'
	#num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size

	lr = 0.000001
	train_op = tf.train.GradientDescentOptimizer( lr ).minimize( total_loss )
	return train_op

def sampleMinibatch( images, labels, batch_size=1 ):
	full_data_size = images.shape[0]
	mask = np.random.choice( full_data_size, batch_size )
	mask = [9] # for now
	print 'ImageSlice# ', mask
	minibatchImages = images[ mask ]
	minibatchLabels = labels[ mask ]
	return (minibatchImages, minibatchLabels)

def test_ConvNet():
	print 'Begin Test of ConvNet'
	
	with tf.Graph().as_default():
		train_dir = '/Users/johnlambert/Documents/Stanford_2015-2016/SpringQuarter_2015-2016/CS_231A/FinalProject'
		batch_size = 1
		max_epochs = 1000
		#with tf.Graph().as_default():
		global_step = tf.Variable( 0 , trainable = False )
		images, labels = getDendriticSpineData()
		# print images.shape
		# print images.dtype
		# print labels.shape
		# print labels.dtype
		inputX_placeholder = tf.placeholder( tf.float32, shape=[ batch_size, 1024, 1024, 3 ], name="imagesForSeg" )
		gtY_placeholder = tf.placeholder( tf.float64, shape=[ batch_size, 1024, 1024, 1 ], name="labelsForSeg")

		#minibatch = sampleMinibatch( images, labels, 1 )
		#minibatchImages, minibatchLabels = minibatch # this is one image

		logits = inference( inputX_placeholder )
		loss, predictions, pooledLabels = computeLoss( logits, gtY_placeholder, batch_size )
		train_op = trainNetwork( loss, global_step )
		#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		saver = tf.train.Saver(tf.all_variables())
		init = tf.initialize_all_variables()
		
		with tf.Session() as session:
			#session = tf.InteractiveSession()
			session.run(init)

			summary_writer = tf.train.SummaryWriter( train_dir, session.graph )

			for step in xrange( 1, max_epochs, 1 ):
				start_time = time.time()

				minibatch = sampleMinibatch( images, labels, 1 )
				minibatchImages, minibatchLabels = minibatch # this is one image

				#rescale from -1 to 1
				minibatchImages -= 127
				minibatchImages /= 255.0

				print 'minibatchImages.shape',minibatchImages.shape # ( should be 1 x 1024 x 1024 x 1)

				print 'Sum of all of the labels at full size', np.sum( minibatchLabels )
				#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
				feed = { inputX_placeholder: minibatchImages, gtY_placeholder: minibatchLabels  }
				loss_value , y_pred, logits_vals, pooledLabelsVals , _  = session.run( [ loss, predictions, logits, pooledLabels, train_op ], feed_dict=feed)
				print 'Sum of all of the labels at 1/4 by 1/4 size', np.sum( pooledLabelsVals )
				print 'y_pred.shape',y_pred.shape
				# print 'logits_vals.shape', logits_vals.shape
				# print
				# summary_str = sess.run(summary_op)
				# summary_writer.add_summary(summary_str, step)

				if step%10 == 0: # or %100
					imagesToPlot = np.reshape( minibatchImages , (1024,1024,3) )
					#plt.subplot(1, 1, 1)
					plt.imshow( imagesToPlot.astype('uint8') ) #.transpose(2,1,0) )
					#plt.axis('off')
					#plt.gcf().set_size_inches(10, 10)
					plt.rcParams["axes.titlesize"] = 8
					plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
					plt.rcParams['image.interpolation'] = 'nearest'
					plt.rcParams['image.cmap'] = 'gray'
					plt.gca().axis('off')
					plt.axis('off')
					plt.title( 'Segmentation Input Image')
					plt.savefig('input_image%d.png' % (step) )
					#plt.show()

					print 'minibatchLabels.shape',minibatchLabels.shape

					labelsToPlot = np.reshape( minibatchLabels , (1024,1024) )
					#plt.subplot(1, 1, 1)
					plt.imshow( labelsToPlot.astype('uint8') , cmap='Greys_r')
					plt.axis('off')
					plt.gcf().set_size_inches(10, 10)
					plt.title( 'Segmentation GT')
					plt.savefig('gt_label_Seg%d.png' % (step) )
					#plt.show()


					y_pred = np.argmax( y_pred, axis=1)
					y_pred = np.reshape( y_pred, (256,256) )
					#plt.subplot(1, 1, 1)
					plt.imshow( y_pred.astype('uint8') , cmap='Greys_r')
					plt.axis('off')
					plt.gcf().set_size_inches(10, 10)
					plt.title( 'Segmentation prediction')
					plt.savefig('segmentation_prediction%d.png' % (step) )

					pooledLabelsVals = np.reshape( pooledLabelsVals, (256,256) )
					#plt.subplot(1, 1, 1)
					plt.imshow( pooledLabelsVals.astype('uint8') , cmap='Greys_r')
					plt.axis('off')
					plt.gcf().set_size_inches(10, 10)
					plt.title( 'gt downsampled')
					plt.savefig('gt_Downsampled_max_pool_%d.png' % (step) )
					#plt.show()

				#if i%100 == 0:
				#     train_accuracy = accuracy.eval(feed_dict={
				#         x:batch[0], y_: batch[1], keep_prob: 1.0})
				#     print("step %d, training accuracy %g"%(i, train_accuracy))

				# print("test accuracy %g"%accuracy.eval(feed_dict={
				#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
				duration = time.time() - start_time
				assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
				#if step % 10 == 0:
				num_examples_per_step = 1 #FLAGS.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)

				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
				# if step % 100 == 0:
				# 	summary_str = session.run(summary_op)
				# 	summary_writer.add_summary(summary_str, step)
				# Save the model checkpoint periodically.
				if step % 1000 == 0 or (step + 1) == max_epochs: #FLAGS.max_steps:
					checkpoint_path = os.path.join( train_dir, 'model_5005epochs.ckpt')
					saver.save(session, checkpoint_path, global_step=step)


				# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
				# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
				# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
				# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				# sess.run(tf.initialize_all_variables())
				# for i in range(20000):
				#   batch = mnist.train.next_batch(50)
				#   if i%100 == 0:
				#     train_accuracy = accuracy.eval(feed_dict={
				#         x:batch[0], y_: batch[1], keep_prob: 1.0})
				#     print("step %d, training accuracy %g"%(i, train_accuracy))
				#   train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

				# print("test accuracy %g"%accuracy.eval(feed_dict={
				#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



if __name__ == "__main__":
    test_ConvNet()