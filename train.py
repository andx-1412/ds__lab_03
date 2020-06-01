import model
import data_reader
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()

#load graph
mlp = model.graph(data_reader.VOCAB_SIZE,hidden_size = 50)
loss,predict_label = mlp.buid_graph()
train = mlp.trainer(loss,0.1)


#load data
train_data_reader = data_reader.data_reader(data_reader.VOCAB_SIZE,128,'tf_idf.txt')


# train 
saver = tf.train.Saver() # object để save graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())     # khởi tạo cho các tensor variable 
    #saver.restore(sess,'saved_variable)            # restore lại weight
    step, MAX_STEP = 0,10
    while step < MAX_STEP:
        batch_data,batch_label = train_data_reader.next_batch()
        label_eval, loss_eval, _ = sess.run([predict_label,loss,train], feed_dict = {mlp.input : batch_data,mlp.output : batch_label})
        step+=1
    
    saver.save(sess,'saved_variable') # save lại graph   


