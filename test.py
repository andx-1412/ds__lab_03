import model
import data_reader
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()




mlp = model.graph(data_reader.VOCAB_SIZE,hidden_size = 50)
loss,predict_label = mlp.buid_graph()



test_data_reader = data_reader.data_reader(data_reader.VOCAB_SIZE,128,'test_tf_idf.txt')
saver = tf.train.Saver()
with tf.Session() as sess:
    
    saver.restore(sess,'saved_variable')
    true_pred= 0
    while True:
        batch_data,batch_label = test_data_reader.next_batch()
        label_eval= sess.run([predict_label], feed_dict = {mlp.input : batch_data,mlp.output : batch_label})
        match = np.equal(batch_label, label_eval)
        for i in match:
            count = 0
            for j in i:
                if (j== True):
                    count+=1
            if(count == model.NUM_CLASSES ):
                true_pred+=1
        if(test_data_reader.start ==0):
            break

     
    print('accurate on test data:', true_pred/(test_data_reader.data.shape[0]))