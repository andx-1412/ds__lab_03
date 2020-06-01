# class bao gồm:  +  kiến trúc của  graph, 
#                 +  phương thức trả về loss để train
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()
NUM_CLASSES = 20 #  kích thước output của graph




class graph:
    def __init__(self, vocab_size, hidden_size): # lấy kích thước input( = với kích thước bộ từ điển khi xây dựng tf_idf) và số lớp ẩn của graph
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size          # số neuron ở lớp ẩn

    def buid_graph(self):   #tạo graph
        
        hidden_matrix1 = tf.get_variable(name = 'input_weight', shape = (self.vocab_size, self.hidden_size), initializer = tf.random_normal_initializer(seed = 2020) )
        bias_input = tf.get_variable(name = 'input_bias', shape  = (self.hidden_size),initializer = tf.random_normal_initializer(seed = 2020)  )
        hidden_matrix2 = tf.get_variable(name = 'hidden_weight', shape = (self.hidden_size, NUM_CLASSES), initializer = tf.random_normal_initializer(seed = 2020) )
        bias_hidden =  tf.get_variable(name = 'hidden_bias', shape  = (NUM_CLASSES),initializer = tf.random_normal_initializer(seed = 2020)  )


        self.input = tf.placeholder(tf.float32, shape = [None, self.vocab_size])
        self.output = tf.placeholder(tf.int32,shape = [None,1 ])           #số batch thay đổi
        hidden_layer = tf.matmul(self.input, hidden_matrix1) + bias_input
        hidden_layer  = tf.sigmoid(hidden_layer)                            #hidden layer sau khi đã qua hàm activation
        logit = tf.matmul(hidden_layer, hidden_matrix2) + bias_hidden       #ouput của graph
        label_one_hot = tf.one_hot(indices= self.output, depth = NUM_CLASSES, dtype = tf.float32) # tạo one-hot label dựa trên output nhập
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = label_one_hot, logits = logit)     # cross entropy sau khi đã tự softmax-> không cần layer softmax
        loss = tf.reduce_mean(loss)
        prob = tf.nn.softmax(logit)     #lớp softmax tính xác suất thuộc các lớp
        prob = tf.argmax(prob, axis = 1) # axis = 1 : lấy xác xuất lớn nhất theo hàng ngang(hòng dọc: batch size)
        predict_labels = tf.squeeze(prob) # gộp các output label từ các phần tử trong batch
        

        return loss, predict_labels


    def trainer(self,loss,lr):  # trả về cái để sess.run() để train graph
        return tf.train.AdamOptimizer(lr).minimize(loss)



