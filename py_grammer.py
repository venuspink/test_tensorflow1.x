import numpy as np
import tensorflow as tf

a = 0.878

# print(round(a * 100, 2))

single_test_x = [[90, 266, 30, 44, 10, 1, 32, 20, 98, 9]]

arr = np.array(single_test_x)



# print(arr[:,:-1])


pred = [False, True, True]

ppp = tf.cast(pred, tf.int32)


aaa = tf.reduce_mean(tf.cast(pred, tf.float32))
print("aaa",aaa)

sess = tf.Session()

ar = aaa.eval(session=sess)

print(ar)



obp = (75+17+1+18+54)/(279+54)

slg = (75+(2*17)+(3*1)+(4*18))/279

ops = obp + slg

print(round(obp, 3), round(slg, 3), round(ops,3))
