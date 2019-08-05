import numpy
import matplotlib.pyplot
import scipy.special
#%matplotlib inline
datadir="D:\\project\\golang\\src\\deepLearnning\\dataset\\"
traindata="mnist_train_100.csv"
testdata="mnist_test_10.csv"
openfile=open(datadir+testdata,'r')
data_list=openfile.readlines()
openfile.close()
#print(len(data_list))
#print(data_list[0])
all_values=data_list[0].split(",")
image_list=numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_list,cmap='Greys',interpolation='None')
inputs_list=[[1,2],[3,4]]
inputs = numpy.array(inputs_list, ndmin=2).T
print(inputs)
#targets = numpy.array(targets_list, ndmin=2).T
tmp=numpy.transpose(inputs_list)
print("------")
print(tmp)
tmp1=scipy.special.expit(inputs_list)
print(tmp1)
print(pow(10,-0.5))