import numpy
import matplotlib.pyplot
import scipy.special
input_nodes = 10
hidden_nodes = 5
output_nodes = 3
wih = numpy.random.normal(0.0, pow(10, -0.5), (5, 10))
print("-------wih------")
print(wih)
who = numpy.random.normal(0.0, pow(5, -0.5), (3, 5))
print("-------who------")
print(who)
#------------------
inputs_list=[0.01,0.01,0.01,0.01,0.87,0.37,0.99,0.01,0.01,0.67]
print("-------inputs_list------")
print(inputs_list)
targets_list=[0.01,0.99,0.01]
print("-------targets_list------")
print(targets_list)
inputs = numpy.array(inputs_list, ndmin=2).T
print("-------inputs------")
print(inputs)
targets = numpy.array(targets_list, ndmin=2).T
learning_rate = 0.2
print("-------targets------")
print(targets)
# calculate signals into hidden layer
hidden_inputs = numpy.dot(wih, inputs)
print("-------hidden_inputs------")
print(hidden_inputs)
 # calculate the signals emerging from hidden layer
hidden_outputs = scipy.special.expit(hidden_inputs)
print("-------hidden_outputs------")
print(hidden_outputs)     
# calculate signals into final output layer
final_inputs = numpy.dot(who, hidden_outputs)
# calculate the signals emerging from final output layer
print("-------final_inputs------")
print(final_inputs)
final_outputs = scipy.special.expit(final_inputs)
print("-------final_outputs------")
print(final_outputs)
# output layer error is the (target - actual)
output_errors = targets - final_outputs
print("-------output_errors------")
print(output_errors)
# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
print("-------who------")
print(who)
print("-------who.T------")
print(who.T)
hidden_errors = numpy.dot(who.T, output_errors) 
print("-------hidden_errors------")
print(hidden_errors)
tmp_errs=output_errors * final_outputs * (1.0 - final_outputs) 
print("-------tmp_errs------")
print(tmp_errs) 
tmp_outputs=numpy.transpose(hidden_outputs) 
print("-------tmp_outputs------")
print(tmp_outputs)  
# update the weights for the links between the hidden and output layers
who += learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
print("-------who------")
print(who)  
# update the weights for the links between the input and hidden layers
wih += learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
print("-------wih------")
print(wih) 