import numpy as np

train_data = np.loadtxt("data_train.csv", 
                        delimiter=",")
test_data = np.loadtxt("data_test.csv", 
                        delimiter=",")


#print(train_data[0])
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size




# transform labels into one hot representation

train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)


def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid


from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)
    
            
ANN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                    no_of_out_nodes = 10, 
                    no_of_hidden_nodes = 100,
                    learning_rate = 0.01)
    
    
for i in range(len(train_data)):
    ANN.train(train_data[i], train_labels_one_hot[i])
    
for i in range(10):
    res = ANN.run(test_data[i])

corrects, wrongs = ANN.evaluate(train_data, train_labels)
print("Accuracy train:{}% ".format( corrects / ( corrects + wrongs)*100))
corrects, wrongs = ANN.evaluate(test_data, test_labels)
print("Accuracy: test:{}%".format(corrects / ( corrects + wrongs)*100))
