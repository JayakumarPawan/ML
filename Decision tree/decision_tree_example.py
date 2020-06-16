import numpy as np
'''
First attempt: .9423 using .01
Second attempt: .98076 using .05
'''
def split(data, feature,  threshold):
    left, right = [], []

    for x in data:
        if x[feature] < threshold:
            left.append(x)
        else:
            right.append(x)
    return(left, right)
def Gini_Impurity(data):
    labels = [i[-1] for i in data]
    class_distribution = [labels.count(0), labels.count(1)]
    num_samples = sum(class_distribution)
    if num_samples == 0:
        return(0)
    total = 0
    for x in class_distribution:
        ratio = x/num_samples
        total+= ratio**2
    return 1 - total
def best_split(data):
    information_gain = 0
    left, right = -1, -1
    for feature in range(len(data[0])-1):
        sel_feature = feature
        for threshold in range(1,11):
            sel_threshold = threshold
            temp_left, temp_right = split(data, feature, threshold)
            temp_ig = informationGain(data, temp_left, temp_right)
            if temp_ig > information_gain:
                threshold_and_feature = [sel_feature,sel_threshold]
                information_gain = temp_ig
                left = temp_left
                right = temp_right
    if left == -1:
        return None
    return left,right,threshold_and_feature
def informationGain(parent, left, right):
    IGP = Gini_Impurity(parent)
    IGL = (len(left)/len(parent))*Gini_Impurity(left)
    IGR = (len(right)/len(parent))*Gini_Impurity(right)
    return(IGP-IGL-IGR)
class Decision_tree:

    def __init__(self, data, threshold):
        self.left = None
        self.right = None
        self.data = data
        self.threshold_and_feature = threshold
    def populate(self, data):
        if Gini_Impurity(data) == 0:
            print("impurity is low")
            return
        elif informationGain(self.data, best_split(self.data)[0], best_split(self.data)[1]) <.05:
            print("IG is low")
            return
        else:
            self.threshold_and_feature = best_split(self.data)[2]
            print("set")
            self.left = Decision_tree(best_split(self.data)[0],-1)
            self.left.populate(self.left.data)

            self.right = Decision_tree(best_split(self.data)[1],-1)
            self.right.populate(self.right.data)

    def toString(self,layer):
        print("layer: ",layer,"   ",self.data[0],self.threshold_and_feature)
        if(self.left != None):
            self.left.toString(layer+1)
        elif self.right != None:
            self.right.toString(layer+1)
    def predict(self,val):
        if(self.left == None and self.right == None):
            print("no more",self.data[0][-1])
            return(self.data[0][-1])
        elif(self.left == None):
            print("no left branch")
            if self.right.data[0][-1] == 0:
                return 1
            return 0
        elif(self.right == None):
            print("no right branch")
            if self.left.data[0][-1] == 0:
                return 1
            return 0
        else:
            if(val[self.threshold_and_feature[0]] > self.threshold_and_feature[1]):
                print("went right")
                return self.right.predict(val)
            else:
                print("went left")
                return self.left.predict(val)

''' mini example
data = [[1,0,0],[1,1,0],[2,2,1],[1,2,1],[3,0,1],[3,2,1],[3,1,1]]
a = Gini_Impurity(data)
c = informationGain(data,split(data,0,2)[0],split(data,0,2)[1])

tree = Decision_tree(data,-1)
tree.populate(data)
tree.toString(0)
a = tree.predict([1,0])
print(a) '''
#training = np.genfromtxt("training.csv", delimiter = ",",dtype = "int32").tolist()
training = np.genfromtxt("train.csv", delimiter = ",",dtype = "float32").tolist()
test = np.genfromtxt("test.csv", delimiter = ",",dtype = "float32").tolist()
tree = Decision_tree(training,-1)
tree.populate(training)
tree.toString(0)
print(tree.predict(test[0]))
#test = np.genfromtxt("testing.csv", delimiter = ",",dtype = "int32").tolist()
'''with open("submissionSample.csv",'w') as record:
    record.write("id,solution\n")
    counter = 1
    for problem in test:
        record.write(str(counter)+","+str(tree.predict(problem))+"\n")
        counter+=1'''
