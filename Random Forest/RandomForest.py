import numpy as np
import random
n = 6
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
    information_gain = 0.05
    left, right = [], []
    threshold_and_feature = []
    for feature in range(7):
        sel_feature = feature
        for threshold in range(4):
            sel_threshold = threshold
            temp_left, temp_right = Decision_tree.split(data, feature, threshold)
            temp_ig = informationGain(data, temp_left, temp_right)
            if temp_ig > information_gain:
                threshold_and_feature = [sel_threshold, sel_feature]
                information_gain = temp_ig
                left = temp_left
                right = temp_right
    print(threshold_and_feature)
    return left,right,threshold_and_feature




    return(left, right)

def informationGain(parent, left, right):
    IGP = Gini_Impurity(parent)
    IGL = (len(left)/len(parent))*Gini_Impurity(left)
    IGR = (len(right)/len(parent))*Gini_Impurity(right)
    return(IGP-IGL-IGR)

class Decision_tree:

    def __init__(self, data, threshold= None,parent = None):
        self.left = None
        self.right = None
        self.data = data
        self.threshold_and_feature = threshold
        self.parent = parent
    def populate(self, layer = 0):
        print("LAYER: {} THRESHOLD/FEATURE: {}\n\n DATA: {}\n\n".format(layer,self.threshold_and_feature,self.data))
        if Gini_Impurity(self.data) == 0:
            return
        else:
            self.left = Decision_tree(best_split(self.data)[0],self)
            if len(self.left.data) == 0:
                self.left = None
            else:
                print("\n\n\n\n\n\n\n\n\n\n\n\n\n",type(best_split(self.data)[2]))
                self.threshold_and_feature = best_split(self.data)[2]
                self.left.populate(layer+1)
            self.right = Decision_tree(best_split(self.data)[1],self)
            if len(self.right.data) == 0:
                self.right = None
            else:
                self.threshold_and_feature = best_split(self.data)[2]
                self.right.populate(layer+1)
    def split(data, feature,  threshold):
        left, right = [], []

        for x in data:
            if x[feature] < threshold:
                left.append(x)
            else:
                right.append(x)
        return(left, right)
    def predict(self,val):
        if self == None:
            return(self.parent.data[0][-1])
        elif(self.threshold_and_feature == None):
            return(self.data[0][-1])
        else:
            if(val[self.threshold_and_feature[1]] > self.threshold_and_feature[0]):
                return self.right.predict(val)
            else:
                return self.left.predict(val)
class RandomForest:
    def __init__(self, data, sample_size, feature_subset): # hyperparameters d n
        self.data = data
        self.trees = []
        self.sample_size = sample_size
        self.feature_subset = feature_subset
    def random_subset(data,sample_size):
        return random.sample(data, sample_size)



    def grow_forest(self,tree_count): #hyperparameter k
        for _ in range(tree_count):
            data = RandomForest.random_subset(self.data, tree_count)
            tree = Decision_tree(data)
            tree.populate()
            self.trees.append(tree)
            print("tree {} grew!".format(_))
    def predict(self,val):
        avg = 0
        for tree in self.trees:
            avg+=tree.predict(val)
        avg/=len(self.trees)
        if avg>=.5:
            return 1
        else:
             return 0



train = np.genfromtxt("train.csv", delimiter = ",",dtype = "float32").tolist()[:20]
test = np.genfromtxt("test.csv", delimiter = ",",dtype = "float32").tolist()

tree = Decision_tree(train)
tree.populate()
with open("sample_submission.csv",'w') as record:
    record.write("id,solution\n")
    counter = 1
    for problem in test:
        record.write(str(counter)+","+str(tree.predict(problem))+"\n")
        counter+=1
