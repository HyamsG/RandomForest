__author__ = 'Gal Hyams, Anna Kutsela and Chen Lirz'
#python3
import gc 
import numpy as np
import random
import math
import re
import time
import operator

# Parameters
global percentage
percentage = 0.8 # lambda
global sigma0
sigma0 = 0
global n0
n0 = 5

global numOfBins
numOfBins = 10
global where_Y_starts #divide the matrix to fearutes and lables on this index
where_Y_starts = 72  #where y starts is 78-6 72
global thrForEmotions
thrForEmotions = 0.5
global the_ntree
the_ntree = 1
global the_mtry
the_mtry = 1 #changed when run


#all the praseing: only for the toy data
def parse_xy(data_matrix):#remove the first row from the matrix
    matrix_parsed= np.delete(data_matrix, (0), axis=0)

class data_matrix:

    def __init__(self,name, where_Y_starts):
        self.name=name
        self.where_Y_starts = where_Y_starts
        self.colnamesX, self.colnamesY = self.setcolnames(where_Y_starts)
        self.matrix=self.setmatrix(where_Y_starts)

    def setcolnames(self, where_Y_starts):
        colnames=(open(self.name,"r").read().split('\n')[0])#extract the first line
        colnames = re.split('\t',colnames) #split by tabs
        return colnames[:where_Y_starts], colnames[where_Y_starts:]

    def setmatrix(self, where_Y_starts):

        #colnum=len(self.colnames)

        data = np.loadtxt(self.name,skiprows=1)#get the data from the file,without the first column and the first row
        #data = np.matrix(data)
        dataX = data [:, 0:where_Y_starts] # the X part
        dataY = data[:, where_Y_starts:] # the Y part

        return (dataX, dataY)

class node:
    ''' a node of a decision tree'''
    def __init__(self, x_matrix, y_matrix, feature = None):
        self.feature = feature # feature is an integer
        self.threshold = 0 
        self.right = None # >= threshold
        self.left = None # < threshold
        self.x = x_matrix
        self.y = y_matrix

    def add_child(self, N):
        '''a full binary tree. each internal node has 2 children'''
        if self.left == None:
            self.left = N
        else:
            self.right = N

    def clear_matrix(self):
        '''
        remove the matrix of the node, once those are no linger needed. does so to release unused memory.
        '''
        self.x, self.y = [-1], [-1]

class tree:
    '''decision tree. using classes make the code more modular and readable'''
    def __init__(self, root):
        self.root = root
    #currently not in use#
    def get_dict_of_features(self):
        '''going over the tree inorder and filling up the dictionary.
        not saving the dict as a fild of the class, for not overloading the memmory'''
        dict = {} #the retured value. keys of the dictionary are feauters, and values are number of times is featurs had appeared.
        fill_tree_features_dict(self.root, dict)
        return dict

    def get_list_of_features(self):
        '''a list of integers. each cell in the list is sutable for a feature defined by this colmn at the matrix
        ot saving the dict as a fild of the class, for not overloading the memmory'''
        features_lst = np.zeros(self.root.x.shape[1]) # an array of len = number of features
        fill_tree_features_lst(self.root,features_lst)
        return features_lst

def fill_tree_features_lst(node, features_lst):
    '''inplace filling of the features list. each cell of the list is suitable for a specific feature'''
    if node != None: #in case the function was called on an ampty tree
        features_lst[node.feature] +=1
    if node.right != None: # if T has a 'right' child, he has a 'left' child as well
        fill_tree_features_lst(node.right, features_lst)
        fill_tree_features_lst(node.left, features_lst)

#not in use
def fill_tree_features_dict(node, dict):
    if node != None:
        if node.feature in dict:
            dict[node.feature] = dict[node.feature] + 1
        else:
            dict[node.feature] = 1
    if node.right != None: # if T has a 'right' child, he has a 'left' child as well
        fill_features_dict(node.right, dict)
        fill_features_dict(node.left, dict)

class forest:
    ''' a list of trees '''
    def __init__(self, treesList):
        self.treesList = treesList
        self.integer_to_feature_set = set() # to know which cell in the featers list sutable for wich feature.
        
    def add_tree(self,T):
        self.treesList.append(T)

    def get_list_of_features(self):
        ''' the features list is not saved as a filed in the class, for not overloading the memory
        returns a list. the index of a cell in the list is the index of the feature. the value in the cell in the list is the number of times this feature appeared '''
        if len(self.treesList) == 0:
            return np.zeros(0)
        features_lst = np.zeros(self.treesList[0].root.x.shape[1]) # an array of len = number of features
        for tree in self.treesList:
            features_lst += tree.get_list_of_features()
        return features_lst


def get_variance(group):
    ''' as described in the instructions
    group is a matrix
     getting the variance of a group of samples. each sample is a vector
     the return value is a scalar'''
    S_size = len(group)
    if S_size == 0:
        return 0
    res = 0
    y_avg = np.mean(group, axis=0) #finding the avarage
    S_size = len(group)
    for i in range(S_size):
        res += (np.linalg.norm(group[i]-y_avg))**2 # return the norm of what's in the () 
    return res/S_size
    
    
def RF_PCT(X, Y, ntree, percentage, mtry, sigma0, n0):
    ''' as shown at the psaudocode. percentage is lambda. '''
    forest0 = forest(list())#a new instance of forest
    n = int(len(X)*percentage) #len(X) returns the size of the first dimension of X
    for i in range(ntree):
        S = np.random.randint(len(X), size = n)# an array of size n of randome integers
        I = (X[S,:], Y[S,:]) # X[S,:] means making a new matrix, taking only the rows appears in S
        T = tree(RFPCT(I[0], I[1], mtry, sigma0, n0))
        forest0.add_tree(T)
    fcounts = forest0.get_list_of_features() # a containing how many times each feature appeared. see the doc of this function for mor info
    #print (fcounts)
    return forest0, fcounts

def trace_down(T,sample):
    ''' Trace down a tree (T) with sample: find the suteable leaf in the tree for the sample '''
    if T.root.right == None: # if T has no 'right' child, he has no 'left' child as well
        return T
    if sample[T.root.feature] < T.root.threshold:
        return trace_down(tree(T.root.left), sample)
    else:
        return trace_down(tree(T.root.right), sample)

def RF_PCT_predict(model, x):
    ''' by the psudocode.
    model is a forest. x is a sample'''
    p = np.zeros(len(model.treesList[0].root.y[1]))
    for T in model.treesList:
        Np = trace_down(T,x)# the leaf that best descrive x in the model
        Yp = Np.root.y
        p = np.add(p, np.average(Yp, axis = 0))
    return p/len(model.treesList) # len of model is the number of the trees

def crossValidation(X,Y,b):
    '''making a cross validation as described in the assignment.
    b ia a vector of integers. Each integer in b represent a group: the group the sample in the suitable row in X and Y will be a part of, in the cross validation process'''
    P = np.zeros(Y.shape)
    j=0
    for i in range(np.amax(b)+1):
  #      Ptemp = [[]]
        AX = X[np.where(b==i)] #current test data
        BX = X[np.where(b!=i)] #current learning data
        BY = Y[np.where(b!=i)] #current learning data
        learnedForest, fcounts = RF_PCT(BX, BY, the_ntree, percentage, the_mtry, sigma0, n0)# percentage is lambda

        #we don't use r (features)
        for p in AX:
            P[j] = RF_PCT_predict(learnedForest, p) # the predicted result for this sample
            j+=1
    return P


def binary_partitions(I,f):
    '''
    :param f: the function
    :return: a list of pairs of nodes
    '''
    X, Y = I
    a = np.amin(X[:,f]) # to know where to start the bins from
    b = np.amax(X[:,f]) # to know where to end the bins
    if a == b:
        thrs = [a]
    else:
        thrs = np.arange(a + (b-a)/numOfBins, b, (b-a)/numOfBins) # the thrash holds that makes the bins
    nodePairs = list() # a list of tuples. each tuple is a diviation of the node based on one of the thrash holds defined the bins
    for thr in thrs:
        R = node(X[np.where(X[:,f]>=thr)], Y[np.where(X[:,f] >= thr)])
        L = node(X[np.where(X[:,f]<thr)], Y[np.where(X[:,f] < thr)])
        R.threshold = L.threshold = thr
        nodePairs+= [(L, R)]
    return nodePairs

def RFPCT(X,Y,mtry,sigma0,n0):
    ''' build random predictive tree. as described in the pseudocode
    '''
    Np = node(X, Y)
    if ((len(X) < n0) or (get_variance(Y) <= sigma0)):  # a leaf by definition
        Np.clear_matrix()
        return Np
    I = (X, Y)
    fs = random.sample(range(X.shape[1]), mtry)
    f, gain, p = Best_Partition(I, fs)
    Np.feature = f
    Np.threshold = p[0].threshold
    for c in p:
        Nc = RFPCT(c.x, c.y, mtry, sigma0, n0)
        Np.add_child(Nc)
        Np.clear_matrix()
    return Np
    
def Best_Partition(I,fs):
    '''
    :param fs: list of functions
    :return: the bast partition of the information I, by a function from fs
    '''
    gain = 0
    sov = 0 #  sum of variances of the binary partition parts
    res_f = None
    res_p = None
    (x, y) = I
    q=0
    y_var = get_variance(y)
    group_size = len(y)
    for f in fs:       
        ps = binary_partitions(I,f) # binary_partitions returns a list of pairs of nodes with threshold included
        for p in ps: # p is a pair of nodes           
            sov=0
            for i in p: # i is a node
               sov = sov + (len(i.y) / group_size) * get_variance(i.y)
            h = y_var - sov
            if h > gain:
                q+=1
                res_f, gain, res_p = f, h, p
    return res_f, gain, res_p

def simplePerformanceScores(y,p,thr):
    '''
    calculate precision, recall, err and FPR
    '''
    #y is a
    #converting y from binary to boolean
    zero_vector = np.zeros(len(y))
    y = y > zero_vector
    #the cross validation
    Yp = np.in1d(np.arange(0,len(p),1), np.where(p >= thr))  # returns a boolean array, of length of len(p), and the value "True" is in the cell i iff  p[i]=>thr 
    TP = np.sum(np.bitwise_and(Yp,y))
    TN = len(p) - np.sum(np.bitwise_or(Yp,y))  # count for how may i: Yp[i] == 0 and y[i] == 0
    FP = np.sum(np.bitwise_and(Yp, np.logical_not(y)))  # # count for how may i: Yp[i] == 1 and y[i] == 0
    FN = np.sum(np.bitwise_and(y, np.logical_not(Yp)))  # # count for how may i: Yp[i] == 0 and y[i] == 1

    precision = TP / (TP + FP) # return a float
    recall = TP / (TP+FN)
    err = (FP+FN)/len(y)
    FPR = FP/(FP + TN)

    return (precision, recall, err, FPR)

def ROCAUC(y,p):
    thrs = np.sort(p)
    len_thrs = len(thrs)
    TPRs= np.zeros(len_thrs)
    FPRs = np.zeros(len_thrs)
    for i in range(len(thrs)):
        r = simplePerformanceScores(y,p,thrs[len(thrs) - i - 1])
        TPRs[i] = r[1]
        FPRs[i] = r[3]
    AUC = np.trapz(TPRs, x=FPRs)
    return AUC

def precision_ROCAUC(y,p):
    thrs = np.sort(p)
    len_thrs = len(thrs)
    PRs = np.zeros(len_thrs)
    REs = np.zeros(len_thrs)
    for i in range(len(thrs)):
        r = simplePerformanceScores(y,p,thrs[len(thrs) - i - 1])
        PRs[i] = r[0]
        REs[i] = r[1]
    AUC = np.trapz(PRs, x=REs)
    return AUC

def ROCAUC_and_precision_ROCAUC(y,p):
    '''combine the two. run faster this way'''
    thrs = np.sort(p)
    len_thrs = len(thrs)
    TPRs = np.zeros(len_thrs)
    FPRs = np.zeros(len_thrs)
    PRs = np.zeros(len_thrs)
    REs = np.zeros(len_thrs)
    for i in range(len(thrs)):
        r = simplePerformanceScores(y,p,thrs[i])
        TPRs[i] = r[1]
        FPRs[i] = r[3]
        PRs[i] = r[0]
    AUC, precision_AUC = np.trapz(TPRs, FPRs), np.trapz(PRs, TPRs)
    return (AUC, precision_AUC)


def feature_count(fcounts, x_lable):
    '''counts how many features'''
    lable_count = {}
    for i in range(len(fcounts)):
        lable_count[x_lable[i]] = fcounts[i]
    return lable_count
    
def write_dicti_to_file11(dictionary, out_file):
    '''print the lables counting dictionary to a file'''
    f = open(out_file, 'w')    
    for k,v in dictionary.items():
        f.write(str(k)+ ": " + str(v) +'\n')
    f.close()

def write_dicti_to_file(dictionary, out_file):
    '''print the lables counting dictionary to a file'''
    f = open(out_file, 'w')
    sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1))
    for tup in sorted_dict:
        f.write(str(tup[0])+ ": " + str(tup[1]) +'\n')
    f.close()
    
def average_error_rate(b,Xdata, Ydata):
    '''make the calculation of the average error rate
    a naive algorithm'''
    f = open("run_results_avg_err_rate.txt", 'w')
    mtry_lst = [5,10,20,40]
    global the_ntree
    the_ntree = 100
    for i in range(len(mtry_lst)):
        sum = 0
        t = 0
        global the_mtrj
        the_mtry = mtry_lst[i]
        for j in range(Ydata.shape[1]):
            start_time = time.time()
            P = crossValidation(Xdata,Ydata,b)
            end_time = time.time()
            t += end_time-start_time
            sum += error_plot(Ydata[:,j], P[:,j])
        avgE = sum / Ydata.shape[1]
        avgT = t / Ydata.shape[1]
        f.write("mtry= "+ str(mtry_lst[i])+ ";" + " avg: " + str(avgE) +";"+ "avgT: "+ str(avgT)+ "seq \n")
    f.close()
    end_time = time.time()
    #print ("time= ")
    #print(start_time - end_time)



def error_plot(y,p):
    '''a sub function in the avg error rate computation'''
    r=simplePerformanceScores(y,p,0.5)
    error=r[2]
    return error

def run_RF_PCT(Xdata, Ydata):
    global the_mtry
    the_mtry = int(math.sqrt(len(Xdata)))
    forest0, fcounts = RF_PCT(Xdata, Ydata, the_ntree, percentage, the_mtry, sigma0, n0)
    print(forest0, fcounts)


if __name__ == "__main__":
    x_path, y_path, res_path = sys.argv[2], sys.argv[3], sys.argv[4]
    sys.argv[1:]
    Xdata = np.loadtxt(x_path)
    Ydata = np.loadtxt(y_path)
    run_RF_PCT(x_path, y_path)
    exit(0) #tab this line to calculte AUC ect.
    

    global the_mtry
    the_mtry = int(math.sqrt(len(Xdata)))


    #filepath =  r"D:\My Documents\Downloads\emotions.txt"
    #data_matrix = data_matrix(filepath, where_Y_starts)
    a = len(Xdata) / 10.0
    #if the data is composed from diferent datasets, make b such that the i entry is the index of the dataset the sample came from
    b = np.array([int(x/(len(Xdata)/10)) for x in range(len(Xdata))])
    #P = crossValidation(data_matrix.matrix[0],data_matrix.matrix[1],b)
#    b = np.random.permutation(b)
   # print(Ydata.shape[1])
    #exit(0) 
    f = open("res_path", 'w')
    ntree_lst = [1, 10, 25, 50, 100]

    #global the_mtry
    #the_mtry = int(math.sqrt(len(data_matrix.matrix[0])))

    start_time = time.time()
    for j in range(len(ntree_lst)):
        global the_ntree
        the_ntree = ntree_lst[j]
        P = crossValidation(Xdata,Ydata,b)
        print("runit ntree= ", the_ntree)
        for i in range(Ydata.shape[1]):
            print("i = ", i)
            f.write("lable= "+ str(i)+ " ntree: "+ str(ntree_lst[j])+ "\n")
            f.write ("ROC AUC = " + str(ROCAUC(Ydata[:,i], P[:,i]))+"\n")
            f.write ("AUPR = " + str(precision_ROCAUC(Ydata[:,i], P[:,i]))+"\n")
            f.write("error= " + str(error_plot(Ydata[:,i], P[:,i]))+ "\n")
            gc.collect()
    f.close()
    end_time = time.time()
    #print("time:")
    #print(end_time - start_time)

    global the_ntree
    the_ntree = 100
    learnedForest, fcounts = RF_PCT(Xdata, Ydata, the_ntree, percentage, the_mtry, sigma0, n0)
    f_names = [i for i in range(len(Xdata[0]))]
    write_dicti_to_file(feature_count(fcounts,f_names), "fcounts results1923.txt")
    
    average_error_rate(b,Xdata,Ydata)
    