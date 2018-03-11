
def diagonalize(matrix):
    x = np.diag(np.diag(matrix))
    return x

def sigma(matrix):
    cov = np.cov(matrix.T)
    return cov

def qda(x,u,sigma,r):
    u = u.T #check to make sure mu is already a numpy matrix. u is now 1 x 4
    
    #det of sigma = np.linalg.det(sigma)
    #inv of sigma = np.linalg.inv(sigma)
    
    prob = 1/np.linalg.det(sigma)

    sample = []
    for i in range(0,r):
        sample = x[[i],:] #gets that specific row of x
        sample = sample - u
        sample = sample.T #should now be 4 X 1
        inv = np.linalg.inv(sigma)
        exponent = np.dot(sample.T,inv)
        exponent = np.dot(exponent,sample)
        exponent*=(-0.5)
        prob*=np.exp(exponent)
        print(prob)

    return

def lda(x,u,sigma,r):
#lda(setosa_test_numpy,setosa_train_mu,sigma)
    u = u.T #now u is 1 X 4
    inv = np.linalg.inv(sigma)
    
    sample = []
    
    for i in range(0,r):
        sample = x[[i],:]
        sample = sample - u
        sample = sample.T #should now be 4 X 1
        prob = np.dot(sample.T,inv)
        prob = np.dot(prob,sample)
        print(prob)
    
    #highest probability here is lowest number outputted (because didn't multiply expression with -1/2)

    return




def qdaVariableCheck(test,train,mu,r):
#(setosa_test_numpy,setosa_test_numpy,setosa_train_mu)
    
    #inv = np.linalg.inv(sigma)
    
    sample = []

    #need to delete sigma matrix per row or column???

    #new_mu = np.delete(u,3,0) #deletes third row of column vector

    #calculate probabilities without third feature
    new_test = np.delete(test,3,1) #deletes third column of test data
    new_train = np.delete(train,3,1)
    new_mu = np.delete(mu,3,0) #deletes third row of column vector
    
    new_sigma = np.cov(new_train.T)



    print("qda prob without third feature:")
    qda(new_test,new_mu,new_sigma,(r-1))

    #calculate probabilities without 2nd feature
    new_test = np.delete(test,2,1) #deletes 2ND column of test data
    new_train = np.delete(train,2,1)
    new_mu = np.delete(mu,2,0) #deletes 2ND row of column vector
    
    new_sigma = np.cov(new_train.T)


    print("qda prob without 2nd feature:")
    qda(new_test,new_mu,new_sigma,(r-1))


    #calculate probabilities without first feature
    new_test = np.delete(test,1,1) #deletes 1st column of test data
    new_train = np.delete(train,1,1)
    new_mu = np.delete(mu,1,0) #deletes first row of column vector
    
    new_sigma = np.cov(new_train.T)


    print("qda prob without first feature:")
    qda(new_test,new_mu,new_sigma,(r-1))


#calculate probabilities without 0th feature
    new_test = np.delete(test,0,1) #deletes 0th column of test data
    new_train = np.delete(train,0,1)
    new_mu = np.delete(mu,0,0) #deletes 2ND row of column vector
    
    new_sigma = np.cov(new_train.T)
    
    print("qda prob without 0th feature:")
    qda(new_test,new_mu,new_sigma,(r-1))

    return








import numpy as np

    #put into main function, then call sigma from main

    #def make_matrices():
alldata = []

    #put all data into one matrix

with open('iris.txt', 'r') as input_file:
    for line in input_file:
        row = line.split(',')
        for i in range(0,4):
            row[i] = float(row[i])
        row[4] = row[4].strip('\n')
        alldata.append(row)

    #make np train and test matrices
setosa_train = []

print (alldata)

alldata2 = []
for line in alldata:
    row = []
    for elt in line[0:4]:
        elt=float(elt)
        row.append(elt)
    alldata2.append(row)
#print ("\n")
#print (alldata2)

setosa_test = []

versicolor_train = []

versicolor_test = []

virginica_train = []

virginica_test = []


for i in range(0,40):
    setosa_train.append(alldata2[i])

for i in range(40,50):
    setosa_test.append(alldata2[i])

for i in range(50,90):
    versicolor_train.append(alldata2[i])

for i in range(90,100):
    versicolor_test.append(alldata2[i])

for i in range(100,140):
    virginica_train.append(alldata2[i])

for i in range(140,150):
    virginica_test.append(alldata2[i])

setosa_train_numpy = np.asarray(setosa_train)
setosa_test_numpy = np.asarray(setosa_test)
versicolor_train_numpy = np.asarray(versicolor_train)
versicolor_test_numpy = np.asarray(versicolor_test)
virginica_train_numpy = np.asarray(virginica_train)
virginica_test_numpy = np.asarray(virginica_test)

#print (virginica_test_numpy)


#get means and x vectors
setosa_train_mu = np.mean(setosa_train_numpy, axis=0)
#print(setosa_train_mu)
versicolor_train_mu = np.mean(versicolor_train_numpy, axis=0)
#print(versicolor_train_mu)
virginica_train_mu = np.mean(virginica_train_numpy, axis=0)
#print(virginica_train_mu)

#transpose matrices
setosa_train_mu = setosa_train_mu.reshape(4,1)
versicolor_train_mu = versicolor_train_mu.reshape(4,1)
virginica_train_mu = virginica_train_mu.reshape(4,1)

#get sigma matrices
#print("sigma matrix for setosa is:")
setosa_train_sigma = sigma(setosa_train_numpy)
#print(setosa_train_sigma)
versicolor_train_sigma = sigma(versicolor_train_numpy)
#print(versicolor_train_sigma)
virginica_train_sigma = sigma(virginica_train_numpy)
#print(virginica_train_sigma)

#print(setosa_train_sigma.shape)
inv = np.linalg.inv(setosa_train_sigma)
#print(inv.shape) #==4 X 4


#get qda probabilities for setosa test
print("qda of setosa test data with setosa mu and sigma is:\n")
qda(setosa_test_numpy,setosa_train_mu,setosa_train_sigma,10)
print("qda of setosa test data with versicolor mu and sigma is:\n")
qda(setosa_test_numpy,versicolor_train_mu,versicolor_train_sigma,10)
print("qda of setosa test data with virginica mu and sigma is:\n")
qda(setosa_test_numpy,virginica_train_mu,virginica_train_sigma,10)

#get qda probabilities for versicolor test
print("qda of versicolor test data with setosa mu and sigma is:\n")
qda(versicolor_test_numpy,setosa_train_mu,setosa_train_sigma,10)
print("qda of versicolor test data with versicolor mu and sigma is:\n")
qda(versicolor_test_numpy,versicolor_train_mu,versicolor_train_sigma,10)
print("qda of versicolor test data with virginica mu and sigma is:\n")
qda(versicolor_test_numpy,virginica_train_mu,virginica_train_sigma,10)

#get qda probabilities for virginica test
print("qda of virginica test data with setosa mu and sigma is:\n")
qda(virginica_test_numpy,setosa_train_mu,setosa_train_sigma,10)
print("qda of virginica test data with versicolor mu and sigma is:\n")
qda(virginica_test_numpy,versicolor_train_mu,versicolor_train_sigma,10)
print("qda of virginica test data with virginica mu and sigma is:\n")
qda(virginica_test_numpy,virginica_train_mu,virginica_train_sigma,10)

#get qda probabilities for setosa training
print("qda of setosa training data with setosa mu and sigma is:\n")
qda(setosa_train_numpy,setosa_train_mu,setosa_train_sigma,40)
print("qda of setosa training data with versicolor mu and sigma is:\n")
qda(setosa_train_numpy,versicolor_train_mu,versicolor_train_sigma,40)
print("qda of setosa training data with virginica mu and sigma is:\n")
qda(setosa_train_numpy,virginica_train_mu,virginica_train_sigma,40)

#get qda probabilities for versicolor training
print("qda of versicolor training data with setosa mu and sigma is:\n")
qda(versicolor_train_numpy,setosa_train_mu,setosa_train_sigma,40)
print("qda of versicolor training data with versicolor mu and sigma is:\n")
qda(versicolor_train_numpy,versicolor_train_mu,versicolor_train_sigma,40)
print("qda of versicolor training data with virginica mu and sigma is:\n")
qda(versicolor_train_numpy,virginica_train_mu,virginica_train_sigma,40)

#get qda probabilities for virginica training
print("qda of virginica training data with setosa mu and sigma is:\n")
qda(virginica_train_numpy,setosa_train_mu,setosa_train_sigma,40)
print("qda of virginica training data with versicolor mu and sigma is:\n")
qda(virginica_train_numpy,versicolor_train_mu,versicolor_train_sigma,40)
print("qda of virginica training data with virginica mu and sigma is:\n")
qda(virginica_train_numpy,virginica_train_mu,virginica_train_sigma,40)

#lda calculations
sigma = setosa_train_sigma + versicolor_train_sigma + virginica_train_sigma
sigma*=(1/3)

#lda probabilities for setosa test
print("lda of setosa test data with setosa mu and sigma is:\n")
lda(setosa_test_numpy,setosa_train_mu,sigma,10)
print("lda of setosa test data with versicolor mu and sigma is:\n")
lda(setosa_test_numpy,versicolor_train_mu,sigma,10)
print("lda of setosa test data with virginica mu and sigma is:\n")
lda(setosa_test_numpy,virginica_train_mu,sigma,10)

#lda probabilities for versicolor test
print("lda of versicolor test data with setosa mu and sigma is:\n")
lda(versicolor_test_numpy,setosa_train_mu,sigma,10)
print("lda of versicolor test data with versicolor mu and sigma is:\n")
lda(versicolor_test_numpy,versicolor_train_mu,sigma,10)
print("lda of versicolor test data with virginica mu and sigma is:\n")
lda(versicolor_test_numpy,virginica_train_mu,sigma,10)

#lda probabilities for virginica test
print("lda of virginica test data with setosa mu and sigma is:\n")
lda(virginica_test_numpy,setosa_train_mu,sigma,10)
print("lda of virginica test data with versicolor mu and sigma is:\n")
lda(virginica_test_numpy,versicolor_train_mu,sigma,10)
print("lda of virginica test data with virginica mu and sigma is:\n")
lda(virginica_test_numpy,virginica_train_mu,sigma,10)

#lda probabilities for setosa training
print("lda of setosa training data with setosa mu and sigma is:\n")
lda(setosa_train_numpy,setosa_train_mu,sigma,40)
print("lda of setosa training data with versicolor mu and sigma is:\n")
lda(setosa_train_numpy,versicolor_train_mu,sigma,40)
print("lda of setosa training data with virginica mu and sigma is:\n")
lda(setosa_train_numpy,virginica_train_mu,sigma,40)

#lda probabilities for versicolor training
print("lda of versicolor training data with setosa mu and sigma is:\n")
lda(versicolor_train_numpy,setosa_train_mu,sigma,40)
print("lda of versicolor training data with versicolor mu and sigma is:\n")
lda(versicolor_train_numpy,versicolor_train_mu,sigma,40)
print("lda of versicolor training data with virginica mu and sigma is:\n")
lda(versicolor_train_numpy,virginica_train_mu,sigma,40)

#lda probabilities for virginica training
print("lda of virginica training data with setosa mu and sigma is:\n")
lda(virginica_train_numpy,setosa_train_mu,sigma,40)
print("lda of virginica training data with versicolor mu and sigma is:\n")
lda(virginica_train_numpy,versicolor_train_mu,sigma,40)
print("lda of virginica training data with virginica mu and sigma is:\n")
lda(virginica_train_numpy,virginica_train_mu,sigma,40)

#testing to see if any of the variables are just noise

#setosa qda variable check
print("QDA variable check of setosa data with setosa mu and sigma:\n")
qdaVariableCheck(setosa_test_numpy,setosa_train_numpy,setosa_train_mu,10)
print("QDA variable check of setosa data with versicolor mu and sigma:\n")
qdaVariableCheck(setosa_test_numpy,versicolor_train_numpy,versicolor_train_mu,10)
print("QDA variable check of setosa data with virginica mu and sigma:\n")
qdaVariableCheck(setosa_test_numpy,virginica_train_numpy,virginica_train_mu,10)

#versicolor qda variable check
print("QDA variable check of versicolor data with setosa mu and sigma:\n")
qdaVariableCheck(versicolor_test_numpy,setosa_train_numpy,setosa_train_mu,10)
print("QDA variable check of versicolor data with versicolor mu and sigma:\n")
qdaVariableCheck(versicolor_test_numpy,versicolor_train_numpy,versicolor_train_mu,10)
print("QDA variable check of versicolor data with virginica mu and sigma:\n")
qdaVariableCheck(versicolor_test_numpy,virginica_train_numpy,virginica_train_mu,10)

#virginica qda variable check
print("QDA variable check of virginica data with setosa mu and sigma:\n")
qdaVariableCheck(virginica_test_numpy,setosa_train_numpy,setosa_train_mu,10)
print("QDA variable check of virginica data with versicolor mu and sigma:\n")
qdaVariableCheck(virginica_test_numpy,versicolor_train_numpy,versicolor_train_mu,10)
print("QDA variable check of virginica data with virginica mu and sigma:\n")
qdaVariableCheck(virginica_test_numpy,virginica_train_numpy,virginica_train_mu,10)

#setosa qda training variable check
print("QDA training variable check of setosa data with setosa mu and sigma:\n")
qdaVariableCheck(setosa_train_numpy,setosa_train_numpy,setosa_train_mu,40)
print("QDA training variable check of setosa data with versicolor mu and sigma:\n")
qdaVariableCheck(setosa_train_numpy,versicolor_train_numpy,versicolor_train_mu,40)
print("QDA training variable check of setosa data with virginica mu and sigma:\n")
qdaVariableCheck(setosa_train_numpy,virginica_train_numpy,virginica_train_mu,40)

#versicolor qda training variable check
print("QDA training variable check of versicolor data with setosa mu and sigma:\n")
qdaVariableCheck(versicolor_train_numpy,setosa_train_numpy,setosa_train_mu,40)
print("QDA training variable check of versicolor data with versicolor mu and sigma:\n")
qdaVariableCheck(versicolor_train_numpy,versicolor_train_numpy,versicolor_train_mu,40)
print("QDA training variable check of versicolor data with virginica mu and sigma:\n")
qdaVariableCheck(versicolor_train_numpy,virginica_train_numpy,virginica_train_mu,40)

#virginica qda training variable check
print("QDA training variable check of virginica data with setosa mu and sigma:\n")
qdaVariableCheck(virginica_train_numpy,setosa_train_numpy,setosa_train_mu,40)
print("QDA training variable check of virginica data with versicolor mu and sigma:\n")
qdaVariableCheck(virginica_train_numpy,versicolor_train_numpy,versicolor_train_mu,40)
print("QDA training variable check of virginica data with virginica mu and sigma:\n")
qdaVariableCheck(virginica_train_numpy,virginica_train_numpy,virginica_train_mu,40)

#part 6:
diag_setosa_sigma = diagonalize(setosa_train_sigma)
diag_versicolor_sigma = diagonalize(versicolor_train_sigma)
diag_virginica_sigma = diagonalize(virginica_train_sigma)

#diagonal lda calculations:
diag_sigma = diag_setosa_sigma + diag_versicolor_sigma + diag_virginica_sigma
diag_sigma*=(1/3)

#diagonal lda probabilities:

#diagonal lda probabilities for setosa
print("diagonal lda of setosa data with setosa mu and sigma is:\n")
lda(setosa_test_numpy,setosa_train_mu,diag_sigma,10)
print("diagonal lda of setosa data with versicolor mu and sigma is:\n")
lda(setosa_test_numpy,versicolor_train_mu,diag_sigma,10)
print("diagonal lda of setosa data with virginica mu and sigma is:\n")
lda(setosa_test_numpy,virginica_train_mu,diag_sigma,10)

#diagonal lda probabilities for versicolor
print("diagonal lda of versicolor data with setosa mu and sigma is:\n")
lda(versicolor_test_numpy,setosa_train_mu,diag_sigma,10)
print("diagonal lda of versicolor data with versicolor mu and sigma is:\n")
lda(versicolor_test_numpy,versicolor_train_mu,diag_sigma,10)
print("diagonal lda of versicolor data with virginica mu and sigma is:\n")
lda(versicolor_test_numpy,virginica_train_mu,diag_sigma,10)

#diagonal lda probabilities for virginica
print("diagonal lda of virginica data with setosa mu and sigma is:\n")
lda(virginica_test_numpy,setosa_train_mu,diag_sigma,10)
print("diagonal lda of virginica data with versicolor mu and sigma is:\n")
lda(virginica_test_numpy,versicolor_train_mu,diag_sigma,10)
print("diagonal lda of virginica data with virginica mu and sigma is:\n")
lda(virginica_test_numpy,virginica_train_mu,diag_sigma,10)

#diagonal lda probabilities for setosa training
print("diagonal lda of setosa training data with setosa mu and sigma is:\n")
lda(setosa_train_numpy,setosa_train_mu,diag_sigma,40)
print("diagonal lda of setosa training data with versicolor mu and sigma is:\n")
lda(setosa_train_numpy,versicolor_train_mu,diag_sigma,40)
print("diagonal lda of setosa training data with virginica mu and sigma is:\n")
lda(setosa_train_numpy,virginica_train_mu,diag_sigma,40)

#diagonal lda probabilities for versicolor training
print("diagonal lda of versicolor training data with setosa mu and sigma is:\n")
lda(versicolor_train_numpy,setosa_train_mu,diag_sigma,40)
print("diagonal lda of versicolor training data with versicolor mu and sigma is:\n")
lda(versicolor_train_numpy,versicolor_train_mu,diag_sigma,40)
print("diagonal lda of versicolor training data with virginica mu and sigma is:\n")
lda(versicolor_train_numpy,virginica_train_mu,diag_sigma,40)

#diagonal lda probabilities for virginica training
print("diagonal lda of virginica training data with setosa mu and sigma is:\n")
lda(virginica_train_numpy,setosa_train_mu,diag_sigma,40)
print("diagonal lda of virginica training data with versicolor mu and sigma is:\n")
lda(virginica_train_numpy,versicolor_train_mu,diag_sigma,40)
print("diagonal lda of virginica training data with virginica mu and sigma is:\n")
lda(virginica_train_numpy,virginica_train_mu,diag_sigma,40)


#diagonal qda probabilities:

#diagonal qda probabilities for setosa
print("diagonal qda of setosa data with setosa mu and sigma is:\n")
qda(setosa_test_numpy,setosa_train_mu,diag_setosa_sigma,10)
print("diagonal qda of setosa data with versicolor mu and sigma is:\n")
qda(setosa_test_numpy,versicolor_train_mu,diag_versicolor_sigma,10)
print("diagonal qda of setosa data with virginica mu and sigma is:\n")
qda(setosa_test_numpy,virginica_train_mu,diag_virginica_sigma,10)

#diagonal qda probabilities for versicolor
print("diagonal qda of versicolor data with setosa mu and sigma is:\n")
qda(versicolor_test_numpy,setosa_train_mu,diag_setosa_sigma,10)
print("diagonal qda of versicolor data with versicolor mu and sigma is:\n")
qda(versicolor_test_numpy,versicolor_train_mu,diag_versicolor_sigma,10)
print("diagonal qda of versicolor data with virginica mu and sigma is:\n")
qda(versicolor_test_numpy,virginica_train_mu,diag_virginica_sigma,10)

#diagonal qda probabilities for virginica
print("diagonal qda of virginica data with setosa mu and sigma is:\n")
qda(virginica_test_numpy,setosa_train_mu,diag_setosa_sigma,10)
print("diagonal qda of virginica data with versicolor mu and sigma is:\n")
qda(virginica_test_numpy,versicolor_train_mu,diag_versicolor_sigma,10)
print("diagonal qda of virginica data with virginica mu and sigma is:\n")
qda(virginica_test_numpy,virginica_train_mu,diag_virginica_sigma,10)

#get diagonal qda probabilities for setosa training
print("diagonal qda of setosa training data with setosa mu and sigma is:\n")
qda(setosa_train_numpy,setosa_train_mu,diag_setosa_sigma,40)
print("diagonal qda of setosa training data with versicolor mu and sigma is:\n")
qda(setosa_train_numpy,versicolor_train_mu,diag_versicolor_sigma,40)
print("diagonal qda of setosa training data with virginica mu and sigma is:\n")
qda(setosa_train_numpy,virginica_train_mu,diag_virginica_sigma,40)

#get diagonal qda probabilities for versicolor training
print("diagonal qda of versicolor training data with setosa mu and sigma is:\n")
qda(versicolor_train_numpy,setosa_train_mu,diag_setosa_sigma,40)
print("diagonal qda of versicolor training data with versicolor mu and sigma is:\n")
qda(versicolor_train_numpy,versicolor_train_mu,diag_versicolor_sigma,40)
print("diagonal qda of versicolor training data with virginica mu and sigma is:\n")
qda(versicolor_train_numpy,virginica_train_mu,diag_virginica_sigma,40)

#get diagonal qda probabilities for virginica training
print("diagonal qda of virginica training data with setosa mu and sigma is:\n")
qda(virginica_train_numpy,setosa_train_mu,diag_setosa_sigma,40)
print("diagonal qda of virginica training data with versicolor mu and sigma is:\n")
qda(virginica_train_numpy,versicolor_train_mu,diag_versicolor_sigma,40)
print("diagonal qda of virginica training data with virginica mu and sigma is:\n")
qda(virginica_train_numpy,virginica_train_mu,diag_virginica_sigma,40)

print("sigma virginica is")
print(virginica_train_sigma)


