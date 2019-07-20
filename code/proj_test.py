import numpy as np
import decision_tree_f_test as dtree
# import time
from numba import jit
import optimization_test as opt
import math
import sys


def getRatingMatrix(filename):
    # Open the file for reading data
    file = open(filename, "r")

    while 1:
        # Read all the lines from the file and store it in lines
        lines = file.readlines(1000000000)

        # if Lines is empty, simply break
        if not lines:
            break

        # Create a list to hold all the data
        data = []
        data_fo = []

        print("Number of Lines: ", len(lines))

        # For each Data Entry, get the rating and the f-o pairs in their respective list
        for line in lines:
            # print("dealing with new line")
            # Get all the attributes by splitting on ','
            list1 = line.split("\n")[0].split(",")
            list2 = list1.pop()
            # list1 store userid itemid and rating
            # list2 store all f-o pair for each userid
            list2 = list2.split(" ")
            list2 = [int(j) for j in list2]
            list1 = [int(j) for j in list1]
            # Add to the data
            data.append(list1)
            data_fo.append(list2)

        index_f = []
        for i in data_fo:
            index_f.extend(i)

        index_f = np.array(index_f)
        index_f = index_f[np.argmax(index_f)] + 1
        print("Number of feature : ", index_f)
        # convert data into numpy form
        data = np.array(data)

        # Get the indices of the maximum Values in each column
        a = np.argmax(data, axis=0)
        # print(a)
        num_users = data[a[0]][0] + 1
        num_items = data[a[1]][1] + 1

        # print "Max values Indices: ", a
        print("Number of Users: ", num_users)
        print("Number of Items: ", num_items)

        # print(data_fo)
        # print(data)

        ratingMatrix = np.zeros((num_users, num_items), dtype=float)
        opinionMatrix = np.zeros((num_users, index_f), dtype=float)
        opinionMatrix = np.full(opinionMatrix.shape,10000)
        opinionMatrix_I = np.zeros((num_items, index_f), dtype=float)
        opinionMatrix_I = np.full(opinionMatrix_I.shape, 10000)
        # print(len(data))
        # print(len(data_fo))

        for i in range(len(data)):
            list1 = data[i]  # userid itemid rating in line i
            list2 = data_fo[i]  # all f-o pair in line i
            ratingMatrix[list1[0]][list1[1]] = list1[2]
            f_positive = 0
            f_negtive = 0
            for j in range(0, len(list2), 2):
                # list2[j] is feature_id list2[j+1] is value of opinion
                if(list2[j+1]>0):
                    f_positive+=1
                elif(list2[j+1]<0):
                    f_negtive+=1
                opinionMatrix[list1[0]][list2[j]] = f_positive+f_negtive
                opinionMatrix_I[list1[1]][list2[j]] = f_positive-f_negtive

        return ratingMatrix, opinionMatrix, opinionMatrix_I


def getNDCG(predict, real, N):
    NDCG = []
    predict = np.array(predict)
    real = np.array(real)
    #fout = open('reclist.txt', 'w')
    for i in range(len(predict)):
        arg_pre = np.argsort(-predict[i])
        rec_pre = real[i][arg_pre]
        #fout.write('user'+ str(i)+ 'value of real rating with predict ranking :'+str(rec_pre))
        rec_pre = [rec_pre[k] for k in range(N)]  # value of real rating with Top N predict recommendation
        # rec_pre = np.array(rec_pre)
        arg_real = np.argsort(-real[i])  # ideal ranking of real rating with Top N
        rec_real = real[i][arg_real]
        #fout.write('user', str(i)+'value of real rating with ideal ranking :'+ str(rec_real))
        rec_real = [rec_real[k] for k in range(N)]
        # print("rec_pre",rec_pre)
        # print("rec_real",rec_real)
        dcg = 0
        idcg = 0
        for j in range(N):
            dcg = dcg + rec_pre[j] / math.log2(j + 2)
            idcg = idcg + rec_real[j] / math.log2(j + 2)
        #print("dcg",dcg)
        #print("idcg",idcg)
        if(idcg!=0):
            NDCG.append(dcg / idcg)
    print(NDCG)
    sum = 0
    for i in range(len(NDCG)):
        sum = sum + NDCG[i]
    ndcg = sum / len(NDCG)
    return ndcg

'''
Use Numba.cuda to accelerate the matrix factorization
'''
# def vec_inner(vec_1, vec_2, r):
#     for i in range(len(vec_1)):


@jit
def update_vectors(rating, userID, itemID, user_vectors, item_vectors, k, learningRate, lmd_u, lmd_v):
    for j in range(len(rating)):
        pred = 0
        for i in range(k):
            pred += user_vectors[userID[j], i] * item_vectors[itemID[j], i]
        err = rating[j] - pred
        user_vectors[userID[j], :] += learningRate * (err * item_vectors[itemID[j], :] - lmd_u * user_vectors[userID[j], :])
        item_vectors[itemID[j], :] += learningRate * (err * user_vectors[userID[j], :] - lmd_v * item_vectors[itemID[j], :])

    error = 0
    for j in range(len(rating)):
        pred = 0
        for i in range(k):
            pred += user_vectors[userID[j], i] * item_vectors[itemID[j], i]
        error += pow((rating[j] - pred), 2)

    return user_vectors, item_vectors, error


def MF(k, learningRate, lmd_u, lmd_v, noOfIteration, file_training):
    # k is the latent dimension for matrix factorization.
    # learningRate is the  rate for parameter updating,
    # lamda_user, lamda_item are regularization  parameters,
    # noOfIteration is an integer specifying the number of iterations to run,
    # file_training is a string specifying the file directory for training dataset.
    file = open(file_training, 'r')
    lines = file.readlines()
    numberOfUsers = 0
    numberOfItems = 0
    userID = np.zeros((len(lines)), dtype=int)
    itemID = np.zeros((len(lines)), dtype=int)
    rating = np.zeros((len(lines)))
    count = 0

    print("Preparing data.........")
    for line in lines:
        listOfLine = line.split("\n")[0].split(",")
        userID[count] = int(listOfLine[0])
        # print(userID[count])
        if userID[count] + 1 > numberOfUsers:
            numberOfUsers = userID[count] + 1
        itemID[count] = int(listOfLine[1])
        # print(itemID[count])
        if itemID[count] + 1 > numberOfItems:
            numberOfItems = itemID[count] + 1
        rating[count] = float(listOfLine[2])
        count = count + 1

    # Inialization for the latent model.
    np.random.seed(0)
    user_vectors = np.random.rand(int(numberOfUsers), k)
    item_vectors = np.random.rand(int(numberOfItems), k)

    # parameter update by Stochastic Gradient Descent
    print("Calculating error")
    error = np.zeros((noOfIteration))
    for i in range(noOfIteration):

        user_vectors, item_vectors, error[i] = update_vectors(rating, userID, itemID, user_vectors, item_vectors, k, learningRate, lmd_u, lmd_v)

    return user_vectors, item_vectors


# Returns the rating Matrix with approximated ratings for all users for all items using fMf
def alternateOptimization(opinion_matrix, opinion_matrix_I, rating_matrix, NUM_OF_FACTORS, MAX_DEPTH, File):
    # Save and print the Number of Users and Movies
    NUM_USERS = rating_matrix.shape[0]
    NUM_MOVIES = rating_matrix.shape[1]
    NUM_FEATURE = opinion_matrix.shape[1]
    print("Number of Users", NUM_USERS)
    print("Number of Item", NUM_MOVIES)
    print("Number of Feature", NUM_FEATURE)
    print("Number of Latent Factors: ", NUM_OF_FACTORS)

    # Create the user and item profile vector of appropriate size.
    # Initialize the item vectors according to MF
    user_vectors, item_vectors = MF(20, 0.05, 0.02, 0.02, 100, File)
    # user_vectors = np.random.rand(NUM_USERS, NUM_OF_FACTORS)
    # item_vectors = np.random.rand(NUM_MOVIES, NUM_OF_FACTORS)

    i = 0
    print("Entering Main Loop of alternateOptimization")
    decTree = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
    # Do converge Check
    while i < 5:

        # Create the decision Tree based on item_vectors
        #print("Creating Tree.. for i = ", i, "for user")
        #decTree = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
        #decTree.fitTree_U(decTree.root, opinion_matrix, rating_matrix, item_vectors, NUM_OF_FACTORS)
        #print("print user tree ", i)
        #decTree.printtree(decTree.root)
        print("Getting the user vectors from tree")
        # Calculate the User vectors using dtree
        user_vectors_before = user_vectors
        #user_vectors = decTree.getVectors_f(opinion_matrix, NUM_OF_FACTORS)
        # adding personalized term
        for index in range(len(rating_matrix)):
            indice = np.array([index])
            user_vectors[index] = opt.cf_user(rating_matrix, item_vectors, user_vectors[index], indice,
                                              NUM_OF_FACTORS)

        print("Creating Tree.. for i = ", i, "for item")
        decTreeI = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
        decTreeI.fitTree_I(decTreeI.root, opinion_matrix_I, rating_matrix, user_vectors, NUM_OF_FACTORS)
        print("print item tree " , i)
        decTreeI.printtree(decTreeI.root)
        print("Getting the item vectors from tree")
        item_vectors_before = item_vectors
        item_vectors = decTreeI.getVectors_f(opinion_matrix_I, NUM_OF_FACTORS)
        for index in range(len(rating_matrix[0])):
            indice = np.array([index])
            item_vectors[index] = opt.cf_item(rating_matrix, user_vectors, item_vectors[index], indice,
                                              NUM_OF_FACTORS)

        # Calculate Error for Convergence check
        Pred_before = np.dot(user_vectors_before, item_vectors_before.T)
        Pred = np.dot(user_vectors, item_vectors.T)
        Error = Pred_before - Pred
        Error = Error.flatten()
        error = np.dot(Error, Error)
        if error < 0.1:
            break
        i = i + 1

    return decTree, decTreeI, user_vectors, item_vectors


def printTopKMovies(test, predicted, K):
    # Gives top K  recommendations
    print("Top Movies Not rated by the user")

    for i in range(len(test)):  # for each user
        zero_list = []
        item_list = []
        for j in range(len(test[0])):  # for each item
            if test[i][j] == 0:
                zero_list.append(predicted[i][j])  # rating value
                item_list.append(j)  # item index

            zero_array = np.array(zero_list)
            item_array = np.array(item_list)

            args = np.argsort(zero_array)
            item_array = item_array[args]
        if K < len(item_array):
            print("user ", i, " : ", item_array[0:K])
        else:
            print("user", i, " : ", item_array)


if __name__ == "__main__":
    # Get the Data
    #File_train, File_test, MAX_DEPTH, NUM_OF_FACTORS = sys.argv[1:]
    File_train = "test_train.txt"
    (rating_matrix, opinion_matrix, opinion_matrixI) = getRatingMatrix(File_train)
    print("Dimensions of the training dataset: ", rating_matrix.shape)
    # Set the number of Factors

    NUM_OF_FACTORS = 20
    MAX_DEPTH = 6
    user_vectors, item_vectors = MF(20, 0.05, 0.02, 0.02, 100, File_train)
    # Build decision tree on training set
    (decisionTree, decisionTreeI, user_vectors, item_vectors) = alternateOptimization(opinion_matrix,
                                                                                      opinion_matrixI,rating_matrix,NUM_OF_FACTORS,MAX_DEPTH, File_train)

    #user_vectors,item_vectors=MF(20, 0.05, 0.02,0.02, 1000, File)
    Predicted_Rating = np.dot(user_vectors, item_vectors.T)
    np.savetxt('item_vectors6.txt', item_vectors, fmt='%0.8f')
    np.savetxt('user_vectors6.txt', user_vectors, fmt='%0.8f')
    np.savetxt('rating_predict6.txt', Predicted_Rating, fmt='%0.8f')
    print("print user tree")
    #decisionTree.printtree(decisionTree.root)
    print("print item tree")
    #decisionTree.printtree(decisionTreeI.root)
    File_test = "test_test.txt"
    (test_r, test_opinion, test_opinionI) = getRatingMatrix(File_test)
    #Predicted_Rating[np.where(rating_matrix[:, :] > 0)] = 0.0
    NDCG = getNDCG(Predicted_Rating, test_r, 10)
    #print("NDCG@10: ", NDCG)
    NDCG = getNDCG(Predicted_Rating, test_r, 20)
    #print("NDCG@20: ", NDCG)
    NDCG = getNDCG(Predicted_Rating, test_r, 50)
    #print("NDCG@50: ", NDCG)