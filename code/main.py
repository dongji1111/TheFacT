# Import the Required Libraries
# import autograd.numpy as np
import numpy as np
import math
import decision_tree as dt
import optimization as opt
import argparse
from sklearn.metrics import mean_squared_error


# load the data
def getRatingMatrix(filename):
    # Open the file for reading data
    data = []
    data_fo = []
    feature = []
    with open(filename) as file:
        for line in file:
            d = line[:-1].split(",")
            list1 = [int(x) for x in d[:-1]]
            list2 = [int(x) for x in d[-1].split(" ")]

            data.append(list1)
            data_fo.append(list2)
            for i in list2:
                feature.append(i)

    data = np.array(data)
    # calculate the number of users, items and features in the dataset
    num_users = data[:, 0].max() + 1
    print ("Number of users: ", num_users)
    num_items = data[:, 1].max() + 1
    print ("Number of items: ", num_items)
    num_features = max(feature) + 1
    print ("Number of features: ", num_features)
    
    # create rating matrix, and user_opinion, item_opinion matrices
    # user_opinion: user preference for each feature
    # item_opinion: item performance on each feature
    rating_matrix = np.zeros((num_users, num_items), dtype=float)
    user_opinion = np.zeros((num_users, num_features), dtype=float)
    item_opinion = np.zeros((num_items, num_features), dtype=float)
    # update the matrices with input data
    # get the accumulated feature opinion scores for users and items.
    for i in range(len(data)):
        user_id, item_id, rating = data[i]
        rating_matrix[user_id][item_id] = rating
        for j in range(0, len(data_fo[i]), 2):
            user_opinion[user_id][data_fo[i][j]] += data_fo[i][j + 1]
            item_opinion[item_id][data_fo[i][j]] += data_fo[i][j + 1]

    # use the sign function to change the accumulated opinion matrices
    user_opinion = np.sign(user_opinion)
    item_opinion = np.sign(item_opinion)


    # for i in range(len(user_opinion)):
    #     for j in range(len(user_opinion[0])):
    #         if user_opinion[i, j] > 0:
    #             user_opinion[i, j] = 1
    #         else:
    #             if user_opinion[i, j] < 0:
    #                 user_opinion[i, j] = -1
    #             else:
    #                 user_opinion[i, j] = 0

    # for i in range(len(item_opinion)):
    #     for j in range(len(item_opinion[0])):
    #         if item_opinion[i, j] > 0:
    #             item_opinion[i, j] = 1
    #         else:
    #             if item_opinion[i, j] < 0:
    #                 item_opinion[i, j] = -1
    #             else:
    #                 item_opinion[i, j] = 0
    return rating_matrix, user_opinion, item_opinion


# calculate NDCG for the prediction
# predict: predicted label
# gt: ground truth
# N: parameter for NDCG
def getNDCG(predict, gt, N):
    NDCG = []
    predict = np.array(predict)
    gt = np.array(gt)

    fout = open('../results/reclist.txt', 'w')
    for i in range(len(predict)):
        arg_pred = np.argsort(-predict[i])
        rec_pred = gt[i][arg_pred]
        fout.write('user' + i + 'value of real rating with predict ranking :' + rec_pre)
        rec_pred = [rec_pred[k] for k in range(N)]
        arg_gt = np.argsort(-gt[i])
        rec_gt = gt[i][arg_gt]
        fout.write('user' + i + 'value of real rating with ideal ranking :' + rec_real)
        rec_gt = [rec_gt[k] for k in range(N)]

        dcg = 0
        idcg = 0
        for j in range(N):
            dcg = dcg + rec_pred[j] / math.log2(j + 2)
            idcg = idcg + rec_gt[j] / math.log2(j + 2)
        NDCG.append(dcg / idcg)
    print(NDCG)
    ndcg_sum = 0
    for i in range(len(NDCG)):
        ndcg_sum = ndcg_sum + NDCG[i]
    ndcg = ndcg_sum / len(NDCG)
    return ndcg


# solve ridge regression with stochastic gradient descent
# num_dim: the dimension of latent factors
# lr: learning rate
# lambda_u: regularization parameter for user vectors
# lambda_v: regularization parameter for item vectors
# rating_matrix: the ground truth
def MatrixFactorization(num_dim, lr, lamda_u, lambda_v, num_iters, rating_matrix):

    # get the nonzero values of the rating matrix
    np.random.seed(0)
    user_index, item_index = rating_matrix.nonzero()
    mask_matrix = rating_matrix.nonzero()
    num_records = len(user_index)
    num_users = user_index.max() + 1
    num_items = item_index.max() + 1
    # randomly initialize the user, item vectors.
    user_vector = np.random.rand(num_users, num_dim)
    item_vector = np.random.rand(num_items, num_dim)

    # for each iteration
    for it in range(num_iters):
        # for each non-zero rating records
        for i in range(num_records):
            u_id, v_id = user_index[i], item_index[i]
            r = rating_matrix[u_id, v_id]
            print u_id, v_id, r
            # update latent factors of users and items
            user_vector[u_id] += lr * ((r - np.dot(user_vector[u_id], item_vector[v_id])) * item_vector[v_id] - lambda_u * user_vectors[u_id])
            item_vector[v_id] += lr * ((r - np.dot(user_vector[u_id], item_vector[v_id])) * user_vector[u_id] - lambda_v * item_vectors[v_id])
            
        # calculte the training error
        pred = np.dot(user_vector, item_vector.T)
        error = mean_squared_error(pred[mask_matrix], rating_matrix[mask_matrix])
    return user_vector, item_vector
        

def MF(k, lr, lambda_u, lambda_v, num_iters, filename):

    file = open(filename, 'r')
    lines = file.readlines()
    num_users = 0
    num_items = 0
    userID = np.zeros((len(lines)), dtype=int)
    itemID = np.zeros((len(lines)), dtype=int)
    rating = np.zeros((len(lines)))
    count = 0

    print("Preparing data.........")
    for line in lines:
        listOfLine = line.split("\n")[0].split(",")
        userID[count] = int(listOfLine[0])
        if userID[count] + 1 > numberOfUsers:
            numberOfUsers = userID[count] + 1
        itemID[count] = int(listOfLine[1])
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
        print("Iteration times: ", i)
        for j in range(len(lines)):
            user_vectors[userID[j], :] = user_vectors[userID[j], :] + learningRate*((rating[j] - np.dot(user_vectors[userID[j],:], item_vectors[itemID[j],:]))*item_vectors[itemID[j], :]-lmd_u*user_vectors[userID[j], :])
            item_vectors[itemID[j], :] = item_vectors[itemID[j], :] + learningRate*((rating[j] - np.dot(user_vectors[userID[j],:], item_vectors[itemID[j],:]))*user_vectors[userID[j], :]-lmd_v*item_vectors[itemID[j], :])

        for j in range(len(lines)):
            temp = rating[j] - np.dot(user_vectors[userID[j], :], item_vectors[itemID[j], :])
            error[i] = error[i] + temp * temp
        print(error[i])
    return user_vectors, item_vectors


def AlternateOptimization(rating_matrix, user_opinion, item_opinion, num_dim, max_depth):
    # Save and print the Number of Users and Movies
    num_users, num_items = rating_matrix.shape
    num_features = user_opinion.shape[1]
    print("Number of users", num_users)
    print("Number of items", num_items)
    print("Number of features", num_features)
    print("Number of latent dimensions: ", num_dim)
    print("Maximum depth of the regression tree: ", max_depth)
    # initialize the parameters
    lr = 0.05
    lambda_u = 0.02
    lambda_v = 0.02
    num_iters = 50

    # Create the user and item profile vector of appropriate size.
    # Initialize the item vectors according to MF
    user_vector, item_vector = MatrixFactorization(num_dim, lr, lambda_u, lambda_v, num_iters, rating_matrix)
    pred = np.dot(user_vector, item_vector.T)

    i = 0
    print("Entering Main Loop of AlternateOptimization")
    tree = dt.Tree(dt.Node(None, 1), num_dim, max_depth)
    # create the tree with i=5 iterations.
    while i < 5:
        # log the previous results
        user_vector_old = user_vector
        item_vector_old = item_vector
        pred_old = pred


        # Create the decision Tree based on item_vectors
        print("Creating tree, iter = ", i, "for user")
        user_tree = dt.Tree(dt.Node(None, 1), num_dim, max_depth)
        user_tree.fitTree_U(user_tree.root, user_opinion, rating_matrix, item_vector, num_dim)

        print("Getting the user vectors from tree.")
        # Calculate the User vectors using dtree
        user_vector = user_tree.getVectors_f(user_opinion, num_dim)
        # adding personalized term
        for idx in range(num_users):
            indice = np.array([idx])
            user_vector[idx] = opt.cf_user(rating_matrix, item_vector, user_vector[idx], indice, num_dim)

        print("Creating Tree.. for i = ", i, "for item")
        item_tree = dt.Tree(dt.Node(None, 1), num_dim, max_depth)
        item_tree.fitTree_I(item_tree.root, item_opinion, rating_matrix, user_vector, num_dim)

        print("Getting the item vectors from tree")
        item_vector = decTreeI.getVectors_f(item_opinion, num_dim)
        for idx in range(num_items):
            indice = np.array([idx])
            item_vectors[idx] = opt.cf_item(rating_matrix, user_vectors, item_vectors[idx], indice, num_dim)

        # Calculate Error for Convergence check
        pred = np.dot(user_vector, item_vector.T)

        error = (pred_old - pred).flatten()
        err = np.dot(error, error)
        if err < 0.1:
            break
        i = i + 1

    return decTree, decTreeI, user_vectors, item_vectors


def printTopKMovies(test, predicted, K):
    # Gives top K  recommendations
    print("Top Movies Not rated by the user")

    for i in range(len(test)):
        zero_list = []
        item_list = []
        for j in range(len(test[0])):
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
    # initialization
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="training filename", default="../data/yelp_train.txt")
    parser.add_argument("--test_file", help="test filename", default="../data/yelp_test.txt")
    parser.add_argument("--num_dim", help="the number of latent dimension", default=20)
    parser.add_argument("--max_depth", help="the maximum depth of the tree", default=6)

    args = parser.parse_args()
    train_file = args.train_file
    test_file = args.test_file
    num_dim = int(args.num_dim)
    max_depth = int(args.max_depth)
    rating_matrix, user_opinion, item_opinion = getRatingMatrix(train_file)

    # build the Factorization tree with the training dataset
    user_tree, item_tree, user_vector, item_vector = AlternateOptimization(user_opinion, item_opinion, rating_matrix, num_dim, max_depth, train_file)
    pred_rating = np.dot(user_vector, item_vector.T)
    pred_rating[np.where[rating_matrix > 0]] = 0.0
    # save the results
    np.savetxt("../results/item_vector.txt", item_vector, fmt='%0.8f')
    np.savetxt("../results/user_vector.txt", user_vector, fmt="%0.8f")
    np.savetxt("../results/rating_matrix.txt", pred_rating, fmt="%0.8f")

    # test on test data with the trained model
    test_rating, user_opinion_test, item_opinion_test = getRatingMatrix(test_file)

    # get the NDCG results
    print("print user tree")
    user_tree.printtree(user_tree.root)
    print("print item tree")
    item_tree.printtree(item_tree.root)

    NDCG = getNDCG(pred_rating, test_rating, 10)
    print("NDCG@10: ", NDCG)
    NDCG = getNDCG(pred_rating, test_rating, 20)
    print("NDCG@20: ", NDCG)
    NDCG = getNDCG(pred_rating, test_rating, 50)
    print("NDCG@50: ", NDCG)
