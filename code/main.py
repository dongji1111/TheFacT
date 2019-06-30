# Import the Required Libraries
import autograd.numpy as np
import math
import decision_tree_f as dtree
import optimization as opt
import argparse


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
    num_users = data[:, 0].max() + 1
    print ("Number of users: ", num_users)
    num_items = data[:, 1].max() + 1
    print ("Number of items: ", num_items)
    num_features = max(feature) + 1
    print ("Number of features: ", num_features)

    rating_matrix = np.zeros((num_users, num_items), dtype=float)
    user_opinion = np.zeros((num_users, num_features), dtype=float)
    item_opinion = np.zeros((num_items, num_features), dtype=float)

    for i in range(len(data)):
        user_id, item_id, rating = data[i]
        rating_matrix[user_id][item_id] = rating
        for j in range(0, len(data_fo[i]), 2):
            user_opinion[user_id][data_fo[i][j]] += data_fo[i][j + 1]
            item_opinion[item_id][data_fo[i][j]] += data_fo[i][j + 1]

    for i in range(len(user_opinion)):
        for j in range(len(user_opinion[0])):
            if user_opinion[i, j] > 0:
                user_opinion[i, j] = 1
            else:
                if user_opinion[i, j] < 0:
                    user_opinion[i, j] = -1
                else:
                    user_opinion[i, j] = 0

    for i in range(len(item_opinion)):
        for j in range(len(item_opinion[0])):
            if item_opinion[i, j] > 0:
                item_opinion[i, j] = 1
            else:
                if item_opinion[i, j] < 0:
                    item_opinion[i, j] = -1
                else:
                    item_opinion[i, j] = 0
    return rating_matrix, user_opinion, item_opinion


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


# k is the latent dimension for matrix factorization.
# learningRate is the  rate for parameter updating,
# lamda_user, lamda_item are regularization  parameters,
# noOfIteration is an integer specifying the number of iterations to run,
# file_training is a string specifying the file directory for training dataset.
def MF(k, learningRate, lmd_u, lmd_v, noOfIteration, file_training):

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
    user_vectors, item_vectors = MF(20, 0.05, 0.02, 0.02, 50, File)

    i = 0
    print("Entering Main Loop of alternateOptimization")
    decTree = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
    # Do converge Check
    while i < 5:
        # Create the decision Tree based on item_vectors
        print("Creating Tree.. for i = ", i, "for user")
        decTree = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
        decTree.fitTree_U(decTree.root, opinion_matrix, rating_matrix, item_vectors, NUM_OF_FACTORS)

        print("Getting the user vectors from tree")
        # Calculate the User vectors using dtree
        user_vectors_before = user_vectors
        user_vectors = decTree.getVectors_f(opinion_matrix, NUM_OF_FACTORS)
        # adding personalized term
        for index in range(len(rating_matrix)):
            indice = np.array([index])
            user_vectors[index] = opt.cf_user(rating_matrix, item_vectors, user_vectors[index], indice, NUM_OF_FACTORS)

        print("Creating Tree.. for i = ", i, "for item")
        decTreeI = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
        decTreeI.fitTree_I(decTreeI.root, opinion_matrix_I, rating_matrix, user_vectors, NUM_OF_FACTORS)

        print("Getting the item vectors from tree")
        item_vectors_before = item_vectors
        item_vectors = decTreeI.getVectors_f(opinion_matrix_I, NUM_OF_FACTORS)
        for index in range(len(rating_matrix[0])):
            indice = np.array([index])
            item_vectors[index] = opt.cf_item(rating_matrix, user_vectors, item_vectors[index], indice, NUM_OF_FACTORS)

        # Calculate Error for Convergence check
        Pred_before = np.dot(user_vectors_before, item_vectors_before.T)
        Pred = np.dot(user_vectors, item_vectors.T)
        Error = Pred_before - Pred
        Error = Error[np.nonzero(Error)]
        error = np.dot(Error, Error)
        if error < 0.1:
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
    user_tree, item_tree, user_vector, item_vector = alternateOptimization(user_opinion, item_opinion, rating_matrix, num_dim, max_depth, train_file)
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
