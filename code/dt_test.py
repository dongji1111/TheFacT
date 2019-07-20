# Import the Required Libraries
import optimization_test as opt
import multiprocessing as mp
import autograd.numpy as np
import time


# Split the users/items into Like, Dislike and Unknown by user/item-feature opinion
def split(data, feature_index, split_point):  # data should be opinion matrix
    # Get the indices for the when the opinion value is like
    indices_like = np.where((data[:, feature_index] > split_point)&(data[:, feature_index] != 10000))[0]

    # Get the indices for the when the opinion value is dislike
    indices_dislike = np.where(data[:, feature_index] <= split_point)[0]

    # Get the indices for the when the opinion is unknown
    indices_unknown = np.where(data[:, feature_index] == 10000)[0]

    return indices_like, indices_dislike, indices_unknown
    # return data[indices_like, :], data[indices_dislike, :], data[indices_unknown, :]


# This class represents each Node of the Decision Tree
class Node:
    def __init__(self, parent_node, node_depth):
        # Each Node has a Like, Dislike and Unknown Child
        # It also stores the index of the item or feature on which its splits the data
        self.parent = parent_node
        #self.indice = indices
        self.depth = node_depth
        self.like = None
        self.dislike = None
        self.unknown = None
        self.feature_index = None  # add feature_index to splits the data
        self.split_point = None # add split_point for feature to splits the data
        self.vector = None


class Tree:
    # __init__() sets the root node, currentDepth and maxdepth of the tree
    def __init__(self, root_node, K, max_depth):
        self.root = root_node
        self.root.vector = np.random.rand(K)
        self.max_depth = max_depth

    def printtree(self, current_node):
        if current_node == None:
            print("None")
            return

        print("Depth of current node:", current_node.depth)
        print("Split feature of current node: ", current_node.feature_index)
        print("Split point of current node",current_node.split_point)
        print("like")
        self.printtree(current_node.like)
        print("dislike", end=" ")
        self.printtree(current_node.dislike)
        print("unknown", end=" ")
        self.printtree(current_node.unknown)

    def getVectors_f(self,opinion_matrix, K):
        ultimate_vector = np.zeros((len(opinion_matrix), K))
        current_node = self.root

        for i in range(len(opinion_matrix)):
            while current_node.like != None or current_node.dislike != None or current_node.unknown != None:
                if opinion_matrix[i][current_node.feature_index] == 10000:
                    current_node = current_node.unknown
                elif opinion_matrix[i][current_node.feature_index] <= current_node.split_point:
                    current_node = current_node.dislike
                else:
                    current_node = current_node.like
            # print("zz", temp.shape)
            ultimate_vector[i] = current_node.vector
        # return the  vector
        return ultimate_vector

    def find_split_point(self, opinion_matrix, feature_index, NUMBER_OF_BIN):
        u_f_opinion = opinion_matrix[:, feature_index]
        arg = np.argsort(u_f_opinion)
        u_f_opinion = u_f_opinion[arg]
        #print(u_f_opinion)
        number_of_unknown = np.sum(u_f_opinion == 10000)
        distance = (len(opinion_matrix)-number_of_unknown)//NUMBER_OF_BIN
        split_point_lists = [u_f_opinion[(i+1)*distance] for i in range(NUMBER_OF_BIN-1)]
        split_point_lists = np.unique(split_point_lists)
        return split_point_lists

    # recursively builds up the entire tree from the root Node
    def fitTree_U(self, current_node, opinion_matrix, rating_matrix, item_vectors, K):
        # rating_matrix only consists of rows which are users corresponding to the current Node
        # Check if the maxDepth is reached
        print("current depth of the tree", current_node.depth)
        t1 = time.time()
        if current_node.depth+1 > self.max_depth:
            return
        if len(rating_matrix) == 0:
            return

        # Calulate the Error Before the Split
        print("Calculate error")
        error_before = opt.lossfunction_all(rating_matrix, item_vectors, current_node.vector, 1)

        print("Error Before: ", error_before)
        # Create a numy_array to hold the split_criteria Values
        params = {}
        feature_splitpoint_matrix = []
        count = 0
        # pool = mp.Pool(20)

        for feature_index in range(len(opinion_matrix[0])):
            # Split the rating_matrix into like, dislike and unknown
            NUMBER_OF_BIN =5
            split_points = self.find_split_point(opinion_matrix, feature_index, NUMBER_OF_BIN)
            feature_splitpoint_matrix.append(split_points)
            for split_point in split_points:
                (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, feature_index, split_point)
                params[count] = []
                params[count].extend((rating_matrix, item_vectors, current_node.vector, indices_like, indices_dislike, indices_unknown, K))
                count += 1

        # Calculate the split criteria value
        print("Calculating the split criteria value")
        results = []

        params_index = 0
        for feature_index in range(len(opinion_matrix[0])):
            print("feature_index",feature_index)
            temp = []
            # start = time.time()
            # result = pool.apply_async(opt.cal_splitvalue, params[feature_index])
            for split_point in feature_splitpoint_matrix[feature_index]:
                print("split_point",split_point)
                result = opt.cal_splitvalue(params[params_index][0], params[params_index][1], params[params_index][2],
                                            params[params_index][3], params[params_index][4], params[params_index][5],
                                            params[params_index][6])
                params_index +=1
                temp.append(result)
            results.append(temp)

        #results = np.array(results)
        temp_value = []
        temp_index = []

        for i in range(len(opinion_matrix[0])):
            temp_value.append( min(results[i]))
            temp_index.append( results[i].index(min(results[i])))
        bestFeature = temp_value.index(min(temp_value))
        best_split_point = temp_index[bestFeature]

        #for feature_index in range(len(opinion_matrix[0])):
        #    # split_values[feature_index] = results[feature_index].get()
        #    split_values[feature_index] = results[feature_index]
        # pool.close()
        # pool.join()

        #bestFeature = np.argmin(split_values)
        print("bestFeature index: ", bestFeature)
        print("Split point:", best_split_point)
        t2 = time.time()
        print("Time used to create the layer: ", t2 - t1)

        # Store the feature_index for the current_node
        current_node.feature_index = bestFeature
        current_node.split_point = best_split_point

        # Split the rating_matrix into like, dislike and unknown
        (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, bestFeature, best_split_point)
        split_value = opt.cal_splitvalue(rating_matrix, item_vectors, current_node.vector,
                                         indices_like, indices_dislike, indices_unknown, K)

        like = rating_matrix[indices_like]
        like_op = opinion_matrix[indices_like]

        dislike = rating_matrix[indices_dislike]
        dislike_op = opinion_matrix[indices_dislike]

        unknown = rating_matrix[indices_unknown]
        unknown_op = opinion_matrix[indices_unknown]

        # Calculate the User Profile Vector for each of the three classes
        # print "optimizing like, dislike and unknown..."

        # Calculate the User Profile Vector for each of the three classes
        like_vector = current_node.vector
        dislike_vector = current_node.vector
        unknown_vector = current_node.vector
        if len(indices_like) > 0:
            like_vector = opt.cf_user(rating_matrix, item_vectors, current_node.vector, indices_like, K)
        if len(indices_dislike) > 0:
            dislike_vector = opt.cf_user(rating_matrix, item_vectors, current_node.vector, indices_dislike, K)
        if len(indices_unknown) > 0:
            unknown_vector = opt.cf_user(rating_matrix, item_vectors, current_node.vector, indices_unknown, K)

        # CONDITION check condition RMSE Error check is CORRECT
        if split_value < error_before:
            # Recursively call the fitTree_f function for like, dislike and unknown Nodes creation
            current_node.like = Node(current_node, current_node.depth + 1)
            current_node.like.vector = like_vector
            if len(like) != 0:
                self.fitTree_U(current_node.like, like_op, like, item_vectors, K)

            current_node.dislike = Node(current_node, current_node.depth + 1)
            current_node.dislike.vector = dislike_vector
            if len(dislike) != 0:
                self.fitTree_U(current_node.dislike, dislike_op, dislike, item_vectors, K)

            current_node.unknown = Node(current_node, current_node.depth + 1)
            current_node.unknown.vector = unknown_vector
            if len(unknown) != 0:
                self.fitTree_U(current_node.unknown, unknown_op, unknown, item_vectors, K)
        else:
            print("can't spilt")

    def fitTree_I(self, current_node, opinion_matrix, rating_matrix, user_vectors, K):
        # rating_matrix only consists of rows which are users corresponding to the current Node
        # Check if the maxDepth is reached
        t1 = time.time()
        if current_node.depth+1 > self.max_depth:
            return
        print("current depth of the tree", current_node.depth)
        if len(rating_matrix) == 0:
            return

        # Calulate the Error Before the Split
        print("Calculate error")
        error_before = opt.lossfunction_all(rating_matrix, current_node.vector, user_vectors, 0)
        print("Error Before: ", error_before)
        # Create a numy_array to hold the split_criteria Values
        NUMBER_OF_BIN = 5
        params = {}
        # pool = mp.Pool()
        count =0
        feature_splitpoint_matrix=[]
        for feature_index in range(len(opinion_matrix[0])):
            split_points = self.find_split_point( opinion_matrix, feature_index, NUMBER_OF_BIN)
            feature_splitpoint_matrix.append(split_points)
            for split_point in split_points:
                (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, feature_index, split_point)
                # Split the rating_matrix into like, dislike and unknown
                params[count] = []
                params[count].extend((rating_matrix, user_vectors, current_node.vector, indices_like, indices_dislike, indices_unknown, K))
                count += 1

        # Calculate the split criteria value
        print("Calculating the split criteria value")

        results = []
        params_index = 0
        for feature_index in range(len(opinion_matrix[0])):
            # result = pool.apply_async(opt.cal_splitvalue, params[feature_index])
            print("feature_index",feature_index)
            # t1 = time.time()
            temp = []
            for split_point in feature_splitpoint_matrix[feature_index]:
                print("split_point", split_point)
                result = opt.cal_splitvalueI(params[params_index][0], params[params_index][1], params[params_index][2],
                                             params[params_index][3], params[params_index][4], params[params_index][5],
                                             params[params_index][6])
                params_index += 1
                temp.append(result)
            results.append(temp)
            # t2 = time.time()
            # print("Time used to calculate the feature:", t2 - t1)

        #for feature_index in range(len(opinion_matrix[0])):
            # split_values[feature_index] = results[feature_index].get()
         #   split_values[feature_index] = results[feature_index]
        # pool.close()
        # pool.join()
        #results = np.array(results)
        temp_value = []
        temp_index = []

        for i in range(len(opinion_matrix[0])):
            temp_value.append(min(results[i]))
            temp_index.append(results[i].index(min(results[i])))
        bestFeature = temp_value.index(min(temp_value))
        best_split_point = temp_index[bestFeature]

        print("bestFeature index: ", bestFeature)
        print("Split point:", best_split_point)
        t2 = time.time()
        print("Time used to create the layer: ", t2 - t1)

        # Store the feature_index for the current_node
        current_node.feature_index = bestFeature
        current_node.split_point = best_split_point

        # Split the rating_matrix into like, dislike and unknown
        (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, bestFeature, best_split_point)
        split_value = opt.cal_splitvalueI(rating_matrix, user_vectors, current_node.vector, indices_like,
                                          indices_dislike, indices_unknown,K)
        like = rating_matrix[:, indices_like]
        like_op = opinion_matrix[indices_like]

        dislike = rating_matrix[:, indices_dislike]
        dislike_op = opinion_matrix[indices_dislike]

        unknown = rating_matrix[:, indices_unknown]
        unknown_op = opinion_matrix[indices_unknown]

        # Calculate the User Profile Vector for each of the three classes
        # print "optimizing like, dislike and unknown..."

        # Calculate the User Profile Vector for each of the three classes
        like_vector = current_node.vector
        dislike_vector = current_node.vector
        unknown_vector = current_node.vector

        if len(indices_like) > 0:
            like_vector = opt.cf_item(rating_matrix, user_vectors, current_node.vector, indices_like, K)
        if len(indices_dislike) > 0:
            dislike_vector = opt.cf_item(rating_matrix, user_vectors, current_node.vector, indices_dislike, K)
        if len(indices_unknown) > 0:
            unknown_vector = opt.cf_item(rating_matrix, user_vectors, current_node.vector, indices_unknown, K)

        # CONDITION check condition RMSE Error check is CORRECT
        if split_value < error_before:
            # Recursively call the fitTree_f function for like, dislike and unknown Nodes creation
            current_node.like = Node(current_node, current_node.depth + 1)
            current_node.like.vector = like_vector
            if len(like_op) != 0:
                self.fitTree_I(current_node.like, like_op, like, user_vectors, K)

            current_node.dislike = Node(current_node, current_node.depth + 1)
            current_node.dislike.vector = dislike_vector
            if len(dislike_op) != 0:
                self.fitTree_I(current_node.dislike, dislike_op, dislike, user_vectors, K)

            current_node.unknown = Node(current_node, current_node.depth + 1)
            current_node.unknown.vector = unknown_vector
            if len(unknown_op) != 0:
                self.fitTree_I(current_node.unknown, unknown_op, unknown, user_vectors, K)
        else:
            print("can't spilt")
