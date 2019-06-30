import optimization as opt
import multiprocessing as mp
import autograd.numpy as np


# Split the users into Like, Dislike and Unknown Users by user-feature opinion
def split(data, feature_index):  # data should be opinion matrix
    # Get the indices for the when the opinion value is 1
    indices_like = np.where(data[:, feature_index] == 1.0)[0]

    # Get the indices for the when the opinion value is -1
    indices_dislike = np.where(data[:, feature_index] == -1.0)[0]

    # Get the indices for the when the opinion value is 0
    indices_unknown = np.where(data[:, feature_index] == 0)[0]

    return indices_like, indices_dislike, indices_unknown


# This class represents each Node of the Decision Tree
class Node:
    def __init__(self, parent_node, node_depth):
        # Each Node has a Like, Dislike and Unknown Child
        # It also stores the index of the item or feature
        # on which its splits the data
        self.parent = parent_node
        self.depth = node_depth
        self.like = None
        self.dislike = None
        self.unknown = None
        self.feature_index = None  # add feature_index to splits the data
        self.vector = None


class Tree:
    # __init__() sets the root node, currentDepth and maxdepth of the tree
    def __init__(self, root_node, K, max_depth):
        self.root = root_node
        self.root.vector = np.random.rand(K)
        self.max_depth = max_depth

    # add! fucntion used to traverse a tree based on the feature opinion
    def traverse_f(self, answers):
        # answers is a 1*F vector
        # rand_prob[rand_prob > prob_failure] = 1
        current_node = self.root
        # print("Before")
        # Traverse the tree till you reach the leaf
        while current_node.like != None or current_node.dislike != None or current_node.unknown != None:
            if answers[current_node.feature_index] == 0:
                current_node = current_node.like
            elif answers[current_node.feature_index] == 1:
                current_node = current_node.dislike
            else:
                current_node = current_node.unknown

        # return the user vector associated with the lead node
        # print("zzz", current_node.vector.shape)
        return current_node.vector

    def printtree(self, current_node):
        if current_node == None:
            print("None")
            return
        # print("Depth of current node:", current_node.depth, end=" ")
        print("Split feature of current node: ", current_node.feature_index)
        # print("like",end=" ")
        self.printtree(current_node.like)
        # print("dislike", end=" ")
        self.printtree(current_node.dislike)
        # print("unknown",end=" ")
        self.printtree(current_node.unknown)

    def getVectors_f(self, opinion_matrix, K):
        ultimate_vector = np.zeros((len(opinion_matrix), K))

        for i in range(len(opinion_matrix)):
            # Stores the user response
            # user response is a 1*F vector
            user_response = np.zeros(len(opinion_matrix[0]))

            # Get the responses using the opinion matrix
            for j in range(len(opinion_matrix[0])):
                if opinion_matrix[i][j] == 1:
                    user_response[j] = 0  # like
                elif opinion_matrix[i][j] == -1:
                    user_response[j] = 1  # dislike
                else:
                    user_response[j] = 2  # unknown

            # Traverse the tree and store the vector associated with leaf node reached
            temp = self.traverse_f(user_response)
            # print("zz", temp.shape)
            ultimate_vector[i] = temp
        # return the  vector
        return ultimate_vector

    # recursively builds up the entire tree from the root Node
    def fitTree_U(self, current_node, opinion_matrix, rating_matrix, item_vectors, K):
        # rating_matrix only consists of rows which are users corresponding to the current Node
        # Check if the maxDepth is reached
        print("current depth of the tree", current_node.depth)
        if current_node.depth > self.max_depth:
            return
        if len(rating_matrix) == 0:
            return

        # Calulate the Error Before the Split
        print("Calculate error")
        error_before = opt.lossfunction_all(rating_matrix, item_vectors, current_node.vector, 1)

        print("Error Before: ", error_before)
        # Create a numy_array to hold the split_criteria Values
        split_values = np.zeros(len(opinion_matrix[0]))
        params = {}
        pool = mp.Pool()

        for feature_index in range(len(opinion_matrix[0])):
            # Split the rating_matrix into like, dislike and unknown
            (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, feature_index)

            params[feature_index] = []
            params[feature_index].extend((rating_matrix, item_vectors, current_node.vector, indices_like, indices_dislike, indices_unknown, K))

        # Calculate the split criteria value
        print("Calculating the split criteria value")
        results = []
        for feature_index in range(len(opinion_matrix[0])):
            result = pool.apply_async(opt.cal_splitvalue, params[feature_index])
            results.append(result)

        for feature_index in range(len(opinion_matrix[0])):
            # split_values[feature_index] = results[feature_index]
            split_values[feature_index] = results[feature_index].get()
        pool.close()
        pool.join()

        bestFeature = np.argmin(split_values)
        print("bestFeature index: ", bestFeature)

        # Store the feature_index for the current_node
        current_node.feature_index = bestFeature

        # Split the rating_matrix into like, dislike and unknown
        (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, bestFeature)

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
        if split_values[bestFeature] < error_before:
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
        print("current depth of the tree", current_node.depth)
        if current_node.depth > self.max_depth:
            return
        if len(rating_matrix) == 0:
            return

        # Calulate the Error Before the Split
        print("Calculate error")
        error_before = opt.lossfunction_all(rating_matrix, current_node.vector, user_vectors, 0)
        print("Error Before: ", error_before)
        # Create a numy_array to hold the split_criteria Values
        split_values = np.zeros(len(opinion_matrix[0]))
        params = {}
        pool = mp.Pool()

        for feature_index in range(len(opinion_matrix[0])):
            # Split the rating_matrix into like, dislike and unknown
            (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, feature_index)

            params[feature_index] = []
            params[feature_index].extend((rating_matrix, user_vectors, current_node.vector, indices_like, indices_dislike, indices_unknown, K))

        # Calculate the split criteria value
        print("Calculating the split criteria value")
        results = []
        for feature_index in range(len(opinion_matrix[0])):
            result = opt.cal_splitvalueI(params[feature_index][0], params[feature_index][1], params[feature_index][2], params[feature_index][3], params[feature_index][4], params[feature_index][5], params[feature_index][6])
            results.append(result)

        for feature_index in range(len(opinion_matrix[0])):
            split_values[feature_index] = results[feature_index]
        pool.close()
        pool.join()

        bestFeature = np.argmin(split_values)
        print("bestFeature index: ", bestFeature)

        # Store the feature_index for the current_node
        current_node.feature_index = bestFeature

        # Split the rating_matrix into like, dislike and unknown
        (indices_like, indices_dislike, indices_unknown) = split(opinion_matrix, bestFeature)

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
        if split_values[bestFeature] < error_before:
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
