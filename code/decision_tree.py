import optimization as opt
import multiprocessing as mp
# import autograd.numpy as np
import numpy as np


# Split the users into left, right and empty Users by user-feature opinion
def split(data, feature_index):  # data should be opinion matrix
    # Get the indices for the when the opinion value is 1
    indices_left = np.where(data[:, feature_index] == 1.0)[0]

    # Get the indices for the when the opinion value is -1
    indices_right = np.where(data[:, feature_index] == -1.0)[0]

    # Get the indices for the when the opinion value is 0
    indices_empty = np.where(data[:, feature_index] == 0)[0]

    return indices_left, indices_right, indices_empty


# This class represents each Node of the Decision Tree
class Node:
    def __init__(self, parent_node, node_depth):
        # Each Node has a left, right and empty Child
        # It also stores the index of the item or feature
        # on which its splits the data
        self.parent = parent_node
        self.depth = node_depth
        self.left = None
        self.right = None
        self.empty = None
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
        while current_node.left != None or current_node.right != None or current_node.empty != None:
            if answers[current_node.feature_index] == 0:
                current_node = current_node.left
            elif answers[current_node.feature_index] == 1:
                current_node = current_node.right
            else:
                current_node = current_node.empty

        # return the user vector associated with the lead node
        # print("zzz", current_node.vector.shape)
        return current_node.vector

    def printtree(self, current_node):
        if current_node == None:
            print("None")
            return
        # print("Depth of current node:", current_node.depth, end=" ")
        print("Split feature of current node: ", current_node.feature_index)
        # print("left",end=" ")
        self.printtree(current_node.left)
        # print("right", end=" ")
        self.printtree(current_node.right)
        # print("empty",end=" ")
        self.printtree(current_node.empty)

    def getVectors_f(self, opinion_matrix, K):
        ultimate_vector = np.zeros((len(opinion_matrix), K))

        for i in range(len(opinion_matrix)):
            # Stores the user response
            # user response is a 1*F vector
            user_response = np.zeros(len(opinion_matrix[0]))

            # Get the responses using the opinion matrix
            for j in range(len(opinion_matrix[0])):
                if opinion_matrix[i][j] == 1:
                    user_response[j] = 0  # left
                elif opinion_matrix[i][j] == -1:
                    user_response[j] = 1  # right
                else:
                    user_response[j] = 2  # empty

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
            # Split the rating_matrix into left, right and empty
            (indices_left, indices_right, indices_empty) = split(opinion_matrix, feature_index)

            params[feature_index] = []
            params[feature_index].extend((rating_matrix, item_vectors, current_node.vector, indices_left, indices_right, indices_empty, K))

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

        # Split the rating_matrix into left, right and empty
        (indices_left, indices_right, indices_empty) = split(opinion_matrix, bestFeature)

        left = rating_matrix[indices_left]
        left_op = opinion_matrix[indices_left]

        right = rating_matrix[indices_right]
        right_op = opinion_matrix[indices_right]

        empty = rating_matrix[indices_empty]
        empty_op = opinion_matrix[indices_empty]

        # Calculate the User Profile Vector for each of the three classes
        # print "optimizing left, right and empty..."

        # Calculate the User Profile Vector for each of the three classes
        left_vector = current_node.vector
        right_vector = current_node.vector
        empty_vector = current_node.vector
        if len(indices_left) > 0:
            left_vector = opt.cf_user(rating_matrix, item_vectors, current_node.vector, indices_left, K)
        if len(indices_right) > 0:
            right_vector = opt.cf_user(rating_matrix, item_vectors, current_node.vector, indices_right, K)
        if len(indices_empty) > 0:
            empty_vector = opt.cf_user(rating_matrix, item_vectors, current_node.vector, indices_empty, K)

        # CONDITION check condition RMSE Error check is CORRECT
        if split_values[bestFeature] < error_before:
            # Recursively call the fitTree_f function for left, right and empty Nodes creation
            current_node.left = Node(current_node, current_node.depth + 1)
            current_node.left.vector = left_vector
            if len(left) != 0:
                self.fitTree_U(current_node.left, left_op, left, item_vectors, K)

            current_node.right = Node(current_node, current_node.depth + 1)
            current_node.right.vector = right_vector
            if len(right) != 0:
                self.fitTree_U(current_node.right, right_op, right, item_vectors, K)

            current_node.empty = Node(current_node, current_node.depth + 1)
            current_node.empty.vector = empty_vector
            if len(empty) != 0:
                self.fitTree_U(current_node.empty, empty_op, empty, item_vectors, K)
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
            # Split the rating_matrix into left, right and empty
            (indices_left, indices_right, indices_empty) = split(opinion_matrix, feature_index)

            params[feature_index] = []
            params[feature_index].extend((rating_matrix, user_vectors, current_node.vector, indices_left, indices_right, indices_empty, K))

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

        # Split the rating_matrix into left, right and empty
        (indices_left, indices_right, indices_empty) = split(opinion_matrix, bestFeature)

        left = rating_matrix[:, indices_left]
        left_op = opinion_matrix[indices_left]

        right = rating_matrix[:, indices_right]
        right_op = opinion_matrix[indices_right]

        empty = rating_matrix[:, indices_empty]
        empty_op = opinion_matrix[indices_empty]

        # Calculate the User Profile Vector for each of the three classes
        # print "optimizing left, right and empty..."

        # Calculate the User Profile Vector for each of the three classes
        left_vector = current_node.vector
        right_vector = current_node.vector
        empty_vector = current_node.vector

        if len(indices_left) > 0:
            left_vector = opt.cf_item(rating_matrix, user_vectors, current_node.vector, indices_left, K)
        if len(indices_right) > 0:
            right_vector = opt.cf_item(rating_matrix, user_vectors, current_node.vector, indices_right, K)
        if len(indices_empty) > 0:
            empty_vector = opt.cf_item(rating_matrix, user_vectors, current_node.vector, indices_empty, K)

        # CONDITION check condition RMSE Error check is CORRECT
        if split_values[bestFeature] < error_before:
            # Recursively call the fitTree_f function for left, right and empty Nodes creation
            current_node.left = Node(current_node, current_node.depth + 1)
            current_node.left.vector = left_vector
            if len(left_op) != 0:
                self.fitTree_I(current_node.left, left_op, left, user_vectors, K)

            current_node.right = Node(current_node, current_node.depth + 1)
            current_node.right.vector = right_vector
            if len(right_op) != 0:
                self.fitTree_I(current_node.right, right_op, right, user_vectors, K)

            current_node.empty = Node(current_node, current_node.depth + 1)
            current_node.empty.vector = empty_vector
            if len(empty_op) != 0:
                self.fitTree_I(current_node.empty, empty_op, empty, user_vectors, K)
        else:
            print("can't spilt")
