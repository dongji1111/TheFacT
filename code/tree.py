import optimization as opt
import multiprocessing as mp
# import autograd.numpy as np
import numpy as np
from numpy import linalg as LA

from copy_reg import pickle
from types import MethodType


class Node:
    def __init__(self, parent_node, node_depth):
        self.feature_index = None 
        self.parent = parent_node
        self.depth = node_depth
        self.left = None
        self.right = None
        self.empty = None
        self.vector = None

class Tree:
    def __init__(self, root_node, rating_matrix, opinion_matrix, anchor_vectors, lr, num_dim, max_depth, num_BPRpairs, lambda_anchor, lambda_target, lambda_BPR, num_iter, batch_size, random_seed):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.root = root_node
        self.root.vector = np.random.rand(num_dim)
        self.rating_matrix = rating_matrix
        self.num_target, self.num_anchor = rating_matrix.shape
        self.opinion_matrix = opinion_matrix
        self.num_feature = opinion_matrix.shape[1]
        self.num_dim = num_dim
        self.max_depth = max_depth
        self.num_BPRpairs = num_BPRpairs
        self.lambda_anchor = lambda_anchor
        self.lambda_target = lambda_target
        self.lambda_BPR = lambda_BPR
        self.anchor_vectors = anchor_vectors
        self.mask_matrix = np.nonzero(rating_matrix)
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.lr = lr

    def personalization(self, vector):
        for i in range(self.num_target):
            # print "before: ", vector[i]
            user_rating = self.rating_matrix[np.array([i])]
            vector[i] = self.sgd_update(user_rating, vector[i])
            # print "after: ", vector[i]
        return vector

    def calculate_loss(self, current_node, rating_matrix):
        np.random.seed(self.random_seed)

        target_vectors = np.array([current_node.vector for i in range(rating_matrix.shape[0])])
        value = 0

        target_reg = LA.norm(target_vectors) ** 2
        anchor_reg = LA.norm(self.anchor_vectors) ** 2
        value += self.lambda_anchor * anchor_reg + self.lambda_target * target_reg

        mask_matrix = np.nonzero(rating_matrix)
        pred = np.dot(target_vectors, self.anchor_vectors.T)
        value += LA.norm((rating_matrix - pred)[mask_matrix]) ** 2
        value += self.get_BPR(rating_matrix, target_vectors)

        return value

    def split(self, opinion_matrix, feature_index):
        index_left = np.where(opinion_matrix[:, feature_index] == 1.0)[0]
        index_right = np.where(opinion_matrix[:, feature_index] == -1.0)[0]
        index_empty = np.where(opinion_matrix[:, feature_index] == 0)[0]
        
        return index_left, index_right, index_empty

    def sgd(self, rating_matrix, target_vector):
        delta = np.zeros_like(target_vector)
        num_t, num_a = rating_matrix.shape
        if num_t == 0:
            return delta

        np.random.seed(self.random_seed)
        # target_vector += current_vector
        for i in range(self.batch_size):
            idx = np.random.randint(0, num_t * num_a)
            t, a = idx // num_a, idx % num_a
            if rating_matrix[t, a] != 0:
                delta += -2 * (rating_matrix[t, a] - np.dot(target_vector, self.anchor_vectors[a])) * self.anchor_vectors[a] + 2 * self.lambda_target * target_vector

        for i in range(self.num_BPRpairs):
            id1, id2 = np.random.randint(0, num_t * num_a, 2)
            t1, a1 = id1 // num_a, id1 % num_a
            t2, a2 = id2 // num_a, id2 % num_a
            if rating_matrix[t1, a1] > rating_matrix[t2, a2]:
                diff = np.dot(target_vector.T, self.anchor_vectors[a1] - self.anchor_vectors[a2])
                diff = -diff
                delta += self.lambda_BPR * (self.anchor_vectors[a2] - self.anchor_vectors[a1]) * np.exp(diff) / (1 + np.exp(diff))
        
        return delta

    def sgd_update(self, index_matrix, current_vector):
        if len(index_matrix) <= 0:
            return current_vector
        np.random.seed(self.random_seed)
        target_vector = np.random.random(size=current_vector.shape)

        eps = 1e-8
        sum_square = eps + np.zeros_like(target_vector)
        # SGD procedure
        for i in range(self.num_iter):
            delta = self.sgd(index_matrix, current_vector + target_vector)
            sum_square += np.square(delta)
            lr_t = np.divide(self.lr, np.sqrt(sum_square))
            target_vector -= lr_t * delta
        target_vector += current_vector
        
        return target_vector
            
    def calculate_subtree_value(self, index_matrix, current_vector):
        mmatrix = np.nonzero(index_matrix)
        vector = self.sgd_update(index_matrix, current_vector)
        vector = np.array([vector for i in range(index_matrix.shape[0])])
        pred = np.dot(vector, self.anchor_vectors.T)
        err = LA.norm((pred - index_matrix)[mmatrix]) ** 2

        return err, vector

    def calculate_splitvalue(self, rating_matrix, current_vector, index_left, index_right, index_empty):
        left = rating_matrix[index_left]
        right = rating_matrix[index_right]
        empty = rating_matrix[index_empty]
        left_vector = np.zeros(self.num_dim)
        right_vector = np.zeros(self.num_dim)
        empty_vector = np.zeros(self.num_dim) 
        value = 0

        if len(index_left) > 0:
            err, left_vector = self.calculate_subtree_value(left, current_vector)
            value += err
        if len(index_right) > 0:
            err, right_vector = self.calculate_subtree_value(right, current_vector)
            value += err
        if len(index_empty) > 0:
            err, empty_vector = self.calculate_subtree_value(empty, current_vector)
            value += err

        value += self.lambda_target * (LA.norm(left_vector) ** 2 + LA.norm(right_vector) ** 2 + LA.norm(empty_vector) ** 2)
        value += self.lambda_anchor * (LA.norm(self.anchor_vectors) ** 2)
        value += self.get_BPR(left, left_vector)
        value += self.get_BPR(right, right_vector)
        value += self.get_BPR(empty, empty_vector)
        
        return value

    def get_BPR(self, rating_matrix, target_vector):
        np.random.seed(self.random_seed)
        num_t, num_a = rating_matrix.shape
        if num_t * num_a == 0:
            return 0 
        value = 0
        for i in range(self.num_BPRpairs):
            p1, p2 = np.random.randint(0, num_t * num_a, 2)
            t1, a1 = p1 // num_a, p1 % num_a
            t2, a2 = p2 // num_a, p2 % num_a
            # print "pair index: ",t2, a2
            if rating_matrix[t1, a1] > rating_matrix[t2, a2]:
                diff = np.dot(target_vector[t1].T, self.anchor_vectors[a1]) - np.dot(target_vector[t2].T, self.anchor_vectors[a2])
                diff = -diff
                value += self.lambda_BPR * np.log(1 + np.exp(diff))
        
        return value

    def print_tree(self, current_node, level=0):
        if current_node == None:
            # print '\t' * level, "None"
            return
        print '\t' * level, current_node.feature_index
        self.print_tree(current_node.left, level + 1)
        self.print_tree(current_node.right, level + 1)
        self.print_tree(current_node.empty, level + 1)

    def get_vectors(self):
        vectors = np.zeros((self.num_target, self.num_dim))
        for i in range(self.num_target):
            current_node = self.root

            while current_node.left != None or current_node.right != None or current_node.empty != None:
                next = {}
                next[1] = current_node.left
                next[-1] = current_node.right
                next[0] = current_node.empty
                current_node = next[self.opinion_matrix[i][current_node.feature_index]]
            vectors[i] = current_node.vector
        return vectors

    def __call__(self, rating_matrix, current_vector, index_left, index_right, index_empty):
        r = self.calculate_splitvalue(rating_matrix, current_vector, index_left, index_right, index_empty)
        return r

    def create_tree(self, current_node, opinion_matrix, rating_matrix):
        print "Current depth: ", current_node.depth
        if current_node.depth > self.max_depth:
            print ">>>>>>>>>>>>>>>>>>> STOP: tree depth exceeds the maximum limit."
            return
        if len(rating_matrix) == 0:
            print ">>>>>>>>>>>>>>>>>>> STOP: No rating matrix."
            return

        # print ("Calculate constructing error in previous stage")
        error_old = self.calculate_loss(current_node, rating_matrix)
        # print ("Loss before updating the tree: ", error_old)

        # record the features
        split_values = np.zeros(self.num_feature)
        index_left, index_right, index_empty = self.split(opinion_matrix, 1)
        
        # ####################################################
        # initialize the setting for multi processing
        # pickle(MethodType, _pickle_method, _unpickle_method)
        params = {}
        pool = mp.Pool(1)
        for feature_index in range(self.num_feature):
            index_left, index_right, index_empty = self.split(opinion_matrix, feature_index)
            params[feature_index] = []
            params[feature_index].extend((rating_matrix, current_node.vector, index_left, index_right, index_empty))

        # print ("Calculate the split criteria value")
        results = []
        # calculate the the split value in parallel
        for feature_index in range(self.num_feature):
            result = pool.apply_async(self, params[feature_index])
            results.append(result)
        # retrieve the calculated results into split_values
        for feature_index in range(self.num_feature):
            split_values[feature_index] = results[feature_index].get()
        pool.close()
        pool.join()
        ####################################################

        best_feature = np.argmin(split_values)
        current_node.feature_index = best_feature
        print "Best feature index: ", best_feature

        # create the child nodes with the best feature
        index_left, index_right, index_empty = self.split(opinion_matrix, best_feature)
        left_rating_matrix, left_opinion_matrix = rating_matrix[index_left], opinion_matrix[index_left]
        right_rating_matrix, right_opinion_matrix = rating_matrix[index_right], opinion_matrix[index_right]
        empty_rating_matrix, empty_opinion_matrix = rating_matrix[index_empty], opinion_matrix[index_empty]

        left_vector = self.sgd_update(left_rating_matrix, current_node.vector)
        right_vector = self.sgd_update(right_rating_matrix, current_node.vector)
        empty_vector = self.sgd_update(empty_rating_matrix, current_node.vector)

        if split_values[best_feature] < error_old:
            # left child tree
            current_node.left = Node(parent_node=current_node, node_depth=current_node.depth + 1)
            current_node.left.vector = left_vector
            if len(left_rating_matrix) != 0:
                self.create_tree(current_node.left, left_opinion_matrix, left_rating_matrix)
            # right child tree
            current_node.right = Node(parent_node=current_node, node_depth=current_node.depth + 1)
            current_node.right.vector = right_vector
            if len(right_rating_matrix) != 0:
                self.create_tree(current_node.right, right_opinion_matrix, right_rating_matrix)
            # empty child tree
            current_node.empty = Node(parent_node=current_node, node_depth=current_node.depth + 1)
            current_node.empty.vector = empty_vector
            if len(empty_rating_matrix) != 0:
                self.create_tree(current_node.empty, empty_opinion_matrix, empty_rating_matrix)
        else:
            print ">>>>>>>>>>>>>>>>>>> STOP: cannot not be split any more."