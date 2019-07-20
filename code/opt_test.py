import numpy as np
import time
from numba import jit, autojit
# import multiprocessing as mp


lmd_BPR = 100
lmd_u = 0.1
lmd_v = 0.1
NUM_USER = 7920
NUM_ITEM = 13428
EXP = 2.718281828459045
blockdim = (32, 32)


def lossfunction_all(rating_matrix, movie_vectors, user_vectors, flag):
    if flag == 1:  # used in user tree construction, user_vectors is a 1*K vector
        user_vectors = np.array([user_vectors for i in range(len(rating_matrix))])
    if flag == 0:  # used in item tree construction, movie_vectors is a 1*K vector
        movie_vectors = np.array([movie_vectors for i in range(rating_matrix.shape[1])])
    value = 0
    # Add regularization term
    user_l = user_vectors[np.nonzero(user_vectors)]
    value = value + lmd_u * np.dot(user_l, user_l)
    mov_l = movie_vectors[np.nonzero(movie_vectors)]
    value = value + lmd_v * np.dot(mov_l, mov_l)

    if len(rating_matrix) == 0:
        return value

    predict = np.dot(user_vectors, movie_vectors.T)
    P = predict[np.nonzero(rating_matrix)]
    R = rating_matrix[np.nonzero(rating_matrix)]
    Err = P - R
    value = value + np.dot(Err, Err)

    np.random.seed(0)
    num_pair = 60
    num_user, num_item = rating_matrix.shape
    for i in range(num_pair):
        c1, c2 = np.random.randint(0, num_item * num_user, 2)
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item
        if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
            diff = np.dot(user_vectors[u1, :].T, movie_vectors[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                movie_vectors[i2, :])
            diff = -diff
            value = value + lmd_BPR * np.log(1 + np.exp(diff))

    return value


@jit
def get_user_gradient(selected_points, selected_pairs, rating_matrix, user_vector, movie_vectors, lmd_u, lmd_BPR):
    num_user, num_item = rating_matrix.shape
    delta_u = 0
    for sp in selected_points:
        u1, i1 = sp // num_item, sp % num_item
        if rating_matrix[u1, i1] != 0:
            pred = 0
            for i in range(len(user_vector)):
                pred += user_vector[i] * movie_vectors[i1, i]
            delta_u += -2 * (rating_matrix[u1, i1] - pred) * movie_vectors[i1] + 2 * lmd_u * user_vector

    for j in range(int(len(selected_pairs) / 2)):
        c1 = selected_pairs[j * 2]
        c2 = selected_pairs[j * 2 + 1]
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item

        if rating_matrix[u1, i1] > rating_matrix[u2, i2]:
            diff = 0
            for i in range(len(user_vector)):
                diff += - user_vector[i] * (movie_vectors[i1, i] - movie_vectors[i2, i])
            vec_diff = 0
            vec_diff = movie_vectors[i2] - movie_vectors[i1]
            delta_u += lmd_BPR * vec_diff * pow(EXP, diff) / (1 + pow(EXP, diff))

    return delta_u


def selfgradu(rating_matrix, movie_vectors, current_vector, user_vector, i):
    delta_u = np.zeros_like(user_vector)
    num_point = 3000
    num_pair = 600
    num_user, num_item = rating_matrix.shape
    np.random.seed(i)
    user_vector = user_vector + current_vector
    if len(rating_matrix) == 0:
        return delta_u

    selected_points = np.random.randint(0, num_item * num_user, num_point)
    selected_pairs = np.random.randint(0, num_item * num_user, num_pair * 2)
    delta_u = get_user_gradient(selected_points, selected_pairs, rating_matrix, user_vector, movie_vectors, lmd_u, lmd_BPR)

    return delta_u


@jit
def get_item_gradient(selected_points, selected_pairs, rating_matrix, user_vectors, movie_vector, lmd_v, lmd_BPR):
    num_user, num_item = rating_matrix.shape
    delta_v = 0
    for sp in selected_points:
        u1, i1 = sp // num_item, sp % num_item
        if rating_matrix[u1, i1] != 0:
            pred = 0
            for i in range(len(movie_vector)):
                pred += user_vectors[u1, i] * movie_vector[i]
            delta_v += -2 * (rating_matrix[u1, i1] - pred) * user_vectors[u1] + 2 * lmd_v * movie_vector

    for j in range(int(len(selected_pairs) / 2)):
        c1 = selected_pairs[j * 2]
        c2 = selected_pairs[j * 2 + 1]
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item

        if rating_matrix[u1, i1] > rating_matrix[u2, i2]:
            diff = 0
            for i in range(len(movie_vector)):
                diff += -movie_vector[i] * (user_vectors[u1, i] - user_vectors[u2, i])
            vec_diff = 0
            vec_diff = user_vectors[u2] - user_vectors[u1]
            delta_v += lmd_BPR * vec_diff * pow(EXP, diff) / (1 + pow(EXP, diff))

    return delta_v


def selfgradv(rating_matrix, movie_vector, current_vector, user_vectors, i):
    delta_v = np.zeros_like(movie_vector)
    num_point = 3000
    num_pair = 600
    num_user, num_item = rating_matrix.shape
    np.random.seed(i)
    movie_vector = movie_vector + current_vector
    if len(rating_matrix) == 0:
        return delta_v

    selected_points = np.random.randint(0, num_item * num_user, num_point)
    selected_pairs = np.random.randint(0, num_item * num_user, num_pair * 2)
    delta_v = get_item_gradient(selected_points, selected_pairs, rating_matrix, user_vectors, movie_vector, lmd_v, lmd_BPR)

    return delta_v


def cf_user(rating_matrix, item_vectors, current_vector, indices, K):
    np.random.seed(0)
    user_vector = np.zeros_like(current_vector)
    index_matrix = rating_matrix[indices]
    num_iter = 20
    eps = 1e-8
    lr = 0.1
    # set the variable user_vector to be gradient
    # mg = grad(lossfunction, argnum=2)
    sum_square_u = eps + np.zeros_like(user_vector)

    # SGD procedure:
    for i in range(num_iter):
        delta_u = selfgradu(index_matrix, item_vectors, current_vector, user_vector, i)
        sum_square_u += np.square(delta_u)
        lr_u = np.divide(lr, np.sqrt(sum_square_u))
        #print(np.dot(lr_u * delta_u,lr_u * delta_u))
        user_vector -= lr_u * delta_u
        #final_vector = user_vector+current_vector
        #error=lossfunction_all(index_matrix,item_vectors,final_vector,1)
        #print("sgderror",error)
    user_vector = user_vector + current_vector

    return user_vector


def cf_item(rating_matrix, user_vectors, current_vector, indices, K):
    np.random.seed(0)
    movie_vector = np.zeros_like(current_vector)
    index_matrix = rating_matrix[:, indices]
    num_iter = 100
    eps = 1e-8
    lr = 0.1
    sum_square_v = eps + np.zeros_like(movie_vector)

    # SGD procedure:
    for i in range(num_iter):
        delta_v = selfgradv(index_matrix, movie_vector, current_vector, user_vectors, i)
        sum_square_v += np.square(delta_v)
        lr_v = np.divide(lr, np.sqrt(sum_square_v))
        movie_vector -= lr_v * delta_v
        #final_vector = movie_vector + current_vector
        #error=lossfunction_all(index_matrix, final_vector,user_vectors, 0)
        #print("sgderror", error)
    movie_vector = movie_vector + current_vector

    return movie_vector


@jit
def get_error_user(rating_matrix, sub_matrix, fixed_vectors, current_vector, indices, K):
    vector = cf_user(rating_matrix, fixed_vectors, current_vector, indices, K)
    vector = np.repeat(vector.reshape(1, -1), len(indices), axis=0)
    pred = np.dot(vector, fixed_vectors.T)
    P = pred[np.nonzero(sub_matrix)]
    R = sub_matrix[np.nonzero(sub_matrix)]
    Err = np.linalg.norm(P-R) ** 2
    return vector, Err


@jit
def get_error_item(rating_matrix, sub_matrix, fixed_vectors, current_vector, indices, K):
    vector = cf_item(rating_matrix, fixed_vectors, current_vector, indices, K)
    vector = np.repeat(vector.reshape(1, -1), len(indices), axis=0)
    pred = np.dot(fixed_vectors, vector.T)
    P = pred[np.nonzero(sub_matrix)]
    R = sub_matrix[np.nonzero(sub_matrix)]
    Err = np.linalg.norm(P-R) ** 2
    return vector, Err


@jit
def prepare_item_data(rating_matrix, indices_like, indices_dislike, indices_unknown):
    return rating_matrix[:, indices_like], rating_matrix[:, indices_dislike], rating_matrix[:, indices_unknown]


@jit
def prepare_user_data(rating_matrix, indices_like, indices_dislike, indices_unknown):
    return rating_matrix[indices_like], rating_matrix[indices_dislike], rating_matrix[indices_unknown]


def cal_splitvalue(rating_matrix, movie_vectors, current_vector, indices_like, indices_dislike, indices_unknown, K):
    like, dislike, unknown = prepare_user_data(rating_matrix, indices_like, indices_dislike, indices_unknown)
    value = 0.0
    #print("like")
    if len(indices_like) > 0:
        like_vector, like_error = get_error_user(rating_matrix, like, movie_vectors, current_vector, indices_like, K)
    else:
        like_vector = current_vector
        like_error = 0
    #print("dislike")
    if len(indices_dislike) > 0:
        dislike_vector, dislike_error = get_error_user(rating_matrix, dislike, movie_vectors, current_vector, indices_dislike, K)
    else:
        dislike_vector = current_vector
        dislike_error = 0
    #print("unknown")
    if len(indices_unknown) > 0:
        unknown_vector, unknown_error = get_error_user(rating_matrix, unknown, movie_vectors, current_vector, indices_unknown, K)
    else:
        unknown_vector = current_vector
        unknown_error = 0

    value += like_error + dislike_error + unknown_error
    #print("like dislike unknown",value)
    value += lmd_v * (np.linalg.norm(like_vector) ** 2 + np.linalg.norm(dislike_vector) ** 2 + np.linalg.norm(unknown_vector) ** 2)
    value += lmd_u * (np.linalg.norm(movie_vectors) ** 2)

    np.random.seed(0)
    num_pair = 20

    num_user, num_item = like.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if like[u1][i1] > like[u2][i2]:
                diff = np.dot(like_vector[u1], movie_vectors[i1] - movie_vectors[i2])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = dislike.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if dislike[u1][i1] > dislike[u2][i2]:
                diff = np.dot(dislike_vector[u1], movie_vectors[i1] - movie_vectors[i2])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = unknown.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if unknown[u1][i1] > unknown[u2][i2]:
                diff = np.dot(unknown_vector[u1], movie_vectors[i1] - movie_vectors[i2])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    print(value)
    return value


def cal_splitvalueI(rating_matrix, user_vectors, current_vector, indices_like, indices_dislike, indices_unknown, K):
    # like = rating_matrix[:, indices_like]
    # dislike = rating_matrix[:, indices_dislike]
    # unknown = rating_matrix[:, indices_unknown]
    like, dislike, unknown = prepare_item_data(rating_matrix, indices_like, indices_dislike, indices_unknown)
    value = 0.0
    if len(indices_like) > 0:
        like_vector, like_error = get_error_item(rating_matrix, like, user_vectors, current_vector, indices_like, K)
    else:
        like_vector = current_vector
        like_error = 0
    if len(indices_dislike) > 0:
        dislike_vector, dislike_error = get_error_item(rating_matrix, dislike, user_vectors, current_vector, indices_dislike, K)
    else:
        dislike_vector = current_vector
        dislike_error = 0
    if len(indices_unknown) > 0:
        unknown_vector, unknown_error = get_error_item(rating_matrix, unknown, user_vectors, current_vector, indices_unknown, K)
    else:
        unknown_vector = current_vector
        unknown_error = 0
    value += like_error + dislike_error + unknown_error

    value += lmd_v * (np.linalg.norm(like_vector) ** 2 + np.linalg.norm(dislike_vector) ** 2 + np.linalg.norm(unknown_vector) ** 2)
    value += lmd_u * (np.linalg.norm(user_vectors) ** 2)

    np.random.seed(0)
    num_pair = 20

    num_user, num_item = like.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if like[u1][i1] > like[u2][i2]:
                diff = np.dot(user_vectors[u1] - user_vectors[u2], like_vector[i1])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = dislike.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if dislike[u1][i1] > dislike[u2][i2]:
                diff = np.dot(user_vectors[u1] - user_vectors[u2], dislike_vector[i1])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = unknown.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if unknown[u1][i1] > unknown[u2][i2]:
                diff = np.dot(user_vectors[u1] - user_vectors[u2], unknown_vector[i1])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    print(value)
    return value


@jit
def matmul(matrix1, matrix2, rmatrix):
    for i in range(len(matrix1)):
        for j in range(len(matrix2)):
            for k in range(len(matrix2[0])):
                rmatrix[i][j] += matrix1[i, k] * matrix2[k, j]


# @cuda.jit
# def matmul_cuda(A, B, C):
#     """Perform square matrix multiplication of C = A * B
#     """
#     i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#     j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
#     if i < C.shape[0] and j < C.shape[1]:
#         tmp = 0.
#         for k in range(A.shape[1]):
#             tmp += A[i, k] * B[k, j]
#         C[i, j] = tmp
    # cuda.syncthreads()


# @cuda.jit
# def vec_inner_cuda(vec_1, vec_2, vec_3):
#     i = cuda.grid(1)
#     if i < len(vec_1):
#         vec_3[i] = vec_1[i] * vec_2[i]
#     cuda.syncthreads()


@jit
def vec_inner(vec_1, vec_2):
    r = 0
    for i in range(len(vec_1)):
        r += vec_1[i] * vec_2[i]
    return r


# @cuda.reduce
# def sum_reduce(a, b):
#     return a + b


# @jit
# def calculate_error(user_vector, movie_vectors, pred, rating):
#     for i in range(len(user_vector)):
#         for j in range(len(movie_vectors)):
#             for k in range(len(movie_vectors[0])):
#                 pred[i, j] += user_vector[i, k] * movie_vectors[j, k]
#     mask = rating != 0
#     err = (pred - rating)[mask]
#     r = 0
#     for i in range(len(err)):
#         r += pow(err[i], 2)

#     return r
