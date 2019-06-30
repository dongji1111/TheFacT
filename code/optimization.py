import autograd.numpy as np
# from autograd import grad
# import random

lmd_BPR = 100
lmd_u = 1
lmd_v = 1


def lossfunction_all(rating_matrix, item_vectors, user_vectors, flag):
    if flag == 1:  # used in user tree construction, user_vectors is a 1*K vector
        user_vectors = np.array([user_vectors for i in range(len(rating_matrix))])
    if flag == 0:  # used in item tree construction, item_vectors is a 1*K vector
        item_vectors = np.array([item_vectors for i in range(rating_matrix.shape[1])])
    value = 0
    # Add regularization term
    user_l = user_vectors[np.nonzero(user_vectors)]
    value = value + lmd_u * np.dot(user_l, user_l)
    item_l = item_vectors[np.nonzero(item_vectors)]
    value = value + lmd_v * np.dot(item_l, item_l)

    if len(rating_matrix) == 0:
        return value

    predict = np.dot(user_vectors, item_vectors.T)
    P = predict[np.nonzero(rating_matrix)]
    R = rating_matrix[np.nonzero(rating_matrix)]
    Err = P - R
    value = value + np.dot(Err, Err)

    np.random.seed(0)
    num_pair = 20
    num_user, num_item = rating_matrix.shape
    for i in range(num_pair):
        c1, c2 = np.random.randint(0, num_item * num_user, 2)
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item
        if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
            diff = np.dot(user_vectors[u1, :].T, item_vectors[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                               item_vectors[i2, :])
            diff = -diff
            value = value + lmd_BPR * np.log(1 + np.exp(diff))

    return value


def selfgradu(rating_matrix, item_vectors, current_vector, user_vector):
    delta_u = np.zeros_like(user_vector)
    num_point = 100
    num_pair = 20
    num_user, num_item = rating_matrix.shape
    np.random.seed(0)
    user_vector = user_vector + current_vector
    if len(rating_matrix) == 0:
        return delta_u
    for i in range(num_point):
        c1 = np.random.randint(0, num_item * num_user)
        u1, i1 = c1 // num_item, c1 % num_item
        if rating_matrix[u1][i1] != 0:
            delta_u += -2 * (rating_matrix[u1][i1] - np.dot(user_vector, item_vectors[i1])) * item_vectors[i1] + 2 * lmd_u * user_vector

    for i in range(num_pair):
        c1, c2 = np.random.randint(0, num_item * num_user, 2)
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item
        if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
            diff = np.dot(user_vector.T, item_vectors[i1, :]) - np.dot(user_vector.T, item_vectors[i2, :])
            diff = -diff
            delta_u += lmd_BPR * (item_vectors[i2, :] - item_vectors[i1, :]) * np.exp(diff) / (1 + np.exp(diff))
    return delta_u


def selfgradv(rating_matrix, item_vector, current_vector, user_vectors):
    delta_v = np.zeros_like(item_vector)
    num_point = 100
    num_pair = 20
    num_user, num_item = rating_matrix.shape
    np.random.seed(0)
    item_vector = item_vector + current_vector
    if len(rating_matrix) == 0:
        return delta_v
    for i in range(num_point):
        c1 = np.random.randint(0, num_item * num_user)
        u1, i1 = c1 // num_item, c1 % num_item
        if rating_matrix[u1][i1] != 0:
            delta_v += -2 * (rating_matrix[u1][i1] - np.dot(user_vectors[u1], item_vector)) * user_vectors[u1] + 2 * lmd_v * item_vector
    for i in range(num_pair):
        c1, c2 = np.random.randint(0, num_item * num_user, 2)
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item
        if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
            diff = np.dot(user_vectors[u1].T, item_vector) - np.dot(user_vectors[u2].T, item_vector)
            diff = -diff
            delta_v += lmd_BPR * (user_vectors[u2, :] - user_vectors[u1, :]) * np.exp(diff) / (1 + np.exp(diff))
    return delta_v


def cf_user(rating_matrix, item_vectors, current_vector, indices, K):
    # user_vector is 1*K vector
    np.random.seed(0)
    user_vector = np.random.random(size=current_vector.shape)
    index_matrix = rating_matrix[indices]
    num_iter = 20
    eps = 1e-8
    lr = 0.1
    # set the variable user_vector to be gradient
    # mg = grad(lossfunction, argnum=2)
    sum_square_u = eps + np.zeros_like(user_vector)

    # SGD procedure:
    for i in range(num_iter):
        delta_u = selfgradu(index_matrix, item_vectors, current_vector, user_vector)
        sum_square_u += np.square(delta_u)
        lr_u = np.divide(lr, np.sqrt(sum_square_u))
        user_vector -= lr_u * delta_u

    user_vector = user_vector + current_vector
    return user_vector


def cf_item(rating_matrix, user_vectors, current_vector, indices, K):
    np.random.seed(0)
    item_vector = np.random.random(size=current_vector.shape)
    rating_matrix = rating_matrix[:, indices]
    num_iter = 1000
    eps = 1e-8
    lr = 0.1
    sum_square_v = eps + np.zeros_like(item_vector)

    # SGD procedure:
    for i in range(num_iter):
        delta_v = selfgradv(rating_matrix, item_vector, current_vector, user_vectors)
        sum_square_v += np.square(delta_v)
        lr_v = np.divide(lr, np.sqrt(sum_square_v))
        item_vector -= lr_v * delta_v
    item_vector = item_vector + current_vector

    return item_vector


def cal_splitvalue(rating_matrix, item_vectors, current_vector, indices_like, indices_dislike, indices_unknown, K):
    like = rating_matrix[indices_like]
    dislike = rating_matrix[indices_dislike]
    unknown = rating_matrix[indices_unknown]
    like_vector = np.zeros(K)
    dislike_vector = np.zeros(K)
    unknown_vector = np.zeros(K)
    value = 0.0

    if len(indices_like) > 0:
        # print(indices_like)
        like_vector = cf_user(rating_matrix, item_vectors, current_vector, indices_like, K)
        like_vector = np.array([like_vector for i in range(len(indices_like))])
        pre_like = np.dot(like_vector, item_vectors.T)
        Err_like = pre_like[np.nonzero(like)] - like[np.nonzero(like)]
        value = value + np.dot(Err_like, Err_like)

    if len(indices_dislike) > 0:
        # print(indices_dislike)
        dislike_vector = cf_user(rating_matrix, item_vectors, current_vector, indices_dislike, K)
        dislike_vector = np.array([dislike_vector for i in range(len(indices_dislike))])
        pre_dislike = np.dot(dislike_vector, item_vectors.T)
        Err_dislike = pre_dislike[np.nonzero(dislike)] - dislike[np.nonzero(dislike)]
        value = value + np.dot(Err_dislike, Err_dislike)

    if len(indices_unknown) > 0:
        # print(indices_unknown)
        unknown_vector = cf_user(rating_matrix, item_vectors, current_vector, indices_unknown, K)
        unknown_vector = np.array([unknown_vector for i in range(len(indices_unknown))])
        pre_unknown = np.dot(unknown_vector, item_vectors.T)
        Err_unknown = pre_unknown[np.nonzero(unknown)] - unknown[np.nonzero(unknown)]
        value = value + np.dot(Err_unknown, Err_unknown)

    lkv_l = like_vector[np.nonzero(like_vector)]
    dlkv_l = dislike_vector[np.nonzero(dislike_vector)]
    unkv_l = unknown_vector[np.nonzero(unknown_vector)]
    value = value + lmd_u * (np.dot(lkv_l, lkv_l) + np.dot(dlkv_l, dlkv_l) + np.dot(unkv_l, unkv_l))

    item_l = item_vectors[np.nonzero(item_vectors)]
    value = value + lmd_v * np.dot(item_l, item_l)

    np.random.seed(0)
    num_pair = 20

    num_user, num_item = like.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if like[u1][i1] > like[u2][i2]:
                diff = np.dot(like_vector[u1, :].T, item_vectors[i1, :]) - np.dot(like_vector[u2, :].T,
                                                                                  item_vectors[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = dislike.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if dislike[u1][i1] > dislike[u2][i2]:
                diff = np.dot(dislike_vector[u1, :].T, item_vectors[i1, :]) - np.dot(dislike_vector[u2, :].T,
                                                                                     item_vectors[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = unknown.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if unknown[u1][i1] > unknown[u2][i2]:
                diff = np.dot(unknown_vector[u1, :].T, item_vectors[i1, :]) - np.dot(unknown_vector[u2, :].T,
                                                                                     item_vectors[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    print(value)
    return value


def cal_splitvalueI(rating_matrix, user_vectors, current_vector, indices_like, indices_dislike, indices_unknown, K):
    like = rating_matrix[:, indices_like]
    dislike = rating_matrix[:, indices_dislike]
    unknown = rating_matrix[:, indices_unknown]
    like_vector = np.zeros(K)
    dislike_vector = np.zeros(K)
    unknown_vector = np.zeros(K)
    value = 0

    if len(indices_like) > 0:
        # print(indices_like)
        like_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_like, K)
        like_vector = np.array([like_vector for i in range(len(indices_like))])
        pre_like = np.dot(user_vectors, like_vector.T)
        Err_like = pre_like[np.nonzero(like)] - like[np.nonzero(like)]
        value = value + np.dot(Err_like, Err_like)
    # print(value)
    if len(indices_dislike) > 0:
        # print(indices_dislike)
        dislike_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_dislike, K)
        dislike_vector = np.array([dislike_vector for i in range(len(indices_dislike))])
        pre_dislike = np.dot(user_vectors, dislike_vector.T)
        Err_dislike = pre_dislike[np.nonzero(dislike)] - dislike[np.nonzero(dislike)]
        value = value + np.dot(Err_dislike, Err_dislike)
    # print(value)

    if len(indices_unknown) > 0:
        # print(indices_unknown)
        unknown_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_unknown, K)
        unknown_vector = np.array([unknown_vector for i in range(len(indices_unknown))])
        pre_unknown = np.dot(user_vectors, unknown_vector.T)
        Err_unknown = pre_unknown[np.nonzero(unknown)] - unknown[np.nonzero(unknown)]
        value = value + np.dot(Err_unknown, Err_unknown)
    # print(value)

    lkv_l = like_vector[np.nonzero(like_vector)]
    dlkv_l = dislike_vector[np.nonzero(dislike_vector)]
    unkv_l = unknown_vector[np.nonzero(unknown_vector)]
    value = value + lmd_v * (np.dot(lkv_l, lkv_l) + np.dot(dlkv_l, dlkv_l) + np.dot(unkv_l, unkv_l))

    user_l = user_vectors[np.nonzero(user_vectors)]
    value = value + lmd_u * np.dot(user_l, user_l)

    np.random.seed(0)
    num_pair = 20
    num_user, num_item = like.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if like[u1][i1] > like[u2][i2]:
                diff = np.dot(user_vectors[u1, :].T, like_vector[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                  like_vector[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = dislike.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if dislike[u1][i1] > dislike[u2][i2]:
                diff = np.dot(user_vectors[u1, :].T, dislike_vector[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                     dislike_vector[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = unknown.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if unknown[u1][i1] > unknown[u2][i2]:
                diff = np.dot(user_vectors[u1, :].T, unknown_vector[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                     unknown_vector[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    print(value)
    return value
