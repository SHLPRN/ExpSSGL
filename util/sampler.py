from random import shuffle, choice


def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def exp_next_batch_pairwise(data, batch_size, dropped_adj, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]    # user list, not id list
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]    # item list, not id list
        ptr = batch_end
        u_idx, i_idx, j_idx, drop_user_idx, drop_pos_idx, drop_neg_idx = [], [], [], [], [], []
        item_list = list(data.item.keys())      # item list, not id list
        for i, user in enumerate(users):
            user_id = data.user[user]
            item_id = data.item[items[i]]
            i_idx.append(item_id)   # item id list
            u_idx.append(user_id)   # user id list
            is_dropped = (dropped_adj[user_id][item_id] == 0)
            if is_dropped:
                drop_user_idx.append(user_id)
                drop_pos_idx.append(item_id)
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                j_idx.append(data.item[neg_item])
                if is_dropped:
                    drop_neg_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx, drop_user_idx, drop_pos_idx, drop_neg_idx
