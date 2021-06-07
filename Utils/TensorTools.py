import torch

from Utils import Constant


def group_divisions(tensor, amount_of_divisions):
    # Tensor input torch.Size([1024, 8]) [amount_of_divisions x batch_size,m]
    unfolded_tenor = tensor.unfold(Constant.BATCH_DIMENSION, amount_of_divisions, amount_of_divisions).transpose(
        Constant.DIVISION_DIMENSION,
        Constant.M_DIMENSION)  # (64, 16, 8)  [batch_size, amount_of_divisions, m]
    grouped_divisions = []
    for div in range(amount_of_divisions):
        # narrow-> starts at div in m dimension, one div at a time
        grouped_division = unfolded_tenor.narrow(Constant.DIVISION_DIMENSION, div, 1).transpose(
            Constant.BATCH_DIMENSION,
            Constant.DIVISION_DIMENSION)
        # Batch of division [1,64,8] [batch_size, 1 division ,m]
        grouped_divisions.append(grouped_division)
    DIVISION_DIMENSION = 0
    concatenated_tensor = torch.cat(grouped_divisions, DIVISION_DIMENSION)  # Concatenate columns into tensor
    return concatenated_tensor


def list_of_tuples_of_indexes_to_split_n_in_k_equal_parts(n, k):
    split_indexes = [i * (n // k) + min(i, n % k) for i in range(k)] + [n]
    list_of_tuples_of_indexes = [(split_indexes[i], split_indexes[i + 1]) for i in range(len(split_indexes) - 1)]
    return list_of_tuples_of_indexes


def parallel_flat_division(tensor, n_parallel_divisions, divide_in_row, divide_in_col):
    dim, row_dims, col_dims = tensor.size()
    list_of_tuples_of_indexes_dim = list_of_tuples_of_indexes_to_split_n_in_k_equal_parts(dim, n_parallel_divisions)
    tensor_list = []
    for tuple_of_indexes_dim in list_of_tuples_of_indexes_dim:
        first_index_dim, second_index_dim = tuple_of_indexes_dim
        row_patch_size = second_index_dim - first_index_dim
        narrowed_tensor = tensor.narrow(Constant.ZERO_DIMENSION, first_index_dim, row_patch_size)
        flattened_narrowed_tensor = flat_divisions(narrowed_tensor, divide_in_row, divide_in_col)
        tensor_list.append(flattened_narrowed_tensor)
    concatenated_tensor = torch.cat(tensor_list, Constant.ZERO_DIMENSION)  # Concatenate columns into tensor
    return concatenated_tensor


def flat_divisions(tensor, divide_in_row, divide_in_col):
    dim, row_dims, col_dims = tensor.size()
    list_of_tuples_of_indexes_row = list_of_tuples_of_indexes_to_split_n_in_k_equal_parts(row_dims, divide_in_row)
    list_of_tuples_of_indexes_col = list_of_tuples_of_indexes_to_split_n_in_k_equal_parts(col_dims, divide_in_col)
    tensor_list = []
    for tuple_of_indexes_row in list_of_tuples_of_indexes_row:
        for tuple_of_indexes_col in list_of_tuples_of_indexes_col:
            first_index_row, second_index_row = tuple_of_indexes_row
            first_index_col, second_index_col = tuple_of_indexes_col
            row_patch_size = second_index_row - first_index_row
            col_patch_size = second_index_col - first_index_col
            tensor_cropped = tensor.narrow(Constant.FIRST_DIMENSION, first_index_row, row_patch_size) \
                .narrow(Constant.SECOND_DIMENSION, first_index_col, col_patch_size) \
                .reshape(Constant.SIZE_ONE_DIMENSION, Constant.SIZE_ONE_DIMENSION, row_patch_size * col_patch_size)
            # Flat division into a column
            tensor_list.append(tensor_cropped)
    concatenated_tensor = torch.cat(tensor_list, Constant.FIRST_DIMENSION)  # Concatenate columns into tensor
    return concatenated_tensor


def flat_divisions_with_batch(tensor, divide_in_row, divide_in_col):
    dim, row_dims, col_dims = tensor.size()
    list_of_tuples_of_indexes_row = list_of_tuples_of_indexes_to_split_n_in_k_equal_parts(row_dims, divide_in_row)
    list_of_tuples_of_indexes_col = list_of_tuples_of_indexes_to_split_n_in_k_equal_parts(col_dims, divide_in_col)
    tensor_list = []
    for d in range(dim):
        dim_tensor_list=[]
        t = tensor.narrow(Constant.ZERO_DIMENSION, d, Constant.SIZE_ONE_DIMENSION)
        for tuple_of_indexes_row in list_of_tuples_of_indexes_row:
            for tuple_of_indexes_col in list_of_tuples_of_indexes_col:
                first_index_row, second_index_row = tuple_of_indexes_row
                first_index_col, second_index_col = tuple_of_indexes_col
                row_patch_size = second_index_row - first_index_row
                col_patch_size = second_index_col - first_index_col
                tensor_cropped = t.narrow(Constant.FIRST_DIMENSION, first_index_row, row_patch_size) \
                    .narrow(Constant.SECOND_DIMENSION, first_index_col, col_patch_size) \
                    .reshape(Constant.SIZE_ONE_DIMENSION, Constant.SIZE_ONE_DIMENSION, row_patch_size * col_patch_size)
                # Flat division into a column
                dim_tensor_list.append(tensor_cropped)
        dim_concatenated_tensor  =torch.cat(dim_tensor_list, Constant.FIRST_DIMENSION)
        tensor_list.append(dim_concatenated_tensor)
    concatenated_tensor = torch.cat(tensor_list, Constant.ZERO_DIMENSION)  # Concatenate columns into tensor
    return concatenated_tensor


def dimension_trace(tensor, dimension):
    dimension_size = tensor.size()[dimension]
    tensor_list = []
    for i in range(dimension_size):
        narrowed_tensor = tensor.narrow(dimension, i, Constant.ONE)
        trace = torch.trace(narrowed_tensor).reshape(Constant.ONE, Constant.ONE)
        tensor_list.append(trace)
    concatenated_tensor = torch.cat(tensor_list, dimension)  # Concatenate columns into tensor
    return concatenated_tensor
