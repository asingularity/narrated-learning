
from utils.w_save_load_helper import load_W_prob
import numpy as np

def experiment():
    nz, W = load_W_prob('tmp.txt')
    from_index = 345
    to_indices = nz[from_index]

    s = np.sum(W[to_indices, from_index])
    print(s)

    N = W.shape[0]
    assert N == 8000 and N == W.shape[1]

    # for each row "from index", have all "to_indices"
    # to find equivalent groups:
    #   make a new group
    #   mark all "to_indices" rows as same group

    group_label_per_row = {}
    last_used_label = -1

    equiv_label_groups = []

    for from_index in range(N):
        to_indices = nz[from_index]

        # only top 1
        weights = list(W[to_indices, from_index])
        print(weights)

        # these are all one group
        all_indices = to_indices + [from_index]

        # (1) find if any are already labeled as a group.
        #   --- what if there are multiple labels already used for different ones?
        #   --- store this label pair as "equivalent" label pair

        unique_labels_already_used = []
        label_to_use = None

        for tmp_index in all_indices:
            if tmp_index in group_label_per_row:
                label = group_label_per_row[tmp_index]
                if label not in unique_labels_already_used:
                    unique_labels_already_used.append(label)

        if len(unique_labels_already_used) > 0:
            label_to_use = unique_labels_already_used[0]

            if len(unique_labels_already_used) > 1:
                equiv_label_groups.append(unique_labels_already_used)

        # (2) if yes, apply that label to all others in this group
        # (3) if not, make a new label and label them all as that group.

        if label_to_use is None:
            label_to_use = last_used_label + 1
            last_used_label = label_to_use

        for tmp_index in all_indices:
            if tmp_index not in group_label_per_row:
                group_label_per_row[tmp_index] = label_to_use

    return

    print()
    print('group_label_per_row')
    print()

    for tmp_index in range(N):
        print('    ', tmp_index, ':', group_label_per_row[tmp_index])
    print()
    print()
    print('equiv_label_groups')
    print()
    print(len(equiv_label_groups))

if __name__ == '__main__':
    experiment()