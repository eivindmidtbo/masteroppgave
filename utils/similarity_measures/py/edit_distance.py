""" Sheet containing the trajectory similarity method edit distance that will be applied to the hashes """

import numpy as np

def edit_distance(hash_x: np.ndarray, hash_y: np.ndarray) -> float:
    """
    Computes the edit distance between two trajectory hashes (Grid | Disk hash)\n
    Runs in layers x O(n^2) time, where n is the length of one hash

    Param
    ---
    hash_x : np array list(list(str))
        The full hash of trajectory x
    hash_y : np array list(list(str))
        The full hash of trajectory y
    
    Returns
    ---
    Their combined edit distance (sum of number of edits divided by longest sequence) and total number of edits (float, float)

    Notes
    ---
    Rewritten from https://github.com/bguillouet/traj-dist/blob/master/traj_dist/pydist/edr.py
    """

    x_len = len(hash_x)
    y_len = len(hash_y)

    if x_len != y_len:
        raise ValueError("Number of layers are different for the hashes. Unable to compute edit distance")

    cost = 0
    c = 0

    for layer in range(x_len):
        X = hash_x[layer]
        Y = hash_y[layer]
        X_len = len(X)
        Y_len = len(Y)
        M = np.zeros((X_len + 1, Y_len + 1))
        
        
        # Edge case if one of hashes is empty
        if (X_len == 0 or Y_len == 0) and X_len != Y_len:
            cost += 1
            c += max(X_len, Y_len)
            continue
        # Edge case if both hashes are empty
        if (X_len == 0 and Y_len == 0):
            cost += 0
            continue
        
        for i in range(1, X_len + 1):
            for j in range(1, Y_len + 1):

                if i == 1:
                    M[i-1][j-1] = j-1
                elif j == 1:
                    M[i-1][j-1] = i-1

                if X[i-1] == Y[j-1]: subcost = 0
                else: subcost = 1

                M[i,j] = min(M[i][j-1] + 1, M[i-1][j] + 1, M[i-1][j-1] + subcost)
        #print(M)
        cost += float(M[X_len][Y_len]) / max([X_len, Y_len])
        c += float(M[X_len][Y_len])
    
    return cost, c



if __name__=="__main__":

    # Simple testing

    assert edit_distance([["AAaa","ABab","ACac","ADad"], ["AAaa","ABaa","ACac"]], [["AAaa", "ABab", "ADad"], ["AAaa", "ABaa", "ACab"]]) == (float(1/4 +1/3), 2)
    assert edit_distance([["s","u","n","d","a","y"]], [["s","a","t","u,","r","d","a","y"]]) == (0.375, 3)
    assert edit_distance([["A","b"], ["a","b","c","d"]], [[], ["a", "c", "d"]]) == (1.25, 3.0)
    assert edit_distance([["a"]], [[]]) == (1,1)
    assert edit_distance([[]], [[]]) == (0,0)
    assert edit_distance([[], ["a", "b"]], [[], ["b", "a"]]) == (1,2)
    assert edit_distance([["a", "b"]], [["b", "a"]]) == (1,2)
    assert edit_distance([["a", "a"]], [["b", "a"]]) == (0.5,1.0)
    assert edit_distance([["b", "a"]], [["b", "a"]]) == (0,0)
    assert edit_distance([["b", "a"]], [["a", "b"]]) == (1,2)
    assert edit_distance([["b", "a","b"]], [["a", "b","a"]]) == (float(2/3), 2)
    assert edit_distance([["b", "a","b","a"]], [["a", "b","a","b"]]) == (0.5, 2)
    print("All tests passed")