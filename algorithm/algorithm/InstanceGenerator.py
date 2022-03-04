import numpy as np
from typing import List, Tuple

# creates Knapsack instances of different types as defined in https://doi.org/10.1016/j.cor.2004.03.002
def create_instance(type: str, n: int, R: int, H: int =100) -> Tuple[List[int], List[int], List[int]]:
    types = ["uncor", "weak_cor", "strong_cor", "inv_str", "alm_str", "subset"]
    if type not in types:
        print("Eine der folgenden Parameter auswählen:")
        print(types)
        return
    weights = [np.random.randint(1, R+1) for _ in range(n)]
    if type == "uncor":
        profits = [np.random.randint(1, R+1) for _ in range(n)]
    elif type == "weak_cor":
        profits = [max(np.random.randint(int(w - R / 10), int(w + R / 10) + 1), 1) for w in weights]
    elif type == "strong_cor":
        profits = [int(w + R / 10) for w in weights]
    elif type == "inv_str":
        profits = [w for w in weights]
        weights = [int(p + R / 10) for p in profits]
    elif type == "alm_str":
        profits = [max(np.random.randint(int(w + R / 10 - R / 500), int(w + R / 10 + R / 500) + 1), 1) for w in weights]
    else:
        profits = weights
    max_weights = [int(h * 0.25 * sum(weights)) for h in range(2, 5)]
    return profits, weights, max_weights

#can save an instance to a file
def create_and_write(type: str, n: int, R: int, H: int =100) -> None:
    profits, weights, max_weight = create_instance(type, n, R, H)
    with open("type={}n={}_input.txt".format(type, str(n)), "w") as f:
        f.write(str(profits))
        f.write(str(weights))
        f.write(str(max_weight))
    f.close()