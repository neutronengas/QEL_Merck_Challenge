"""#!/usr/bin/env python3"""
# coding: utf-8

from numpy import array
from QAOASolver import GridSearcher, Item
from ClassicalSolver import dynamic_programming, lazy_greedy

def main():
    profits = [458, 231, 759, 915, 658, 1054, 1032, 631, 576, 787]
    weights = [358, 131, 659, 815, 558, 954, 932, 531, 476, 687]
    items = [Item(i, profits[i], weights[i]) for i in range(len(profits))]
    max_weight = 3660
    thetas = [-1.0 for _ in profits]
    N_beta = N_gamma = 10
    p = 1
    solving_type = "Copula"
    init_type = "smoothened"
    searcher = GridSearcher(items, max_weight, thetas, N_beta, N_gamma, p, solving_type, init_type)
    best_string, quantum_solution = searcher.get_max_result()
    greedy_output = lazy_greedy(items, max_weight)
    greedy_solution = (array(greedy_output[0]) * array(profits)).sum()
    dynamic_programming_solution = dynamic_programming(items, max_weight)
    print("Quantum solution bit-string(s): " + str(best_string))
    print("Quantum solution value: " + str(quantum_solution))
    print()
    print("Greedy solution value: " + str(greedy_solution))
    print()
    print("Dynamic programming solution value: " + str(dynamic_programming_solution))
    return

if __name__ == "__main__":
    main()