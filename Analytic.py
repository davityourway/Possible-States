from math import factorial


def nPr(n, r):
    return int(factorial(n)/factorial(n-r))

n = 3
m = 3



positions = [["0" for _ in range(n)] for _ in range(m)]
