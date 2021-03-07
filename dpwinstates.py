
# this is currently for thinking through a way of directly determining the number of valid win states
# which is probably not possible

def win_positions(m: int, n: int, k: int):
    total = 0
    for i in range(m):
        for j in range(n):
            if (i + k - 1) < m:
                total += 1
                if (j + k - 1) < n:
                    total += 1
            if (j + k - 1) < n:
                total += 1
                if (i - k + 1) >= 0:
                    total += 1
    return total

print(win_positions(3, 3, 3))
