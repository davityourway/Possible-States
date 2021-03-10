import unittest
import montecarlostates

if __name__ == '__main__':

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '1', '1', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    winset = montecarlostates.find_legal_junctions(winlist, 3)
    print(terminal, winlist, winset)
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(legal, winset)

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '1', '0'],
         ['0', '0', '0', '1', '1', '1', '0'],
         ['0', '0', '0', '0', '1', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    winset = montecarlostates.find_legal_junctions(winlist, 3)
    print(terminal, winlist, winset)
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(legal, winset)

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '1', '1', '1'],
         ['0', '0', '1', '0', '0', '0', '0'],
         ['0', '1', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    winset = montecarlostates.find_legal_junctions(winlist, 3)
    print(terminal, winlist, winset)
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(legal, winset)

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '1', '1', '0'],
         ['0', '0', '1', '0', '0', '0', '0'],
         ['0', '1', '0', '0', '0', '0', '0'],
         ['1', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    winset = montecarlostates.find_legal_junctions(winlist, 3)
    print(terminal, winlist, winset)
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(legal, winset)

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['2', '0', '0', '0', '0', '0', '0'],
         ['2', '0', '0', '1', '0', '0', '0'],
         ['2', '0', '0', '1', '0', '0', '0'],
         ['0', '0', '0', '1', '1', '1', '0'],
         ['0', '0', '1', '2', '0', '0', '0'],
         ['0', '1', '0', '2', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    winset = montecarlostates.find_legal_junctions(winlist, 3)
    print(terminal, winlist, winset)
    terminal, winlistb = a.check_terminal(testmat, (5, 3), "Second")
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(legal, winset)
    print(terminal, winlistb)

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['0', '0', '0', '0', '0', '0', '0'],
         ['0', '1', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['1', '1', '1', '1', '1', '1', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(terminal, winlist, winset)
    legal, winset = a.check_legal(testmat, (3, 0), winlist, winset, "First")
    print(legal, winset)
    print("\n")

    a = montecarlostates.MonteCarlo(7, 7, 3)
    testmat = \
        [['0', '0', '0', '0', '1', '1', '1'],
         ['0', '1', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '1', '1', '1', '1', '1', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0'],
         ['0', '0', '0', '0', '0', '0', '0']]
    terminal, winlist = a.check_terminal(testmat, (3, 3), "First")
    winset = montecarlostates.find_legal_junctions(winlist, 3)
    legal, winset = a.check_legal(testmat, (3, 3), winlist, winset, "First")
    print(terminal, winlist, winset)
    legal, winset = a.check_legal(testmat, (0, 5), winlist, winset, "First")
    print(legal, winset)

