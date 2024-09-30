import unittest

import algoexprt


class TestTwoNums(unittest.TestCase):
    def test_two_nums_sum_mapping(self):
        nums_to_sum = [3, 5, -4, 8, 11, 1, -1, 6]
        target_sum = 10
        self.assertEqual(algoexprt.two_nums_sum_mapping(nums_to_sum, target_sum), [11, -1])

    def test_two_nums_sum_sorted(self):
        nums_to_sum = [3, 5, -4, 8, 11, 1, -1, 6]
        target_sum = 11
        self.assertEqual(algoexprt.two_nums_sum_sorted(nums_to_sum, target_sum), [3, 8])


class TestFindClosestBst(unittest.TestCase):
    def test_find_closest_val_bst(self):
        target_val = 4
        closest_val = 5
        self.assertEqual(algoexprt.find_closest_val_bst(algoexprt.bs_root, target_val), closest_val)


class TestDepthFirstSearch(unittest.TestCase):
    def test_depth_first_search(self):
        depth_first_order = ['A', 'B', 'E', 'F', 'I', 'J', 'C', 'D', 'G', 'K', 'H']
        self.assertEqual(algoexprt.A.depth_first_search([]), depth_first_order)


class TestCalcBranchSumsBST(unittest.TestCase):
    def test_branch_sums(self):
        tree = algoexprt.bs_root
        self.assertEqual(tree.branch_sums(tree), [18, 20, 52, 47])
        self.assertEqual(tree.branch_sums(tree.left), [8, 10])
        self.assertEqual(tree.branch_sums(tree.right), [42, 37])


class TestNthFibonacci(unittest.TestCase):
    def test_nth_fib(self):
        self.assertEqual(algoexprt.get_nth_fib1(20), 4181)
        self.assertEqual(algoexprt.get_nth_fib2(100), 218922995834555169026)
        self.assertEqual(algoexprt.get_nth_fib3(100), 218922995834555169026)


class TestFindThreeLargestNums(unittest.TestCase):
    def test_find_three_largest_nums(self):
        to_find_three_largest = [141, 1, 17, -7, -17, -27, 18, 541, 8, 7, 7]
        assert algoexprt.find_three_largest_nums(to_find_three_largest) == [18, 141, 541]


class TestBinarySearch(unittest.TestCase):
    def test_binary_search(self):
        ordered_list = [-5, 0, 1, 4, 8, 11, 15, 56, 90, 173]
        assert algoexprt.binary_search(ordered_list, 4) == 3
        assert algoexprt.binary_search(ordered_list, 9) is None


class TestInsertionSort(unittest.TestCase):
    def test_insertion_sort(self):
        list_to_sort = [42, 4, 16, 8, 15, 23, 93, 0, 11, 2, 2, 7, 23]
        sorted_list = [0, 2, 2, 4, 7, 8, 11, 15, 16, 23, 23, 42, 93]
        assert algoexprt.insertion_sort(list_to_sort) == sorted_list


if __name__ == '__main__':
    unittest.main()
