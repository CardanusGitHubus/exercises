import unittest

import algoexprt


class TestNumsSums(unittest.TestCase):
    def test_two_nums_sum_mapping(self):
        nums_to_sum = [3, 5, -4, 8, 11, 1, -1, 6]
        target_sum = 10
        self.assertEqual(algoexprt.two_nums_sum_mapping(nums_to_sum, target_sum), [11, -1])

    def test_two_nums_sum_sorted(self):
        nums_to_sum = [3, 5, -4, 8, 11, 1, -1, 6]
        target_sum = 11
        self.assertEqual(algoexprt.two_nums_sum_sorted(nums_to_sum, target_sum), [3, 8])

    def test_three_nums_sum_sorted(self):
        nums_to_sum = [12, 3, 1, 2, -6, 5, -8, 6]
        target_sum = 0
        result = [[-8, 2, 6], [-8, 3, 5], [-6, 1, 5]]
        self.assertEqual(algoexprt.three_nums_sum_sorted(nums_to_sum, target_sum), result)


class TestBst(unittest.TestCase):
    # naive-defined BST
    bs_root = algoexprt.BSTNode(10)
    bs_root.add_child(5)
    bs_root.add_child(15)

    bs_root.left.add_child(2)
    bs_root.left.add_child(5)

    bs_root.left.left.add_child(1)

    bs_root.right.add_child(13)
    bs_root.right.add_child(22)

    bs_root.right.left.add_child(14)

    # full-defined BST
    bst_root = algoexprt.BST(10)
    for node in (5, 12, 4, 1, -4, 8, 17, 22, 35):
        bst_root.insert(node)

    def test_find_closest_val_bst(self):
        target_val = 4
        closest_val = 5
        self.assertEqual(algoexprt.find_closest_val_bst(self.bs_root, target_val), closest_val)

    def test_branch_sums(self):
        self.assertEqual(self.bs_root.branch_sums(self.bs_root), [18, 20, 52, 47])
        self.assertEqual(self.bs_root.branch_sums(self.bs_root.left), [8, 10])
        self.assertEqual(self.bs_root.branch_sums(self.bs_root.right), [42, 37])

    def test_bst_contains(self):
        for node in (5, 12, 4, 1, -4, 8, 17, 22, 35):
            assert self.bst_root.contains(node)
        for node in (2, 3, 6, 7):
            assert not self.bst_root.contains(node)

    def test_validate_bst(self):
        assert algoexprt.validate_bst(self.bst_root)

    def test_traverse_bst(self):
        in_order = [-4, 1, 4, 5, 8, 10, 12, 17, 22, 35]
        pre_order = [10, 5, 4, 1, -4, 8, 12, 17, 22, 35]
        post_order = [-4, 1, 4, 8, 5, 35, 22, 17, 12, 10]
        assert algoexprt.in_order_traverse(self.bst_root, []) == in_order
        assert algoexprt.pre_order_traverse(self.bst_root, []) == pre_order
        assert algoexprt.post_order_traverse(self.bst_root, []) == post_order

    def test_invert_binary_tree(self):
        children = [self.bst_root.left, self.bst_root.right]
        algoexprt.invert_binary_tree(self.bst_root)
        assert children == [self.bst_root.right, self.bst_root.left]
        algoexprt.invert_binary_tree_rec(self.bst_root)
        assert children == [self.bst_root.left, self.bst_root.right]


class TestDepthFirstSearch(unittest.TestCase):
    def test_depth_first_search(self):
        A = algoexprt.Node('A')
        A.add_child('B')
        A.add_child('C')
        A.add_child('D')

        B = A.children[0]
        B.add_child('E')
        B.add_child('F')

        F = B.children[1]
        F.add_child('I')
        F.add_child('J')

        D = A.children[2]
        D.add_child('G')
        D.add_child('H')

        G = D.children[0]
        G.add_child('K')

        depth_first_order = ['A', 'B', 'E', 'F', 'I', 'J', 'C', 'D', 'G', 'K', 'H']
        self.assertEqual(A.depth_first_search([]), depth_first_order)


class TestNthFibonacci(unittest.TestCase):
    def test_nth_fib(self):
        self.assertEqual(algoexprt.get_nth_fib1(20), 4181)
        self.assertEqual(algoexprt.get_nth_fib2(100), 218922995834555169026)
        self.assertEqual(algoexprt.get_nth_fib3(100), 218922995834555169026)


class TestFindThreeLargestNums(unittest.TestCase):
    def test_find_three_largest_nums(self):
        to_find_three_largest = [141, 1, 17, -7, -17, -27, 18, 541, 8, 7, 7]
        assert algoexprt.find_three_largest_nums(to_find_three_largest) == [18, 141, 541]


class TestSearchingAlgorithms(unittest.TestCase):
    def test_binary_search(self):
        ordered_list = [-5, 0, 1, 4, 8, 11, 15, 56, 90, 173]
        assert algoexprt.binary_search(ordered_list, 4) == 3
        assert algoexprt.binary_search(ordered_list, 9) is None

    def test_quickselect(self):
        list_to_select = [42, 4, 16, 8, 15, 23, 93, 0, 11, 2, 2, 7, 23]
        _sorted_list = [0, 2, 2, 4, 7, 8, 11, 15, 16, 23, 23, 42, 93]
        assert algoexprt.quickselect(list_to_select, 5) == _sorted_list[4]


class TestSortingAlgorithms(unittest.TestCase):
    def test_insertion_sort(self):
        list_to_sort = [42, 4, 16, 8, 15, 23, 93, 0, 11, 2, 2, 7, 23]
        sorted_list = [0, 2, 2, 4, 7, 8, 11, 15, 16, 23, 23, 42, 93]
        assert algoexprt.insertion_sort(list_to_sort) == sorted_list

    def test_quick_sort_pythonic(self):
        list_to_sort = [42, 4, 16, 8, 15, 23, 93, 0, 11, 2, 2, 7, 23]
        sorted_list = [0, 2, 2, 4, 7, 8, 11, 15, 16, 23, 23, 42, 93]
        assert algoexprt.quick_sort_pythonic(list_to_sort) == sorted_list

    def test_quick_sort_canonic(self):
        list_to_sort = [42, 4, 16, 8, 15, 23, 93, 0, 11, 2, 2, 7, 23]
        sorted_list = [0, 2, 2, 4, 7, 8, 11, 15, 16, 23, 23, 42, 93]
        assert algoexprt.quick_sort_canonic(list_to_sort) == sorted_list

    def test_merge_sort(self):
        list_to_sort = [42, 4, 16, 8, 15, 23, 93, 0, 11, 2, 2, 7, 23]
        sorted_list = [0, 2, 2, 4, 7, 8, 11, 15, 16, 23, 23, 42, 93]
        assert algoexprt.merge_sort(list_to_sort) == sorted_list

    def test_merge_asc_lists1(self):
        lst_to_merge1 = [2, 4, 7, 23, 42]
        lst_to_merge2 = [0, 2, 2]
        merged_list = [0, 2, 2, 2, 4, 7, 23, 42]
        assert algoexprt.merge_asc_lists1(lst_to_merge1, lst_to_merge2) == merged_list

    def test_merge_asc_lists2(self):
        lst_to_merge1 = [2, 4, 7, 23, 42]
        lst_to_merge2 = [0, 2, 2]
        merged_list = [0, 2, 2, 2, 4, 7, 23, 42]
        assert algoexprt.merge_asc_lists2(lst_to_merge1, lst_to_merge2) == merged_list


class TestCaesarCipherEncryptor(unittest.TestCase):
    def test_caesar_cipher_encryptor(self):
        assert algoexprt.caesar_cipher_encryptor('xyz', 0) == 'xyz'
        assert algoexprt.caesar_cipher_encryptor('xyz', 1) == 'yza'
        assert algoexprt.caesar_cipher_encryptor('xyz', 26) == 'xyz'


class TestSmallestDiff(unittest.TestCase):
    def test_smallest_difference(self):
        arr_1 = [-1, 5, 10, 20, 28, 3]
        arr_2 = [26, 134, 135, 15, 17]
        assert algoexprt.smallest_difference(arr_1, arr_2) == [28, 26]


class TestMoveElemToEnd(unittest.TestCase):
    def test_move_element_to_end(self):
        arr = [-1, 5, 10, 20, 28, 3]
        assert algoexprt.move_element_to_end(arr, 5) == [-1, 3, 10, 20, 28, 5]


class TestSearchInSortedMatrix(unittest.TestCase):
    def test_search_in_sorted_matrix(self):
        matrix = [
            [1, 4, 7, 12, 1000],
            [2, 5, 18, 31, 1001],
            [3, 8, 24, 35, 1002],
            [40, 41, 42, 44, 1004]
        ]
        assert algoexprt.search_in_sorted_matrix(matrix, 44) == [3, 3]


class TestHeapAlgorithms(unittest.TestCase):
    def test_min_heap_const(self):
        arr = [8, 12, 23, 17, 31, 30, 44, 102, 18]
        heap = algoexprt.MinHeap([])
        for i in arr:
            heap.insert(i)
        assert heap.heap == arr

    def test_heap_sort(self):
        arr = [23, 44, 102, 30, 18, 17, 12, 8, 31]
        sorted_heap = [8, 12, 17, 18, 23, 30, 31, 44, 102]
        assert algoexprt.heap_sort(arr) == sorted_heap


class TestKnuthMorrisPratt(unittest.TestCase):
    def test_KMP(self):
        text = 'ABCABAABCABAC'
        pattern = 'CAB'
        pattern_occurs = [2, 8]
        assert algoexprt.KMP(text, pattern) == pattern_occurs


class TestAirportConnections(unittest.TestCase):
    def test_airport_connections(self):
        airports = ['EWH', 'HND', 'ICN', 'JFK', 'LGA', 'BGI', 'ORD', 'DSM', 'SFO',
                    'LHR', 'EYW', 'SAN', 'TLV', 'DEL', 'DOH', 'SIN', 'COG', 'BUD']
        routes = [('EWH', 'HND'), ('HND', 'ICN'), ('HND', 'JFK'), ('ICN', 'JFK'),
                  ('JFK', 'LGA'), ('BGI', 'LGA'), ('ORD', 'BGI'), ('DSM', 'ORD'),
                  ('SFO', 'DSM'), ('LHR', 'SFO'), ('EYW', 'LHR'), ('SAN', 'EYW'),
                  ('SFO', 'SAN'), ('TLV', 'DEL'), ('DEL', 'DOH'), ('DEL', 'COG'),
                  ('DEL', 'SIN'), ('COG', 'BUD'), ('COG', 'SIN')]
        assert algoexprt.airport_connections(airports, routes, 'LGA') == 3


if __name__ == '__main__':
    unittest.main()
