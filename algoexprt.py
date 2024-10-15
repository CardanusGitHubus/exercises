## ALGOEXPERT
### EASY
#### 1 two number sum. (sorted): time O(Nlog(N)) space O(1)
nums_to_sum = [3, 5, -4, 8, 11, 1, -1, 6]
lst_to_sort = [4, 8, 15, 16, 23, 42, 0, 11, 2, 2, 7, 93, 23]

def two_nums_sum_mapping(arr: list, target):
    mapping = set()
    for i in arr:
        if target - i in mapping:
            return [target - i, i]
        mapping.add(i)
    return []


def two_nums_sum_sorted(arr: list, target):
    arr.sort()
    left = 0
    right = len(arr) - 1
    while left < right:
        if arr[left] + arr[right] < target:
            left += 1
        elif arr[left] + arr[right] > target:
            right -= 1
        else:
            return [arr[left], arr[right]]
    return []


#### 2 closest value in binary search tree time O(log(N)) space = time
class BSTNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def add_child(self, value):
        if self.value > value:
            self.left = BSTNode(value)
        elif self.value <= value:
            self.right = BSTNode(value)
        else:
            raise ValueError('invalid value')

    @classmethod
    def branch_sums(cls, root):
        sums = []
        calc_branch_sums(root, 0, sums)
        return sums


def find_closest_val_bst(tree: BSTNode, target):
    current_node = tree
    closest = float('inf')
    while current_node is not None:
        if abs(closest - target) > abs(current_node.value - target):
            closest = current_node.value
        if target < current_node.value:
            current_node = current_node.left
        elif target > current_node.value:
            current_node = current_node.right
        else:
            break
    return closest


#### 3 depth-first search
class Node:
    def __init__(self, name):
        self.children = []
        self.name = name

    def add_child(self, name):
        self.children.append(Node(name))

    def depth_first_search(self, array: list):
        array.append(self.name)
        for child in self.children:
            child.depth_first_search(array)
        return array


#### 4 brahcn sums time O(N) space O(N)
def calc_branch_sums(node: BSTNode, curr_sum, sums):
    if node is None:
        return

    new_curr_sum = curr_sum + node.value
    if node.left is None and node.right is None:
        sums.append(new_curr_sum)
        return

    calc_branch_sums(node.left, new_curr_sum, sums)
    calc_branch_sums(node.right, new_curr_sum, sums)


#### 5 linked list construction
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    # O(1) time | O(1) space
    def set_head(self, node):
        if self.head is None:
            self.head = node
            self.tail = node
            return
        self.insert_before(self.head, node)

    # O(1) time | O(1) space
    def set_tail(self, node):
        if self.tail is None:
            self.set_head(node)
            return
        self.insert_after(self.tail, node)

    # O(1) time | O(1) space
    def insert_before(self, node, node_to_insert):
        if node_to_insert == self.head and node_to_insert == self.tail:
            return
        self.remove(node_to_insert)
        node_to_insert.prev = node.prev
        node_to_insert.next = node
        if node.prev is None:
            self.head = node_to_insert
        else:
            node.prev.next = node_to_insert
        node.prev = node_to_insert

    def insert_after(self, node, node_to_insert):
        if node_to_insert == self.head and node_to_insert == self.tail:
            return
        self.remove(node_to_insert)
        node_to_insert.prev = node
        node_to_insert.next = node.next
        if node.next is None:
            self.tail = node_to_insert
        else:
            node.next.prev = node_to_insert
        node.next = node_to_insert

    # O(p) time | O(1) space
    def insert_at_position(self, position, node_to_insert):
        if position == 1:
            self.set_head(node_to_insert)
            return
        node = self.head
        curr_position = 1
        while node is not None and curr_position != position:
            node = node.next
            curr_position += 1
        if node is not None:
            self.insert_before(node, node_to_insert)
        else:
            self.set_tail(node_to_insert)

    # O(n) time | O(1) space
    def remove_nodes_by_value(self, value):
        node = self.head
        while node is not None:
            node_to_remove = node
            node = node.next
            if node_to_remove.value == value:
                self.remove(node_to_remove)

    # O(1) time | O(1) space
    def remove(self, node):
        if node == self.head:
            self.head = self.head.next
        if node == self.tail:
            self.tail = self.tail.next
        self.remove_node_bindings(node)

    # O(n) time | O(1) space
    def contains_node_with_value(self, value):
        node = self.head
        while node is not None and node.value != value:
            node = node.next
        return node is not None

    # O(1) time | O(1) space
    def remove_node_bindings(self, node):
        if node.prev is not None:
            node.prev.next = node.next
        node.prev = None
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = None
        node.next = None

# TODO: test cases


#### 6 n-th fibonacci number
def get_nth_fib1(n):  # O(n^2)time | O(n) space
    if n == 2:
        return 1
    elif n == 1:
        return 0
    else:
        return get_nth_fib1(n - 1) + get_nth_fib1(n - 2)


def get_nth_fib2(n, memoize = {1: 0, 2: 1}):  # O(n)time | O(n) space
    if n in memoize:
        return memoize[n]
    else:
        memoize[n] = get_nth_fib2(n - 1, memoize) + get_nth_fib2(n - 2, memoize)
        return memoize[n]


def get_nth_fib3(n):  # O(n)time | O(1) space
    last_two = [0, 1]
    counter = 3
    while counter <= n:
        next_fib = last_two[0] + last_two[1]
        last_two[0] = last_two[1]
        last_two[1] = next_fib
        counter += 1
    return last_two[1] if n > 1 else last_two[0]


#### 9 find three largest numbers (time O(n) | space O(1))
def find_three_largest_nums(arr):
    three_largest = [None, None, None]
    for num in arr:
        update_largest(three_largest, num)
    return three_largest


def update_largest(three_largest, num):
    if not three_largest[2] or num > three_largest[2]:
        shift_and_update(three_largest, num, 2)
    elif not three_largest[1] or num > three_largest[1]:
        shift_and_update(three_largest, num, 1)
    elif not three_largest[0] or num > three_largest[0]:
        shift_and_update(three_largest, num, 0)


def shift_and_update(arr, num, idx):
    for i in range(idx + 1):
        if i == idx:
            arr[i] = num
        else:
            arr[i] = arr[i + 1]


#### 8 binary search (time O(n) | space O(1))
def binary_search(arr, item):
    low = 0
    high = len(arr)-1
    while low <= high:
        mid = int(low + high)
        guess = arr[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return None


#### 11 insertion sort (time O(n^2) | space O(1))
def insertion_sort(arr):
    for i in range(1, len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j - 1]:
            swap(j, j-1, arr)
            j -= 1
    return arr


def swap(i, j, arr):
    arr[i], arr[j] = arr[j], arr[i]


#### * quick sort python (avg time O(nlog(n)) | space O(n))
def quick_sort_pythonic(lst):
    if len(lst) <= 1:
        return lst
    elem = lst[0]
    middle = [i for i in lst if i == elem]
    left = list(filter(lambda x: x < elem, lst))
    right = list(filter(lambda x: x > elem, lst))

    return quick_sort_pythonic(left) + middle + quick_sort_pythonic(right)


#### HARD 26 quick sort canonical
# Best: O(nlog(n)) time | O(log(n)) space
# Average: O(nlog(n)) time | O(log(n)) space
# Worst: O(n^2) time | O(log(n)) space
def quick_sort_canonic(array):
    quick_sort_helper(array, 0, len(array) - 1)
    return array


def quick_sort_helper(array, start_idx, end_idx):
    if start_idx >= end_idx:
        return
    pivot_idx = start_idx
    left_idx = start_idx + 1
    right_idx = end_idx
    while right_idx >= left_idx:
        if array[left_idx] > array[pivot_idx] > array[right_idx]:
            swap(left_idx, right_idx, array)
        if array[left_idx] <= array[pivot_idx]:
            left_idx += 1
        if array[right_idx] >= array[pivot_idx]:
            right_idx -= 1
    swap(pivot_idx, right_idx, array)
    left_subarray_is_smaller = right_idx - 1 - start_idx < end_idx - (right_idx + 1)
    if left_subarray_is_smaller:
        quick_sort_helper(array, start_idx, right_idx - 1)
        quick_sort_helper(array, right_idx + 1, end_idx)
    else:
        quick_sort_helper(array, right_idx + 1, end_idx)
        quick_sort_helper(array, start_idx, right_idx - 1)


#### VERY HARD merge_sort (avg time O(nlog(n) | space O(n))
def merge_asc_lists1(a, b):
    i = j = 0
    res = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    if i < len(a):
        res = res + a[i:]
    if j < len(b):
        res = res + b[j:]
    return res


def merge_asc_lists2(a, b):
    i = j = 0
    res = []
    while i < len(a) or j < len(b):
        if i == len(a):
            res.append(b[j])
            j += 1
        elif j == len(b):
            res.append(a[i])
            i += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    return res


def merge_sort(lst):
    if len(lst) <= 1:
        return lst
    middle = len(lst) // 2
    left = merge_sort(lst[:middle])
    right = merge_sort(lst[middle:])
    return merge_asc_lists1(left, right)


#### 14 caeser cipher encryptor (time O(n) | space O(n))
def caesar_cipher_encryptor(string, key):
    new_letters = []
    new_key = key % 26
    # alphabet = list('abcdefghijklmnopqrstuvwxyz') - case w/o unicode codes
    for letter in string:
        new_letters.append(get_new_letter(letter, new_key))
    return ''.join(new_letters)


def get_new_letter(letter, key):  # , alphabet
    new_letter_code = ord(letter) + key
    # new_letter_code = alphabet.index(letter) + key
    # return alphabet[new_letter_code] if new_letter_code <= 25 else alphabet[-1 + new_letter_code % 25]
    return chr(new_letter_code) if new_letter_code <= 122 else chr(96 + new_letter_code % 122)


## ALGOEXPERT
### MEDIUM
#### 1 three number sum. (sorted): time O(n^2) | space O(n)
def three_nums_sum_sorted(arr: list, target):
    arr.sort()
    triplets = []
    for i in range(len(arr) - 2):
        left = i + 1
        right = len(arr) - 1
        while left < right:
            curr_sum = arr[i] + arr[left] + arr[right]
            if curr_sum == target:
                triplets.append([arr[i], arr[left], arr[right]])
                left += 1
                right -= 1
            elif curr_sum < target:
                left += 1
            elif curr_sum > target:
                right -= 1
    return triplets


#### 2 smallest difference O(nlog(n) + mlog(m)) time | O(1) space
def smallest_difference(array_one, array_two):
    array_one.sort()
    array_two.sort()
    idx_one = 0
    idx_two = 0
    smallest = float("inf")
    current = float("inf")
    smallest_pair = []
    while idx_one < len(array_one) and idx_two < len(array_two):
        first_num = array_one[idx_one]
        second_num = array_two[idx_two]
        if first_num < second_num:
            current = second_num - first_num
            idx_one += 1
        elif second_num < first_num:
            current = first_num - second_num
            idx_two += 1
        else:
            return [first_num, second_num]
        if smallest > current:
            smallest = current
            smallest_pair = [first_num, second_num]
    return smallest_pair


#### 3 move elem to the end: O(n) time | O(1) space - where n is the length of the array
def move_element_to_end(array, to_move):
    i = 0
    j = len(array) - 1
    while i < j:
        while i < j and array[j] == to_move:
            j -= 1
        if array[i] == to_move:
            array[i], array[j] = array[j], array[i]
        i += 1
    return array


#### 4 BSTree construction
class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    # avg: O(log(n)) time | O(1) space; worst: O(n) time | O(1) space
    def insert(self, value):
        curr_node = self
        while True:
            if value < curr_node.value:
                if curr_node.left is None:
                    curr_node.left = BST(value)
                    break
                else:
                    curr_node = curr_node.left
            else:
                if curr_node.right is None:
                    curr_node.right = BST(value)
                    break
                else:
                    curr_node = curr_node.right
        return self

    # avg: O(log(n)) time | O(1) space; worst: O(n) time | O(1) space
    def contains(self, value):
        curr_node = self
        while curr_node is not None:
            if value < curr_node.value:
                curr_node = curr_node.left
            elif value > curr_node.value:
                curr_node = curr_node.right
            else:
                return True
        return False

    # avg: O(log(n)) time | O(1) space; worst: O(n) time | O(1) space
    def remove(self, value, parent_node=None):
        current_node = self
        while current_node is not None:
            if value < current_node.value:
                parent_node = current_node
                current_node = current_node.left
            elif value > current_node.value:
                parent_node = current_node
                current_node = current_node.right
            else:
                if current_node.left is not None and current_node.right is not None:
                    current_node.value = current_node.right.get_min_value()
                    current_node.right.remove(current_node.value, current_node)
                elif parent_node is None:
                    if current_node.left is not None:
                        current_node.value = current_node.left.value
                        current_node.right = current_node.left.right
                        current_node.left = current_node.left.left
                    elif current_node.right is not None:
                        current_node.value = current_node.right.value
                        current_node.left = current_node.right.left
                        current_node.right = current_node.right.right
                    else:
                        # This is a single-node tree; do nothing.
                        pass
                elif parent_node.left == current_node:
                    parent_node.left = current_node.left if current_node.left is not None else current_node.right
                elif parent_node.right == current_node:
                    parent_node.right = current_node.left if current_node.left is not None else current_node.right
                break
        return self

    def get_min_value(self):
        current_node = self
        while current_node.left is not None:
            current_node = current_node.left
        return current_node.value

    def __repr__(self):
        return str(self.value)


#### 5 validate BST
def validate_bst(tree: BST):
    return validate_bst_helper(tree, float("-inf"), float("inf"))


def validate_bst_helper(tree, min_value, max_value):
    if tree is None:
        return True
    if tree.value < min_value or tree.value >= max_value:
        return False
    left_is_valid = validate_bst_helper(tree.left, min_value, tree.value)
    return left_is_valid and validate_bst_helper(tree.right, tree.value, max_value)


#### 6 traverse BST
# O(n) time | O(n) space
def in_order_traverse(tree: BST, array):
    if tree is not None:
        in_order_traverse(tree.left, array)
        array.append(tree.value)
        in_order_traverse(tree.right, array)
    return array


# O(n) time | O(n) space
def pre_order_traverse(tree: BST, array):
    if tree is not None:
        array.append(tree.value)
        pre_order_traverse(tree.left, array)
        pre_order_traverse(tree.right, array)
    return array


# O(n) time | O(n) space
def post_order_traverse(tree: BST, array):
    if tree is not None:
        post_order_traverse(tree.left, array)
        post_order_traverse(tree.right, array)
        array.append(tree.value)
    return array


#### 7 invert BST ('mirroring')
def swap_left_right(tree: BST):
    tree.left, tree.right = tree.right, tree.left


# O(n) time | O(n) space
def invert_binary_tree(tree: BST):
    queue = [tree]
    while len(queue):
        current = queue.pop(0)
        if current is None:
            continue
        swap_left_right(current)
        queue.append(current.left)
        queue.append(current.right)


# O(n) time | O(depth) space
def invert_binary_tree_rec(tree: BST):
    if tree is None:
        return
    swap_left_right(tree)
    invert_binary_tree_rec(tree.left)
    invert_binary_tree_rec(tree.right)


#### 17 min heap construction (node value <= children)
class MinHeap:
    def __init__(self, array):
        self.heap = self.build_heap(array)

    # O(n) time | O(1) space
    def build_heap(self, array):
        first_parent_idx = (len(array) - 2) // 2
        for current_idx in reversed(range(first_parent_idx + 1)):
            self.sift_down(current_idx, len(array) - 1, array)
        return array

    # O(log(n)) time | O(1) space
    def sift_down(self, current_idx, end_idx, heap):
        child_one_idx = current_idx * 2 + 1
        while child_one_idx <= end_idx:
            child_two_idx = current_idx * 2 + 2 if current_idx * 2 + 2 <= end_idx else -1
            if child_two_idx != -1 and heap[child_two_idx] < heap[child_one_idx]:
                idx_to_swap = child_two_idx
            else:
                idx_to_swap = child_one_idx
            if heap[idx_to_swap] < heap[current_idx]:
                self.swap(current_idx, idx_to_swap, heap)
                current_idx = idx_to_swap
                child_one_idx = current_idx * 2 + 1
            else:
                return

    # O(log(n)) time | O(1) space
    def sift_up(self, current_idx, heap):
        parent_idx = (current_idx - 1) // 2
        while current_idx > 0 and heap[current_idx] < heap[parent_idx]:
            self.swap(current_idx, parent_idx, heap)
            current_idx = parent_idx
            parent_idx = (current_idx - 1) // 2

    # O(1) time | O(1) space
    def peek(self):
        return self.heap[0]

    # O(log(n)) time | O(1) space
    def remove(self):
        self.swap(0, len(self.heap) - 1, self.heap)
        value_to_remove = self.heap.pop()
        self.sift_down(0, len(self.heap) - 1, self.heap)
        return value_to_remove

    # O(log(n)) time | O(1) space
    def insert(self, value):
        self.heap.append(value)
        self.sift_up(len(self.heap) - 1, self.heap)

    def swap(self, i, j, heap):
        heap[i], heap[j] = heap[j], heap[i]


#### 21 search in sorted matrix O(n+m) time | O(1) space
def search_in_sorted_matrix(matrix, target):
    row = 0
    col = len(matrix[0]) - 1
    while row < len(matrix[0]) and col >= 0:
        if matrix[row][col] > target:
            col -= 1
        elif matrix[row][col] < target:
            row += 1
        else:
            return [row, col]
    return [-1, -1]


#### 26 suffix tree construction
class SuffixTree:
    def __init__(self, string):
        self.root = {}
        self.end_symbol = "*"
        self.populate_suffix_tree_from(string)

    # O(n^2) time | O(n^2) space
    def populate_suffix_tree_from(self, string):
        for i in range(len(string)):
            self.insert_substring_starting_at(i, string)

    def insert_substring_starting_at(self, i, string):
        node = self.root
        for j in range(i, len(string)):
            letter = string[j]
            if letter not in node:
                node[letter] = {}
            node = node[letter]
        node[self.end_symbol] = True

    # O(m) time | O(1) space
    def contains(self, string):
        node = self.root
        for letter in string:
            if letter not in node:
                return False
            node = node[letter]
        return self.end_symbol in node


## ALGOEXPERT
### HARD
#### 19 reverse linked list
# O(n) time | O(1) space - where n is the number of nodes in the Linked List
def reverse_linked_list(head):
    p1, p2 = None, head
    while p2 is not None:
        p3 = p2.next
        p2.next = p1
        p1 = p2
        p2 = p3
    return p1


#### 25 quick select (to find a k-th smallest value) O(n) time | O(1) space
def quickselect(arr, k):
    position = k - 1
    return quickselect_helper(arr, 0, len(arr) - 1, position)


def quickselect_helper(arr, start_idx, end_idx, pos):

    while True:
        if start_idx > end_idx:
            raise Exception('Algorithm should never arrive here!')
        pivot_idx = start_idx
        left_idx = start_idx + 1
        right_idx = end_idx
        while left_idx <= right_idx:
            if arr[left_idx] > arr[pivot_idx] > arr[right_idx]:
                swap(left_idx, right_idx, arr)
            if arr[left_idx] <= arr[pivot_idx]:
                left_idx += 1
            if arr[right_idx] >= arr[pivot_idx]:
                right_idx -= 1
        swap(pivot_idx, right_idx, arr)
        if right_idx == pos:
            return arr[right_idx]
        elif right_idx < pos:
            start_idx = right_idx + 1
        else:
            end_idx = right_idx - 1


#### 27 heap sort (to find a k-th smallest value) O(n) time | O(1) space
# Best: O(nlog(n)) time | O(1) space
# Average: O(nlog(n)) time | O(1) space
# Worst: O(nlog(n)) time | O(1) space
def heap_sort(array):
    build_max_heap(array)
    for end_idx in reversed(range(1, len(array))):
        swap(0, end_idx, array)
        sift_down(0, end_idx - 1, array)
    return array


def build_max_heap(array):
    first_parent_idx = (len(array) - 2) // 2
    for current_idx in reversed(range(first_parent_idx + 1)):
        sift_down(current_idx, len(array) - 1, array)


def sift_down(current_idx, end_idx, heap):
    child_one_idx = current_idx * 2 + 1
    while child_one_idx <= end_idx:
        child_two_idx = current_idx * 2 + 2 if current_idx * 2 + 2 <= end_idx else -1
        if child_two_idx > -1 and heap[child_two_idx] > heap[child_one_idx]:
            idx_to_swap = child_two_idx
        else:
            idx_to_swap = child_one_idx
        if heap[idx_to_swap] > heap[current_idx]:
            swap(current_idx, idx_to_swap, heap)
            current_idx = idx_to_swap
            child_one_idx = current_idx * 2 + 1
        else:
            return


## ALGOEXPERT
### VERY HARD
#### 7 Knuth–Morris–Pratt algorithm
def KMP(text, pattern):
    # base case 1: pattern is empty
    if not pattern:
        print('The pattern occurs with shift 0')
        return

    # base case 2: text is empty, or text's length is less than that of pattern's
    if not text or len(pattern) > len(text):
        print('Pattern not found')
        return

    chars = list(pattern)
    pattern_occurs = []

    # next[i] stores the index of the next best partial match
    next = [0] * (len(pattern) + 1)

    for i in range(1, len(pattern)):
        j = next[i]

        while j > 0 and chars[j] is not chars[i]:
            j = next[j]

        if j > 0 or chars[j] == chars[i]:
            next[i + 1] = j + 1

    i, j = 0, 0
    while i < len(text):
        if j < len(pattern) and text[i] == pattern[j]:
            j = j + 1
            if j == len(pattern):
                print('Pattern occurs with shift', (i - j + 1))
                pattern_occurs.append(i - j + 1)
        elif j > 0:
            j = next[j]
            i = i - 1  # since `i` will be incremented in the next iteration
        i = i + 1
    return pattern_occurs


## ALGOEXPERT
### EXTREMELY HARD
#### 2 Airport connections
# O(a * (a + r) + a + r + alog(a)) time | O(a + r) space - where a is the number of airports and r is number of routes
def airport_connections(airports, routes, starting_airport):
    airport_graph = create_airport_graph(airports, routes)
    unreachable_airport_nodes = get_unreachable_airport_nodes(airport_graph, airports, starting_airport)
    mark_unreachable_connections(airport_graph, unreachable_airport_nodes)
    return get_min_number_of_new_connections(airport_graph, unreachable_airport_nodes)


# O(a + r) time | O(a + r) space
def create_airport_graph(airports, routes):
    airport_graph = {}
    for airport in airports:
        airport_graph[airport] = AirportNode(airport)
    for route in routes:
        # ['LGA', 'JFK']
        airport, connection = route
        airport_graph[airport].connections.append(connection)
    return airport_graph


# O(a + r) time | O(a) space
def get_unreachable_airport_nodes(airport_graph, airports, starting_airport):
    visited_airports = {}
    depth_first_traverse_airports(airport_graph, starting_airport, visited_airports)

    unreachable_airport_nodes = []
    for airport in airports:
        if airport in visited_airports:
            continue
        airport_node = airport_graph[airport]
        airport_node.is_reachable = False
        unreachable_airport_nodes.append(airport_node)
    return unreachable_airport_nodes


def depth_first_traverse_airports(airport_graph, airport, visited_airports):
    if airport in visited_airports:
        return
    visited_airports[airport] = True
    connections = airport_graph[airport].connections
    for connection in connections:
        depth_first_traverse_airports(airport_graph, connection, visited_airports)


# O(a * (a + r)) time | O(a) space
def mark_unreachable_connections(airport_graph, unreachable_airport_nodes):
    for airport_node in unreachable_airport_nodes:
        airport = airport_node.airport
        unreachable_connections = []
        depth_first_add_unreachable_connections(airport_graph, airport, unreachable_connections, {})
        airport_node.unreachable_connections = unreachable_connections


def depth_first_add_unreachable_connections(airport_graph, airport, unreachable_connections, visited_airports):
    if airport_graph[airport].is_reachable:
        return
    if airport in visited_airports:
        return
    visited_airports[airport] = True
    unreachable_connections.append(airport)
    connections = airport_graph[airport].connections
    for connection in connections:
        depth_first_add_unreachable_connections(airport_graph, connection, unreachable_connections, visited_airports)


# O(alog(a) + a + r) time | O(1) space
def get_min_number_of_new_connections(airport_graph, unreachable_airport_nodes):
    unreachable_airport_nodes.sort(key=lambda airport: len(airport.unreachable_connections), reverse=True)

    airports_to_reach = []
    number_of_new_connections = 0
    for airport_node in unreachable_airport_nodes:
        if airport_node.is_reachable:
            continue
        number_of_new_connections += 1
        airports_to_reach.append(airport_node.airport)
        for connection in airport_node.unreachable_connections:
            airport_graph[connection].is_reachable = True
    return number_of_new_connections, airports_to_reach


class AirportNode:
    def __init__(self, airport):
        self.airport = airport
        self.connections = []
        self.is_reachable = True
        self.unreachable_connections = []
