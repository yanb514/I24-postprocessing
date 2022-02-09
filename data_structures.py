# A complete working Python
# program to demonstrate all
# insertion methods

# A linked list node
class Node:

    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

# Class to create a Doubly Linked List
class DoublyLinkedList:

    # Constructor for empty Doubly Linked List
    def __init__(self):
        self.head = None
        self.size = 0

    # Given a reference to the head of a list and an
    # integer, inserts a new node on the front of list
    def push(self, new_data):

        # 1. Allocates node
        # 2. Put the data in it
        new_node = Node(new_data)

        # 3. Make next of new node as head and
        # previous as None (already None)
        new_node.next = self.head

        # 4. change prev of head node to new_node
        if self.head is not None:
            self.head.prev = new_node

        # 5. move the head to point to the new node
        self.head = new_node
        self.size += 1
        
    # Given a node as prev_node, insert a new node after
    # the given node
    def insertAfter(self, prev_node, new_data):

        # 1. Check if the given prev_node is None
        if prev_node is None:
            print("the given previous node cannot be NULL")
            return

        # 2. allocate new node
        # 3. put in the data
        new_node = Node(new_data)

        # 4. Make net of new node as next of prev node
        new_node.next = prev_node.next

        # 5. Make prev_node as previous of new_node
        prev_node.next = new_node

        # 6. Make prev_node ass previous of new_node
        new_node.prev = prev_node

        # 7. Change previous of new_nodes's next node
        if new_node.next:
            new_node.next.prev = new_node

        self.size+=1
        
    # Given a reference to the head of DLL and integer,
    # appends a new node at the end
    def append(self, new_data):

        # 1. Allocates node
        # 2. Put in the data
        new_node = Node(new_data)

        # 3. This new node is going to be the last node,
        # so make next of it as None
        # (It already is initialized as None)

        # 4. If the Linked List is empty, then make the
        # new node as head
        if self.head is None:
            self.head = new_node
            self.size += 1
            return

        # 5. Else traverse till the last node
        last = self.head
        while last.next:
            last = last.next

        # 6. Change the next of last node
        last.next = new_node

        # 7. Make last node as previous of new node
        new_node.prev = last
        self.size += 1
        return

    def delete_element(self, x):
        # TODO: make it more efficient if input is a node pointer
        if self.head is None:
            # print("The list has no element to delete")
            return 
        if self.head.next is None:
            if self.head.data == x:
                self.head = None
                self.size = 0
            # else:
            #     print("Item not found")
            return 

        if self.head.data == x:
            self.head = self.head.next
            self.head.prev = None
            self.size -= 1
            return

        n = self.head
        while n.next is not None:
            if n.data == x:
                break
            n = n.next
        if n.next is not None:
            n.prev.next = n.next
            n.next.prev = n.prev
            self.size -= 1
        else:
            if n.data == x:
                n.prev.next = None
                self.size -= 1
            # else:
            #     print("Element not found")
        
    # This function prints contents of linked list
    # starting from the given node
    def printList(self):

        node = self.head
        temp = []
        while node:
            temp.append(node.data)
            # print(" {}".format(node.data))
            # last = node
            node = node.next
        print(temp)

    def convert_to_set(self):

        node = self.head
        s = set()
        while node:
            s.add(node.data)
            # print(" {}".format(node.data))
            # last = node
            node = node.next
        return s
