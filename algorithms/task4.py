class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_node(node: ListNode):
    prev = None
    cur = node

    while cur is not None:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt

    return prev

def build_list(values):
    head = None

    for value in reversed(values):
        head = ListNode(value, head)

    return head

def to_list(head):
    values = []

    while head is not None:
        values.append(head.val)
        head = head.next

    return values

head = build_list([1, 2, 3, 4, 5])
head1 = reverse_node(head)
print(to_list(head1))

    
