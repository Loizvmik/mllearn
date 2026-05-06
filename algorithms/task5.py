class Stack:
    def __init__(self):
        self.mas=[]
    
    def push(self,val):
        self.mas.append(val)

    def pop(self):
        if len(self.mas)==0:
            exit(0)
        val=self.mas[len(self.mas)-1]
        self.mas=self.mas[:len(self.mas)-1]
        return val
    
    def length(self):
        return len(self.mas)

def func(inp):
    mas = Stack()
    for ch in inp:
        if ch =='{' or ch == '(' or ch=='[':
            mas.push(ch)
        elif ch == '}':
            a=mas.pop()
            if a!='{':
                return False
        elif ch == ')':
            a=mas.pop()
            if a!='(':
                return False
        elif ch == ']':
            a=mas.pop()
            if a!='[':
                return False
    if mas.length()==0:
        return True
    else:
        print(mas.length())
        return False

inp='([])'
print(func(inp))
