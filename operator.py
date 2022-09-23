
# Python Program to perform addition
# of two complex numbers using binary
# + operator overloading.
 from typing import overload



class complex:
    def __init__(self, a, b):
        self.a = a
        self.b = b
 
     # adding two objects
    def __add__(self, other):
        return complex(self.a + other.a,self.b + other.b)
    def __repr__(self):
        stringggg = str(self.a)  +" + " + str(self.b) + "*i"
        return stringggg
 
Ob1 = complex(1, 2)
Ob2 = complex(2, 3)
Ob3 = Ob1 + Ob2
print(Ob3)