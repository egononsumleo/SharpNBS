import random, numpy, math
from statistics import median

import abc
from abc import ABC

#gmpy2.get_context().precision = 256
#gmpy2.get_context().precision = 4000

#random.seed(133337373)

class Tree():
    __slots__ = ('l','r', 'parent', 'rchild', 'lchild')

    def __init__(self, l, r, parent = None):
        self.l = l
        self.r = r
        if parent is None:
            self.parent = self
        else:
            self.parent = parent

        self.rchild = None
        self.lchild = None

    def left(self):
        if self.lchild is None:
            m = (self.l + self.r)//2
            self.lchild = Tree(self.l,m, self)
        return self.lchild

    def right(self):
        if self.rchild is None:
            m = (self.l + self.r)//2
            self.rchild = Tree(m,self.r, self)
        return self.rchild

    def up(self):
        return self.parent

class LazySegTree():
    __slots__ = ('l','r', 'rchild', 'lchild', 'value', 'delayed')
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.rchild = None
        self.lchild = None
        self.value = mpfr(1)
        self.delayed = mpfr(1)

    def apply(self, mult):
        self.delayed *= mult
        self.value *= mult

    def push(self):
        if not gmpy2.is_zero(self.delayed - mpfr(1)):
            if self.rchild is not None:
                self.rchild.apply(self.delayed)
            if self.lchild is not None:
                self.lchild.apply(self.delayed)
            self.delayed = mpfr(1)

    def split(self):
        self.push()
        m = (self.l + self.r)//2
        self.lchild = LazySegTree(self.l, m)
        self.lchild.value = (m - self.l)/(self.r - self.l) * self.value
        self.rchild = LazySegTree(m, self.r)
        self.rchild.value = (self.r - m)/(self.r - self.l) * self.value

    # find smallest i such that ...
    def find(self, target):
        self.push()
        if self.l + 1 == self.r:
            return self.l

        if self.rchild is None:
            self.split()

        if self.lchild.value <= target:
            target -= self.lchild.value
            return self.rchild.find(target)
        else:
            return self.lchild.find(target)


    # Edit 
    def edit(self, l, r, mult):
        self.push()
        if r <= self.l or self.r <= l:
            return
        if l <= self.l and self.r <= r:
            self.apply(mult)
        else:
            if self.lchild is None and self.rchild is None:
                self.split()

            self.lchild.edit(l,r,mult)
            self.rchild.edit(l,r,mult)

    def query(self, l, r):
        self.push()
        res = mpfr(0)
        if r <= self.l or self.r <= l:
            return res
        if l <= self.l and self.r <= r:
            res += self.value 
        else:
            if self.lchild is not None:
                res += self.lchild.query(l,r)
            if self.rchild is not None:
                res += self.rchild.query(l,r)
        return res

    def preorder(self):
        self.push()
        if self.lchild is not None:
            self.lchild.preorder()
        print("{} {} {}".format(self.l,self.r,float(self.value)))
        if self.rchild is not None:
            self.rchild.preorder()

    def find_max(self):
        self.push()
        index = -1
        value = 0
        if self.lchild is None and self.rchild is None:
            index = random.randint(self.l, self.r-1)
            value = self.value/(self.r - self.l)  

        if self.lchild is not None:
            candind, candval = self.lchild.find_max()
            if candval > value:
                value = candval
                index = candind

        if self.rchild is not None:
            candind, candval = self.rchild.find_max()
            if candval > value:
                value = candval
                index = candind

        return index, value
"""
N = 100000
test = LazySegTree(1,N)
x = test.find(mpfr('.5'))
test.edit(1,x, 1.2)
print(x)
test.edit(x-1,N, 1.2)
test.edit(x-1,N, 1.2)
test.edit(x-1,N, 1.2)
test.edit(x-1,N, 1.2)
test.preorder()
print(test.find_max())
#print(test.find(mpfr('.5')))
"""
