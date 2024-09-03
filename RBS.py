import random, numpy, math
from statistics import median
from algorithms import SegTree
#from gmpy2 import mpfr

import abc
from abc import ABC

#gmpy2.get_context().precision = 128

#random.seed(133337373)

class AVL():
    class node():   

        __slots__ = ('node_left', 'node_right', 'interval_left', 'interval_right', 'node_sum', 'interval_sum', 'delayed_operation', 'left_child', 'right_child', 'height', 'balance', 'parent')

        def __init__(self, left, right, node_sum):
            self.node_left = (left)
            self.node_right = (right)
            self.interval_left = (left)
            self.interval_right = ((right))
            self.node_sum = ((node_sum))
            self.interval_sum = self.node_sum
            self.delayed_operation = (1)
            self.left_child = None
            self.right_child = None
            self.height = int(0)
            self.balance = int(0)
            self.recalc_meta()

        def build(self):
            self.interval_sum = self.node_sum;
            if self.left_child != None:
                self.interval_sum += self.delayed_operation * self.left_child.interval_sum
            if self.right_child != None:
                self.interval_sum += self.delayed_operation * self.right_child.interval_sum

        def push(self):
            if not self.delayed_operation == (1):
                if self.left_child != None:
                    self.left_child.apply(self.delayed_operation)
                if self.right_child != None:
                    self.right_child.apply(self.delayed_operation)
                self.delayed_operation = (1)

        def apply(self, value):
            self.delayed_operation *= value
            self.interval_sum *= value
            self.node_sum *= value

        def recalc_meta(self):
            left_height = self.left_child.height if self.left_child != None else 0
            right_height = self.right_child.height if self.right_child != None else 0
            self.height = 1 + max(left_height, right_height)
            self.balance = left_height - right_height
            self.interval_left = self.left_child.interval_left if self.left_child != None else self.node_left
            self.interval_right = self.right_child.interval_right if self.right_child != None else self.node_right
        def edit(self, l, r, mult):
            self.push()

            if l <= self.interval_left and self.interval_right <= r:
                self.apply(mult)
                return

            if self.left_child != None and l < self.node_left:
                self.left_child.edit(l, min(self.node_left, r), mult)

            if self.right_child != None and self.node_right < r:
                self.right_child.edit(max(self.node_right, l),r, mult)

            if l <= self.node_left and self.node_right <= r:
                self.node_sum *= mult

            self.build()

        # really ugly
        def descend(self, value):
            self.push()
            if self.left_child != None:
                if value <= self.left_child.interval_sum:
                    return self.left_child, value
                else:
                    value -= self.left_child.interval_sum

            if value <= self.node_sum:
                return self, value

            value -= self.node_sum 
            return self.right_child, value

    __slots__ = ('super_root')

    def __init__(self, lower_bound = 0, upper_bound = 1):
        self.super_root = self.node(-1,-1,-1)
        self.super_root.right_child = self.node(lower_bound,upper_bound,1)
        self.super_root.right_child.parent = self.super_root

    def root(self):
        return self.super_root.right_child

    def try_split(self,y):
        current_node = self.root() 

        while True:
            next_node, y = current_node.descend(y)
            if next_node == None:
                break
            if next_node == current_node:
                if abs(y) < ('1e-50'):
                    return current_node.node_left
                if abs(y-current_node.node_sum) < ('1e-50'):
                    return current_node.node_right

                return self.split(current_node, y)
            current_node = next_node

    def push_up(self, node):
        while node != self.super_root:
            node.build()
            node = node.parent

    def right_rotate(self, node):
        og_node = node
        og_child = node.left_child

        og_node.push()
        og_child.push()

        og_node.left_child = og_node.left_child.right_child
        if og_node.left_child != None:
            og_node.left_child.parent = og_node

        if og_node.parent.left_child == og_node:
            og_node.parent.left_child = og_child
            og_child.parent = og_node.parent
        else:
            og_node.parent.right_child = og_child
            og_child.parent = og_node.parent

        og_node.parent = og_child
        og_child.right_child = og_node

        og_node.build()
        og_child.build()

        og_node.recalc_meta()
        og_child.recalc_meta()

    def left_rotate(self, node):
        og_node = node
        og_child = node.right_child

        og_node.push()
        og_child.push()

        og_node.right_child = og_node.right_child.left_child
        if og_node.right_child != None:
            og_node.right_child.parent = og_node

        if og_node.parent.left_child == og_node:
            og_node.parent.left_child = og_child
            og_child.parent = og_node.parent
        else:
            og_node.parent.right_child = og_child
            og_child.parent = og_node.parent

        og_node.parent = og_child
        og_child.left_child = og_node

        og_node.build()
        og_child.build()

        og_node.recalc_meta()
        og_child.recalc_meta()
        

    def split(self, node, y):
        og_sum = node.node_sum
        node.push()
        new_interval_left = node.node_left + (node.node_right - node.node_left) * y / node.node_sum
        new_interval_right = node.node_right
        new_interval_sum = node.node_sum - y
        node.node_right = new_interval_left
        node.node_sum = y
        # new node goes to the right subtree
        while True:
            node.push()
            if node.node_right <= new_interval_left:
                if node.right_child == None:
                    node.right_child = self.node(new_interval_left,new_interval_right,new_interval_sum)
                    node.right_child.parent = node
                    node = node.right_child
                    break
                node = node.right_child
            else:
                if node.left_child == None:
                    node.left_child = self.node(new_interval_left,new_interval_right,new_interval_sum)
                    node.left_child.parent = node
                    node = node.left_child
                    break
                node = node.left_child

        self.push_up(node)


        # rebalance
        while node != self.super_root:
            node.recalc_meta()
            if node.balance < -1:
                if node.right_child.balance >= 1:
                    self.right_rotate(node.right_child)
                self.left_rotate(node)
            elif node.balance > 1:
                if node.left_child.balance <= -1:
                    self.left_rotate(node.left_child)
                self.right_rotate(node)
            node = node.parent

        # returns the x split
        return new_interval_left

    def edit(self, l, r, mult):
        self.super_root.right_child.edit((l), (r), (mult))


    def _preorder(self, node):
        print("[%f,[%f,%f],%f] : [%f,%f]"%(node.interval_left,node.node_left,node.node_right,node.interval_right,node.interval_sum,node.node_sum))
        if node.left_child != None:
            self._preorder(node.left_child)
        if node.right_child != None:
            self._preorder(node.right_child)
            
    def preorder(self):
        self._preorder(self.super_root.right_child)

# a problem to solve
class Problem(ABC):
    # function takes in one parameter, [lower_bound, upper_bound]
    # checker checks if the function suceeded
    @abc.abstractmethod
    def flip(self, x):
        pass
    @abc.abstractmethod
    def get_upper_bound():
        pass

class AlgorithmBenchmark(Problem):

    __slots__ = ('generator', 'solver', 'checker', 'upper_bound', 'sample_budget', 'tau')

    def __init__(self, generator, solver, checker, upper_bound):
        self.generator = generator
        self.solver = solver
        self.checker = checker
        self.upper_bound = 2*upper_bound + 1

    def flip(self, x):
        correct = False
        try:
            problem_instance = self.generator(x)
            solution = self.solver(problem_instance)
            #print(solution, problem_instance.ind)
            correct = self.checker(solution, problem_instance)
        except AssertionError as e:
            print(e)
            pass
        return 1 if correct == True else 0

    def get_upper_bound(self):
        return self.upper_bound

def log2(x):
    return math.log(x, 2)

def ln(x):
    return math.log(x)

def round_cnt(n, eps, delta):

    #n, eps, delta = args

    total = 0.0
    beta = 2

    ### stage 1 iteration
    d = log2((1 + beta * eps)/(1 - beta * eps))
    g = .5 * (((.5 + eps) * log2(1 + beta*eps) + (.5 - eps) * log2(1 - beta*eps)))

    total += .5 * log2(n)/g
    total += (d ** 2) * ln(1/delta)/(4 * g ** 2)
    total += d * numpy.sqrt(d * d * ln(1/delta) + 8 * g * .5 * log2(n) * ln(1/delta))/( 4 * (g ** 2))

    return total

def log(x):
    return math.log(x)

def estimate_bias(instance, k, tau, epsilon, delta):
    N = int(2 * tau * (1-tau) * log(2/delta)/((epsilon ** 2)))
    print(N)
    p = 0
    for i in range(0, N):
        #print("flipping")
        p += instance.flip(k)
    p /= N
    return p

def flip(instance, k, epsilon, delta):
    return 1 if estimate_bias(instance, k, epsilon, delta) >= .5 else 0

def H(x):
    return - x * log2(x) - (1-x) * log2(1 - x)

class Solver():
    def __init__(self):
        pass

    def convertOracle2(self, problem, n):
        return lambda x : 0 if x < n else problem.flip(int(round(x)) - (n - 1)) if x <= 2*n - 2 else 1

    def simpleNBS(self, problem, tau, n, eps, delta, iterations):

        f = self.convertOracle2(problem, n)
        #w = AVL(1, 3*n - 2)
        w = SegTree(1, 3*n - 2, 1)
        og_n = n
        n = 3*n - 2
        interval_limit = int(4/(eps*eps))
        adaptive_check = 2

        L = []
        z = pow(2,(H(tau-eps) - H(tau + eps))/(2*eps))
        m = ((1-tau+eps) - 1/(1+z))/(2 * eps)

        doo = (1 - tau - eps)/(1 - tau - (2*m-1)*eps)
        dol = (1 - tau + eps)/(1 - tau - (2*m-1)*eps)
        dlo = (tau + eps)/(tau + (2*m-1)*eps)
        dll = (tau - eps)/(tau + (2*m-1)*eps)

        for i in range(0, iterations):
            x = w.find(m)
            #print(x - (og_n - 1))
            mid = w.query(x,x)
            left = m - w.query(0,x-1)
            right = w.query(0,x) - m

            left/=mid
            right/=mid

            if left <= m:
                to_query = x
            else:
                to_query = x + 1
            
            L.append(x)
            d = f(to_query)

            if d == 0:
                w.mult(1,x-1,doo)
                w.mult(x,x,doo * left + dol * right)
                w.mult(x+1,n-1,dol)
            else:
                w.mult(1,x-1,dlo)
                w.mult(x,x,dlo * left + dll * right)
                w.mult(x+1,n-1,dll)

            if i % interval_limit == interval_limit - 1:
                guess = int(numpy.median(L) - (og_n - 1))
                print("estimating guess " + str(guess))
                bias = estimate_bias(problem, guess, tau, 2*eps/3, delta/(adaptive_check*adaptive_check))
                print("resulting bias " + str(bias))
                adaptive_check += 1
                if abs(bias - tau) <= 2*eps/3:
                    return guess

        return int(numpy.median(L) - (og_n - 1))

    def solve(self, problem, tau = None, delta = None, eps = None):
        iterations = int(2 * tau * (1-tau) * math.log(problem.get_upper_bound())/(eps*eps) + 32*tau*(1-tau)*math.log(1/delta)/(eps*eps))
        print("iter: " + str(iterations))
        return self.simpleNBS(problem, tau, problem.get_upper_bound(), eps, delta/2, iterations)

