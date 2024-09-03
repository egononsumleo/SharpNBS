import RBS, random, time, math, numpy
from Discrete import Tree, LazySegTree
from gmpy2 import mpfr

"""
Contains helper functions and the implementations of the noisy binary search algorithms
"""

# binary entropy function
def H(x):
    return -x * math.log2(x) - (1-x) * math.log2(1-x)

# a quantinty that is used in the optimal bound, see the paper for more details
def Z(tau, eps):
    return pow(2, (H(tau-eps) - H(tau+eps)) / (2 * eps))

# the quantile that gets the maximum amount of information
def Q(tau, eps):
    return (1 - tau + eps - 1/(1+Z(tau,eps))) / (2 * eps)

# the capacity of the "asymmetric binary channel"
def C(tau, eps):
    return math.log2(1 + Z(tau,eps)) + (tau-eps)*H(tau+eps)/(2*eps) - (tau+eps)*H(tau-eps)/(2*eps)

# the optimal number of iterations for the noisy binary search given by fano's inequality
def OptimalBound(N, tau, eps):
    return math.log2(N)/C(tau,eps)

# short hand for logs
def log(x):
    return math.log(x)

def ln(x):
    return math.log(x)

def log2(x):
    return math.log(x,2)

# estimates the bias of a coin (the <k>th coin in <instance>), up to an error of <epsilon> with failure probability <delta>
def estimate_bias(instance, k, epsilon, delta):
    N = int(log(2/delta)/(2 * (epsilon) ** 2))
    p = 0
    for i in range(0, N):
        p += instance.flip(k)
    p /= N
    return p 

# simulates a coin with larger bias via "estimate_bias". If it would exceed the sample budget, it returns a random coin flip
def flip(instance, k, epsilon, delta):
    if instance.sample_budget <= int(log(2/delta)/(2 * (epsilon) ** 2)):
        instance.sample_budget = 0
        return random.randint(0,1)
    return 1 if estimate_bias(instance, k, epsilon, delta) >= .5 else 0

class SegTree(object):
    """
    SegTree with multiply, range sum, query
    """

    def __init__(self, l, r, value):
        self.l = l
        self.r = r
        self.value = mpfr(value)
        self.delayed = mpfr(1)
        self.left = None
        self.right = None

    def query(self, l, r):
        if l <= self.l and self.r <= r:
            return self.value
        elif r < self.l or l > self.r:
            return 0
        else:
            self.push()
            res = mpfr(0)
            if self.left:
                res += self.left.query(l,r)
            if self.right:
                res += self.right.query(l,r)
            return res

    def mid(self):
        return (self.l + self.r)//2

    def apply(self, x):
        self.value *= x
        self.delayed *= x

    def push(self):
        self.split()
        if self.delayed != mpfr(1):
            if self.left != None:
                self.left.apply(self.delayed)
            if self.right != None:
                self.right.apply(self.delayed)
            self.delayed = mpfr(1)

    def split(self):
        """
        Adds child nodes if they don't already exist
        """
        if self.left == None and self.right == None and self.l != self.r:
            self.delayed = mpfr(1)
            m = self.mid()
            ratio = mpfr((m - self.l + 1))/mpfr((self.r - self.l + 1))
            assert ratio > 0
            self.left = SegTree(self.l, m, self.value * ratio)
            assert ratio < 1
            self.right = SegTree(m + 1, self.r, self.value * (1 - ratio))

    def find(self, p):
        """
        Find the least i such that sum of values(1) + ... + values(i) >= p
        """
        if self.l == self.r:
            return self.l

        self.push()
        if self.left.value >= p:
            return self.left.find(p)
        return self.right.find(p - self.left.value)

    def build(self):
        if self.right is not None and self.left is not None:
            self.value = self.right.value + self.left.value

    def mult(self, l, r, value):
        self.push()
        if l <= self.l and self.r <= r:
            self.apply(value)
        elif r < self.l or l > self.r:
            return 
        else:
            if self.left:
                self.left.mult(l,r,value)
            if self.right:
                self.right.mult(l,r,value)
            self.build()

    def print(self):
        if self.left:
            self.left.print()
        print("[{} {}] {}".format(self.l,self.r,self.value))
        if self.right:
            self.right.print()

    def find_max(self):
        # searches through the whole tree
        if self.left is None and self.right is None:
            value = self.value/(mpfr(self.r) - mpfr(self.l) + 1)
            index = random.randint(self.l, self.r)
            assert(index >= self.l and index <= self.r)
            return index, value

        self.push()
        value = 0
        ind = -1
        if self.left != None:
            candind, candvalue = self.left.find_max()
            if candvalue > value:
                value = candvalue
                ind = candind
        if self.right != None:
            candind, candvalue = self.right.find_max()
            if candvalue > value:
                value = candvalue
                ind = candind
        return ind, value

    def size_of_tree(self):
        if self.left is None and self.right is None:
            return 1
        else:
            return 1 + self.left.size_of_tree() + self.right.size_of_tree()

# Implements BayesianScreeningSearch
class BayesianScreeningSearch():
    
    def bayesianLearn(self, instance, n, eps, iterations):
        # the prior is a uniform distribution
        w = SegTree(1,n - 1,1)
        L = []
        tau = mpfr(instance.tau)
        eps = mpfr(eps)
        m = Q(tau,eps)
        

        # the multiplication factors
        doo = (1 - tau - eps)/(1 - tau - (2*m-1)*eps)
        dol = (1 - tau + eps)/(1 - tau - (2*m-1)*eps)
        dlo = (tau + eps)/(tau + (2*m-1)*eps)
        dll = (tau - eps)/(tau + (2*m-1)*eps)

        for _ in range(0, iterations):
            x = w.find(m)
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
            d = instance.flip(to_query)

            if d == 0:
                w.mult(1,x-1,doo)
                w.mult(x,x,(doo * left + dol * right))
                w.mult(x+1,n-1,dol)
            else:
                w.mult(1,x-1,dlo)
                w.mult(x,x,dlo * left + dll * right)
                w.mult(x+1,n-1,dll)

        # finish
        return L

    def reductionToGamma(self, instance, n, eps, iterations, count = None):
        if count is None or count == 0:
            count = int(log(n)**(2))
        L = self.bayesianLearn(instance, n, eps, iterations)
        L = sorted(L)
        R = [1]
        for i in range(1, count):
            # clamp the index to valid values in L
            ind = min(int((iterations*i)//count), len(L) - 1)
            ind = max(ind, 0)
            R.append(L[ind])
        
        R.append(n)
        # remove duplicates
        return sorted(list(set(R)))

    def trueNBS(self, O, n, eps, delta):
        # find parameters
        initial_guess = [2,2,ln(n), eps * max(1 - numpy.cbrt(ln(1/delta) / ln(n)), 1/2)]
        initial_guess = [1]

    def convertOracle2(self, instance, R, n):
        return lambda x : 0 if x < n else instance.flip(R[int(round(x)) - (n)]) if x <= 2*n - 2 else 1

    def simpleNBS(self, instance, R, n, eps, iterations):
        number = max(int(iterations/math.ceil(math.log(n, 2))), 1)
        tau = instance.tau
        l = 1
        r = n 
        while l < r-1:
            m = (l + r)//2
            p = 0
            for i in range(0, number):
                p += instance.flip(R[m-1])
            p /= number
            if p < tau:
                l = m
            else:
                r = m
        return l

    # a simple implementation of the noisy binary search algorithm, for use in the second stage
    def simpleNBS2(self, instance, R, n, eps, iterations):
        f = self.convertOracle2(instance, R, n)
        w = SegTree(1, 3*n - 2, 1)
        og_n = n
        n = 3*n - 2
        L = []
        tau = mpfr(instance.tau)
        eps = mpfr(eps)
        z = pow(2,(H(tau-eps) - H(tau + eps))/(2*eps))
        m = ((1-tau+eps) - 1/(1+z))/(2 * eps)

        doo = (1 - tau - eps)/(1 - tau - (2*m-1)*eps)
        dol = (1 - tau + eps)/(1 - tau - (2*m-1)*eps)
        dlo = (tau + eps)/(tau + (2*m-1)*eps)
        dll = (tau - eps)/(tau + (2*m-1)*eps)

        for i in range(0, iterations):
            x = w.find(m)
            mid = w.query(x,x)
            left = m - w.query(0,x-1)
            right = w.query(0,x) - m

            left/=mid
            right/=mid

            if left <= m:
                to_query = x
            else:
                to_query = x + 1
            
            # TODO check 0 or 1 index
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
        return sorted(L)[len(L)//2] - (og_n)

    def testCoin(self, problem, k, iterations):
        if iterations == 0:
            return random.randint(0,1)
        p = mpfr(0)
        for _ in range(0, iterations):
            p += problem.flip(k)
        p /= iterations
        return p 

    def solve(self, problem_instance, eps):
        N = problem_instance.get_upper_bound()
        iterations = problem_instance.sample_budget
        # we split the sample budget into three parts, one for each stage of the algorithm

        # initial reduction
        iterations1 = 0
        # second stage
        iterations2 = 0
        # testing
        iterations3 = 0

        """
        NOTE, the chosen splitting has not been optimized, and is just a guess
        For certain instances, for example when you are guaranteed to have only one valid answer,
        the first stage is really the only necessary one
        """

        w1 = math.log(N)
        w2 = 3*math.log(math.log(N))
        w3 = math.log(math.log(N))
        
        iterations1 += int(iterations * (w1)/(w1 + w2 + w3))
        iterations2 += int(iterations * (w2)/(w1 + w2 + w3))
        iterations3 += int(iterations * (w3)/(w1 + w2 + w3))

        delta = iterations1 + iterations2 + iterations3 - problem_instance.sample_budget
        # in case of rounding errors
        iterations1 -= delta

        # we need to have at least 2 iterations in each stage
        if iterations1 <= 2 or iterations2 <= 2 or iterations3 <= 2:
            return random.randint(1,N)

        assert iterations1 + iterations2 + iterations3 <= iterations

        return self.solve_detailed(problem_instance, iterations1, iterations2, iterations3, eps)

    def solve_detailed(self, problem_instance, stage1_iterations, stage2_iterations, stage3_iterations, eps):
        N = problem_instance.get_upper_bound()
        # we split the sample budget into three parts, one for each stage of the algorithm

        """
        NOTE, the chosen splitting has not been optimized, and is just a guess
        For certain instances, for example when you are guaranteed to have only one valid answer,
        the first stage is really the only necessary one
        """

        iterations1 = stage1_iterations
        iterations2 = stage2_iterations
        iterations3 = stage3_iterations

        # can probably be optimized
        epsprime = eps

        # stage one
        R = self.reductionToGamma(problem_instance, N, epsprime, iterations1)

        Rprime = sorted(list(set(R)))

        # stage two
        x = self.simpleNBS2(problem_instance, Rprime, len(Rprime), epsprime, iterations2)
        if x == len(R) - 1:
            x-=1

        # stage three
        d1 = self.testCoin(problem_instance, R[x] + 1, iterations3//2) - (problem_instance.tau - problem_instance.eps)
        d2 = problem_instance.tau + problem_instance.eps - self.testCoin(problem_instance, R[x+1] - 1, iterations3//2)
        if d1 > d2:
            return R[x]
        else:
            return R[x+1] - 1

class KarpBacktrack():
    def iterate(self, problem, eps, tree):
        l = tree.l
        r = tree.r
        m = (l + r)//2

        lcnt = 0
        for i in range(0,2):
            lcnt += flip(problem, l, eps, 1/6)

        if lcnt == 2:
            return tree.up()

        rcnt = 0
        for _ in range(0,2):
            lcnt += flip(problem, r, eps, 1/6)
            
        if rcnt == 2:
            return tree.up()

        if flip(problem, m, eps, 1/6):
            return tree.left()
        else:
            return tree.right()

    def solve(self, problem, eps):
        N = int(problem.get_upper_bound())
        tree = Tree(1,N)
        tau = problem.tau
        eps = eps/(2*tau)

        results = []

        while problem.sample_budget > 0:
            tree = self.iterate(problem, eps, tree)
            if random.uniform(0,1) < 1/math.log(N):
                k = int(300 * math.log(N))
                l = tree.l
                ha = 0
                for i in range(0, k):
                    ha += flip(problem, l, eps, 1/6)
                ha/=k

                if ha >= 1/4 and ha <= 3/4:
                    results.append(tree.l)
                    tree = Tree(1,N)
                
                r = tree.r
                hb = 0
                for i in range(0, k):
                    hb += flip(problem, r, eps, 1/6)
                hb/=k

                if hb >= 1/4 and hb <= 3/4:
                    results.append(tree.r)
                    tree = Tree(1,N)

                if tree.l + 1 == tree.r and ha < .5 and .5 < hb:
                    results.append(tree.l)
                    tree = Tree(1,N)
        if len(results) > 0:
            results = sorted(results)
            # return the median result
            return results[len(results)//2]
        else:
            return random.randint(1,N)

class BZ():
    def solve(self, problem, eps):
        n = mpfr(problem.get_upper_bound())
        tree = SegTree(1, n, 1)
        problem.reduction = True
        tau = problem.tau
        eps = mpfr(eps/(2 * tau))

        while problem.sample_budget > 0:
            x = tree.find(.5)
            x += random.randint(0,1)
            q = tree.query(1,x-1)
            result = problem.flip(x)
            if result == 1:
                denom = q*(mpfr('.5')+eps) + (1-q)*(mpfr('.5')-eps)
                tree.mult(1,x-1,(.5+eps)/denom)
                tree.mult(x,n,(.5-eps)/denom)
            else:
                denom = q*(mpfr('.5')-eps) + (1-q)*(mpfr('.5')+eps)
                tree.mult(1,x-1,(.5-eps)/denom)
                tree.mult(x,n,(.5+eps)/denom)   

            if q > .9999 or q < .0001:
                result, amt = tree.find_max()
                if result == problem.crossing_interval:
                    print("Found it")
                else:
                    print("Didn't find it")
                return result
        
        result, amt = tree.find_max()
        if amt > .99:
            return result
        else:
            return -1

class BZPatched():
    def solve(self, problem, eps):
        n = mpfr(problem.get_upper_bound())
        tree = SegTree(1, n, 1)
        problem.reduction = True
        tau = problem.tau
        eps = mpfr(eps/(2 * tau))

        while problem.sample_budget > 0:
            x = tree.find(.5)
            x += random.randint(0,1)
            q = tree.query(1,x-1)
            result = problem.flip(x)
            if result == 1:
                denom = q*(mpfr('.5')+eps) + (1-q)*(mpfr('.5')-eps)
                tree.mult(1,x-1,(.5+eps)/denom)
                tree.mult(x,n,(.5-eps)/denom)
            else:
                denom = q*(mpfr('.5')-eps) + (1-q)*(mpfr('.5')+eps)
                tree.mult(1,x-1,(.5-eps)/denom)
                tree.mult(x,n,(.5+eps)/denom)   

            if q > .9999 or q < .0001:
                result, amt = tree.find_max()
                if result == problem.crossing_interval:
                    print("Found it")
                else:
                    print("Didn't find it")
                return result
        
        result, amt = tree.find_max()
        return result


class KarpMultiplicativeWeights():
    def solve(self, problem, eps):
        n = problem.get_upper_bound()
        tree = SegTree(1, n, 1)
        L = []
        problem.reduction = True
        tau = problem.tau
        eps = eps/(2 * tau)
        samples_per_coin = int(1.1*log(12)/(2 * (eps) ** 2))

        final_iterations = int(math.ceil(math.log(math.log(n))))
        main_iterations = int(problem.sample_budget/samples_per_coin - 2*final_iterations)
        while problem.sample_budget > 2 * final_iterations * int(math.log(12)/(2 * (eps**2)) + 1) + int(math.log(12)/(2 * (eps**2))):
            x = tree.find(.5)
            L.append(x)
            x += random.randint(0,1)
            result = flip(problem, x, eps, 1/6)
            q = tree.query(1,x-1)
            if result == 1:
                tree.mult(1,x-1,3/(2+q))
                tree.mult(x,n,2/(2+q))
            else:
                tree.mult(1,x-1,2/(3-q))
                tree.mult(x,n,3/(3-q))

        u, value = tree.find_max()
        if len(L) == 0:
            return u
        v = int(numpy.median(L))

        hv = 0
        hv1 = 0
        for _ in range(0, final_iterations):
            hv += flip(problem, v, eps, 1/6)
            hv1 += flip(problem, v + 1, eps, 1/6)

        hv/=final_iterations
        hv1/=final_iterations

        res = -1
        if hv1 <= 5/24 or hv >= 19/24:
            res = u
        else:
            res = v
        return res

class NaiveNBS():
    def solve(self, problem, eps):
        n = problem.get_upper_bound()
        tau = problem.tau
        iterations = max(int(problem.sample_budget/math.ceil(math.log(n, 2))),1)
        l = 1
        r = n 
        while l < r-1:
            m = (l + r)//2
            p = 0
            for _ in range(0, iterations):
                p += problem.flip(m)
            p /= iterations
            if p < tau:
                l = m
            else:
                r = m
        return l
