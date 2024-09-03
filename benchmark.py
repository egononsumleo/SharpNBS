import RBS, random, time, math
from algorithms import BayesianScreeningSearch, KarpBacktrack, KarpMultiplicativeWeights, NaiveNBS, BZ, BZPatched
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal, sys
from gmpy2 import mpfr 
import gmpy2
from numpy.random import default_rng

# set precision 128
rng = default_rng()
# TODO unhardcode precision
gmpy2.get_context().precision = 1024

class EdgeCase(RBS.Problem):
    __slots__ = ('interval_count', 'epsilon')
    def __init__(self,interval_count, epsilon):
        self.interval_count = interval_count
        self.epsilon = (epsilon)
    def bias(self,x):
        if x == 0:
            return x
        elif x <= 1:
            return x * (.5 + self.epsilon)
        else:
            return .5 + self.epsilon
    def flip(self,x):
        return random.uniform(0,1) <= self.bias(x)
    def get_upper_bound(self):
        return self.interval_count

class BasicCase(RBS.Problem):
    __slots__ = ('interval_count', 'ind', 'epsilon', 'sample_budget', 'tau')
    def __init__(self, interval_count, ind, epsilon, sample_budget, tau=1/2):
        self.interval_count = interval_count
        self.ind = ind
        self.epsilon = epsilon
        self.eps = epsilon
        self.sample_budget = sample_budget
        self.tau = tau
        self.reduction = False
    def bias(self,x):
        if x <= self.ind:
            return self.tau - self.epsilon
        else:
            return self.tau + self.epsilon
    def flip(self,x):
        if rng.uniform(0,1) >= 1/(2 * self.tau) and self.reduction:
            return 0
        else:
            self.sample_budget -= 1
            if self.sample_budget < 0:
                raise AssertionError("Out of sample_budget!")
            return 1 if rng.uniform(0,1) <= self.bias(int(x)) else 0
    def get_upper_bound(self):
        return self.interval_count

class DummyGenerator():
    __slots__ = ('N', 'tau', 'eps')
    def __init__(self, N, tau, eps):
        self.N = N
        self.eps = eps
        self.tau = tau
    def __call__(self, lim):
        ind = random.randint(1, self.N)
        #ind = self.N//2 -1
        return BasicCase(self.N, ind, self.eps, lim, self.tau)

g_EPS = mpfr('.1')

class DummySolver():
    __slots__ = ('solver')
    def __init__(self, solver):
        #print(solver)
        self.solver = solver
    def __call__(self, instance):
        global g_EPS
        # solver = RBS.Solver()
        # TODO unhardcode eps
        #try: 
        return self.solver.solve(instance, eps = g_EPS)
        #except:
         #   return random.randint(1, instance.get_upper_bound())

class DummyChecker():
    def __call__(self, solution, instance):
        return solution == instance.ind

class WideCase(RBS.Problem):
    __slots__= ('N', 'tau', 'eps', 'interval_width', 'interval_start', 'sample_budget', 'reduction')
    def __init__(self, N, tau, eps, interval_width, interval_start, sample_budget):
        self.N = N
        self.eps = eps
        self.tau = tau
        self.interval_width = interval_width
        self.interval_start = interval_start
        self.reduction = False
        self.sample_budget = sample_budget

    def bias(self,x):
        if x <= self.interval_start:
            return self.tau - self.eps
        elif x <= self.interval_start + self.interval_width:
            return ((self.interval_start + self.interval_width - x) * (self.tau - self.eps) + (x - self.interval_start) * (self.tau + self.eps)) / self.interval_width
        else :
            return self.tau + self.eps
    
    def flip(self,x):
        if rng.uniform(0,1) >= 1/(2 * self.tau) and self.reduction:
            return 0
        else:
            self.sample_budget -= 1
            if self.sample_budget < 0:
                raise AssertionError("Out of sample_budget!")
            return 1 if rng.uniform(0,1) <= self.bias(int(x)) else 0
        
    def get_upper_bound(self):
        return self.N

#vals = rng.standard_normal(10)

class WideIntervalGenerator():
    __slots__ = ('N', 'tau', 'eps', 'interval_width')
    def __init__(self, N, tau, eps, interval_width):
        self.N = N
        self.eps = eps
        self.tau = tau
        self.interval_width = interval_width
    def __call__(self, lim):
        interval_start = random.randint(1, self.N - self.interval_width)
        assert(interval_start > 0)
        return WideCase(self.N, self.tau, self.eps, self.interval_width, interval_start, lim)

class LopsidedCase(RBS.Problem):
    __slots__ = ('N', 'tau', 'eps', 'crossing_interval', 'sample_budget', 'reduction')
    def __init__(self, N, tau, eps, crossing_interval, sample_budget):
        self.N = N
        self.eps = eps
        self.tau = tau
        self.crossing_interval = crossing_interval
        self.sample_budget = sample_budget
        self.reduction = False
    
    def bias(self,x):
        if x <= self.crossing_interval:
            return self.tau - self.eps
        else:
            return self.tau + .8*self.eps

    def flip(self,x):
        if rng.uniform(0,1) >= 1/(2 * self.tau) and self.reduction:
            return 0
        else:
            self.sample_budget -= 1
            if self.sample_budget < 0:
                raise AssertionError("Out of sample_budget!")
            return 1 if rng.uniform(0,1) <= self.bias(int(x)) else 0

    def get_upper_bound(self):
        return self.N

class LopsidedCaseGenerator():
    """
    creates a case with biases tau-.8*eps and tau+eps
    """
    def __init__(self, N, tau, eps):
        self.N = N
        self.eps = eps
        self.tau = tau
    def __call__(self, lim):
        crossing_interval = random.randint(1, self.N)
        #crossing_interval = self.N//2 + 10
        return LopsidedCase(self.N, self.tau, self.eps, crossing_interval, lim)

class StandardChecker():
    def __call__(self, solution, instance):
        #print("checking" + str(solution))
        #print("biases: " + str(instance.bias(solution)) + " " + str(instance.bias(solution+1)))
        result = (
            instance.bias(solution) > instance.tau - instance.eps and instance.bias(solution) < instance.tau + instance.eps or
            instance.bias(solution+1) < instance.tau + instance.eps and instance.bias(solution+1) > instance.tau - instance.eps or 
            instance.bias(solution) <= instance.tau - instance.eps and instance.bias(solution+1) >= instance.tau + instance.eps
        )
        #print("result: " + str(result))
        return result

def H(x):
    return -x * math.log2(x) - (1-x) * math.log2(1-x)

def Z(tau, eps):
    return pow(2, (H(tau-eps) - H(tau+eps)) / (2 * eps))

def Q(tau, eps):
    return (1 - tau + eps - 1/(1+Z(tau,eps))) / (2 * eps)

def C(tau, eps):
    return math.log2(1 + Z(tau,eps)) + (tau-eps)*H(tau+eps)/(2*eps) - (tau+eps)*H(tau-eps)/(2*eps)

def OptimalBound(N, tau, eps):
    return math.log2(N)/C(tau,eps)

#SOLVERS = [BayesianScreeningSearch()]
#eps = .1

MAX_PROCESSES = 15

def execute_one_test(generator, solver, checker, iterations):
    problem = RBS.AlgorithmBenchmark(generator, solver, checker, iterations)
    res = 0
    try:
        res = problem.flip(iterations)
    except AssertionError as e:
        pass
    return res

def execute_k_tests(generator, solver, checker, iterations, k):
    p = 0
    for i in range(0, k):
        p += execute_one_test(generator, solver, checker, iterations)
    print(p)
    return p

def test_solver_wide(solver, N, eps, solver_iterations, testing_iterations):
    executor = ProcessPoolExecutor(MAX_PROCESSES)
    futures = []
    per_thread = testing_iterations // MAX_PROCESSES
    #print("per thread: " + str(per_thread))
    total = 0
    for _ in range(0, MAX_PROCESSES):
        total += per_thread
        futures.append(executor.submit(execute_k_tests, WideIntervalGenerator(N, .5, eps, 100), DummySolver(solver), StandardChecker(), solver_iterations, per_thread))
    
    p = 0  
    for future in as_completed(futures):
        p += future.result()
    return p/total

def test_solver_lopsided(solver, N, eps, solver_iterations, testing_iterations):
    executor = ProcessPoolExecutor(MAX_PROCESSES)
    futures = []
    per_thread = testing_iterations // MAX_PROCESSES
    #print("per thread: " + str(per_thread))
    total = 0
    for _ in range(0, MAX_PROCESSES):
        total += per_thread
        futures.append(executor.submit(execute_k_tests, LopsidedCaseGenerator(N, .5, eps), DummySolver(solver), StandardChecker(), solver_iterations, per_thread))
    
    p = 0  
    for future in as_completed(futures):
        p += future.result()
    return p/total

def test_solver_dummy(solver, N, eps, solver_iterations, testing_iterations):
    p = 0
    for i in range(0, testing_iterations):
        # (runs the solver on solver_iteration samples)
        problem = RBS.AlgorithmBenchmark(
            DummyGenerator(N, .5, eps), 
            DummySolver(solver), 
            DummyChecker(), 
            int(2*solver_iterations)
        )
        p += problem.flip(solver_iterations)
        print("Amt", p/(i+1))
    return p

#print(amount)
#test_solver_dummy(BayesianScreeningSearch(), int(N), mpfr(g_EPS), amount, 400)
#exit(0)
#test_solver_lopsided(BayesianScreeningSearch(), int(N), mpfr(g_EPS), amount, 400)
#test_solver(BZ(), int(N), eps, 2000, 400)

AMT = 1000
file_name = "lopsided_test3.txt"

"""
with open(file_name, "w") as f:
    for solver in [BayesianScreeningSearch(), BZ(), BZPatched()]:
        for i in range(1, 8 + 1):
            N = 10 ** (4 * i)
            amount = int(1.2*OptimalBound(N, .5, g_EPS))
            p = test_solver_lopsided(solver, int(N), mpfr(g_EPS), amount, AMT)
            print(str(solver.__class__.__name__)  + " [N = " + str(N) + ", tau = " + str('.5') + ", eps = " + str(.2) + "]: " + str(p))
            f.write(str(solver.__class__.__name__)  + " [N = " + str(N) + ", tau = " + str('.5') + ", eps = " + str(.2) + "]: " + str(p) + '\n')
"""

#N = 1000
#test_solver(BayesianScreeningSearch(), int(N), eps, 1700000, 400)
MAX_PROCESSES = 15

def execute_one_test(generator, solver, checker, iterations):
    problem = RBS.AlgorithmBenchmark(generator, solver, checker, iterations)
    res = 0
    try:
        res = problem.flip(iterations)
    except AssertionError as e:
        pass
    return res

# currently only works when there is one right answer
def generate_plot_data(solvers, generator, iterations_ratios, tau, eps, testing_iterations, output_file):
    output = []
    for iteration_ratio in iterations_ratios:
        iterations = int(iteration_ratio * OptimalBound(generator.N, tau, eps))
        results = []
        for solver in solvers:
            #print("Benchmarking " + str(solver) + " on problem size " + str(generator.N))
            #print("Generator: " + str(generator), "Iterations: " + str(iterations))
            executor = ProcessPoolExecutor(max_workers=MAX_PROCESSES)
            p = 0
            futures = [executor.submit(execute_one_test, generator, DummySolver(solver), DummyChecker(), iterations) for _ in range(0, testing_iterations)]
            for future in as_completed(futures):
                p += future.result()
                # (runs the solver on solver_iteration samples)
                    #print(e)
                #print("Amt", p/(i+1))
            results.append(p/testing_iterations)
        output.append(str(iterations) + "," +','.join(["%.2f" % number for number in results]))
    print("Iterations," + ','.join([str(solver.__class__.__name__) for solver in solvers]))
    for line in output:
        print(line)
    with open(output_file, 'w') as f:
        f.write("Iterations," + ','.join([str(solver.__class__.__name__) for solver in solvers]) + '\n')
        for line in output:
            f.write(line + '\n')

#TAU = .5
#EPS = .1
#ITERATIONS = [.25,.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,10,15,20]

#for TAU in [.5, .8]:
#    for N in LENGTHS:
#        print("Problem " + str(N) + " with tau " + str(TAU) + " and eps " + str(EPS))
#        outfile_name = "data4/tau" + str(TAU) + "eps" + str(EPS) + "N" + str(N) + ".csv"
#        generate_plot_data(SOLVERS, DummyGenerator(N, TAU, EPS), ITERATIONS , TAU, EPS, 100, outfile_name)

#print(400316 * math.log(1e3))

executor = ProcessPoolExecutor(max_workers=MAX_PROCESSES)

def run_benchmark(generator, solver, checker, max_iterations, tau, eps, delta, append_string):
    result = None
    number_iterations = int(2*OptimalBound(max_iterations, tau, eps))
    attempts = 0
    while result is None:
        attempts += 1
        benchmarker = RBS.Solver()
        problem = RBS.AlgorithmBenchmark(generator, DummySolver(solver), checker, max_iterations)
        candidate_result = benchmarker.solve(problem, tau=tau, delta=delta, eps=eps)
        result = candidate_result
    return solver.__class__.__name__ + " " + append_string + " " + str(result)  + " " + str(attempts)

def generate_estimate_with_error_bars(solver, generator, checker, max_iterations, description_string):
    problem = RBS.AlgorithmBenchmark(generator, solver, checker, max_iterations)
    futures = [
        executor.submit(run_benchmark, generator, solver, checker, max_iterations, .8, .05, .01, description_string + " lower"),
        executor.submit(run_benchmark, generator, solver, checker, max_iterations, .90, .05, .01, description_string + " upper")
    ]
    return futures

SOLVERS = [BayesianScreeningSearch(), NaiveNBS(), KarpMultiplicativeWeights()]
ESTIMATES = [12, 12, 80]

# Random Basic case, tau = .5, eps = .1, 
#LENGTHS = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
LENGTHS = [1000, 1000000]

"""
SOLVERS = [KarpBacktrack()]
ESTIMATES = [20000]
LENGTHS = [1000]
"""

futures = []

EPS = .1

for TAU in [.5]:

    # go through lengths in reverse order
    for N in reversed(LENGTHS):
        base_iterations = OptimalBound(N, TAU, EPS)
        generator = WideIntervalGenerator(N, TAU, EPS, int(10 * math.log(N)))
        checker = StandardChecker()
        for i in range(0,len(SOLVERS)):
            max_iterations = int(base_iterations * ESTIMATES[i])
            description_string = "[N = " + str(N) + ", tau = " + str(TAU) + ", eps = " + str(EPS) + ", max_iterations = " + str(max_iterations) + "]"
            futures += generate_estimate_with_error_bars(SOLVERS[i], generator, checker, max_iterations, description_string)

    file_name = "data/" + generator.__class__.__name__ + " tau" + str(TAU) + "eps" + str(EPS) + ".csv"

    # output problem description
    with open(file_name, 'w') as f:
        for future in as_completed(futures):
            result = future.result()
            print(result)
            f.write(result + '\n')

# add SIGINT handler to kill all processes
