
import RBS, numpy
from algorithms import BayesianScreeningSearch, OptimalBound

# a mock problem, where coin x is 1 with probability logistic(x/LENGTH) and 0 otherwise
class MockProblem(RBS.Problem):
    def __init__(self):
        self.LENGTH = 100000
        self.tau = .5

    def bias(self, x):
        return 1/(1+numpy.exp(-2*(x - self.LENGTH//2)/self.LENGTH)) 

    def bias(self, x):
        if x <= self.LENGTH/2:
            return self.tau - .1
        else:
            return self.tau + .1

    def get_upper_bound(self):
        return 10*self.LENGTH
    def flip(self, x):
        return numpy.random.binomial(1,self.bias(x))

# An example of using NBS on some problem, doubling as a sanity check

# create a default problem

problem = MockProblem()

solver = BayesianScreeningSearch()


# run the solver on the problem
# Note that the allocation of the sample_budget is done in the solver, and not optimized
# can use solved_detailed to control this
number_correct = 0
TRIALS = 1000
EPS = .1
sample_budget = int(15 * OptimalBound(problem.get_upper_bound(), .5, EPS))

# for implementation reasons, we specify the sample budget in the problem itself
problem.sample_budget = sample_budget
# 
problem.eps = EPS

for i in range(TRIALS):
    print(sample_budget)
    solution = solver.solve(problem, EPS)
    if abs(problem.bias(solution) - .5) < EPS:
        number_correct += 1
    print("percent correct: " + str(number_correct/(i+1)))

print("Accuracy: ", number_correct/TRIALS)


