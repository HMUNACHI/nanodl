"""
Implementations of core probability concepts in 
"""

# Probability: Basics

class Probability:
    @classmethod
    def factorial(cls,n):
        running_product = 1
        for factor in range(1, n+1):
            running_product *= factor
        return running_product

    @classmethod
    def combinatorial(cls, choices, choose):
        numerator = cls.factorial(choices)
        denominator = cls.factorial(choices - choose) * cls.factorial(choose)
        return int(numerator / denominator)

    @staticmethod
    def permute(choices, choose):
        running_product = 1
        for _ in range(choose):
            running_product *= choices
            choices -= 1
        return running_product

    @staticmethod
    def prob(n_desired, n_possible):
        return n_desired / n_possible

    @staticmethod
    def complement(prob):
        return 1 - prob

    @staticmethod
    def prob_or(prob_a, prob_b, prob_intersect=0):
        return prob_a + prob_b - prob_intersect

    @staticmethod
    def joint(prob_a, prob_b):
        return prob_a * prob_b

    @staticmethod
    def total(joints=None, marginals=None, posteriors=None):
        if joints:
            return sum(joints)
        if marginals and posteriors:
            pairs = zip(marginals, posteriors)
            return sum([marginal * posterior for marginal, posterior in pairs])
        raise TypeError("Either a list of the joints, \
                         or both list of marginals \
                         and list of posteriors is required")

    @staticmethod
    def conditional(prob_a, prob_b, joint=None, posterior=None):
        if joint:
            return joint / prob_b
        if posterior:
            return (prob_a * posterior) / prob_b
        raise TypeError("Either the joint or posterior probability is required.")

    @staticmethod
    def expectation(values, probabilities):
        pairs = zip(values, probabilities)
        return sum([value * prob for value, prob in pairs])



class Uniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.mean = (a + b) / 2
        self.median = (a + b) / 2
        self.variance = (b - a)**2 / 12
        self.std = self.variance**0.5
        self.skewness = 0
        self.kurtosis = -1.2

    def pmf(self, x):
        if x < self.a or x > self.b:
            return 0
        return 1 / (self.b - self.a)

    def cdf(self, x):
        if x < self.a:
            return 0
        if x > self.b:
            return 1
        return (x - self.a) / (self.b - self.a)



class Bernoulli:
    def __init__(self, p):
        self.p = p
        self.mean = p
        self.mode = 2 * p
        self.variance = p * (1 - p)
        self.std = self.variance**0.5
        self.skewness = (1 - 2*p) / (p*(1 - p))**0.5
        self.kurtosis = (6*(p**2) - 6*p + 1) / (p * (1 - p))

    def pmf(self, x):
        if x not in range(2):
            return 0
        return self.p**x * ((1 - self.p)**(1-x))

    def cdf(self, x):
        if x == 0:
            return 1 - self.p
        if x == 1:
            return self.p
        return 0



class Binomial:
    def __init__(self, p, n):
        self.p = p
        self.n = n
        self.mean = n * p
        self.variance = n * p * (1 - p)
        self.std = self.variance**0.5
        self.C = Probability().combinatorial

    def pdf(self, x):
        return self.C(self.n, x) * self.p**x * (1 - self.p)**(self.n - x)


class Poisson:
    def __init__(self, lamb):
        self.lamb = lamb
        self.mean = lamb
        self.variance = lamb
        self.std = self.variance**0.5
        self.skewness = 1 / lamb**0.5
        self.kurtosis = 3 + (1 / lamb**0.5)
        self.factorial = Probability.factorial

    def pdf(self, x):
        return 2.718218**-self.lamb * self.lamb**x / self.factorial(x)

    def cdf(self, x):
        sums = 0
        for i in range(x):
            sums += (self.lamb**i / self.factorial(i))
        return 2.718218**-self.lamb * sums


class Normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.variance = std**0.5

    def pdf(self, x):
        pi = 3.142
        e = 2.218218
        normalised_x = (x - self.mean)/self.std
        return (1 / self.std * (self.std * pi)**0.5) *\
         (e**-(normalised_x)**2 / 2)

    def cdf(self, x):
        pass