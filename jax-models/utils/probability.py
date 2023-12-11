"""
Implementations of core probability concepts in 
"""

# Probability: Basics

class Probability:
    """
    Class containing methods to compute basic probability concepts.
    """

    @classmethod
    def factorial(cls, n):
        """
        Compute the factorial of a number.

        Args:
            n (int): Input number.

        Returns:
            int: Factorial of the input number.
        """
        running_product = 1
        for factor in range(1, n+1):
            running_product *= factor
        return running_product

    @classmethod
    def combinatorial(cls, choices, choose):
        """
        Compute the binomial coefficient.

        Args:
            choices (int): Total number of choices.
            choose (int): Number of choices to be made.

        Returns:
            int: Binomial coefficient value.
        """
        numerator = cls.factorial(choices)
        denominator = cls.factorial(choices - choose) * cls.factorial(choose)
        return int(numerator / denominator)

    @staticmethod
    def permute(choices, choose):
        """
        Compute the permutation of choices taken choose at a time.

        Args:
            choices (int): Total number of choices.
            choose (int): Number of choices to be made.

        Returns:
            int: Permutation value.
        """
        running_product = 1
        for _ in range(choose):
            running_product *= choices
            choices -= 1
        return running_product

    @staticmethod
    def prob(n_desired, n_possible):
        """
        Compute the probability of an event.

        Args:
            n_desired (int): Number of desired outcomes.
            n_possible (int): Total number of possible outcomes.

        Returns:
            float: Probability of the event.
        """
        return n_desired / n_possible

    @staticmethod
    def complement(prob):
        """
        Compute the complement of a probability.

        Args:
            prob (float): Probability of an event.

        Returns:
            float: Complement of the probability.
        """
        return 1 - prob

    @staticmethod
    def prob_or(prob_a, prob_b, prob_intersect=0):
        """
        Compute the probability of the union of two events.

        Args:
            prob_a (float): Probability of event A.
            prob_b (float): Probability of event B.
            prob_intersect (float): Probability of the intersection of events A and B (default=0).

        Returns:
            float: Probability of the union of events A and B.
        """
        return prob_a + prob_b - prob_intersect

    @staticmethod
    def joint(prob_a, prob_b):
        """
        Compute the joint probability of two independent events.

        Args:
            prob_a (float): Probability of event A.
            prob_b (float): Probability of event B.

        Returns:
            float: Joint probability of events A and B.
        """
        return prob_a * prob_b

    @staticmethod
    def total(joints=None, marginals=None, posteriors=None):
        """
        Compute the total probability based on joints, marginals, or posteriors.

        Args:
            joints (list): List of joint probabilities.
            marginals (list): List of marginal probabilities.
            posteriors (list): List of posterior probabilities.

        Returns:
            float: Total probability computed based on inputs.
        
        Raises:
            TypeError: If neither joints nor both marginals and posteriors are provided.
        """
        if joints:
            return sum(joints)
        if marginals and posteriors:
            pairs = zip(marginals, posteriors)
            return sum([marginal * posterior for marginal, posterior in pairs])
        raise TypeError("Either a list of the joints, or both list of marginals \
                         and list of posteriors is required")

    @staticmethod
    def conditional(prob_a, prob_b, joint=None, posterior=None):
        """
        Compute the conditional probability.

        Args:
            prob_a (float): Probability of event A.
            prob_b (float): Probability of event B.
            joint (float): Joint probability of events A and B.
            posterior (float): Posterior probability of event A.

        Returns:
            float: Conditional probability of event A given event B.
        
        Raises:
            TypeError: If neither joint nor posterior probability is provided.
        """
        if joint:
            return joint / prob_b
        if posterior:
            return (prob_a * posterior) / prob_b
        raise TypeError("Either the joint or posterior probability is required.")

    @staticmethod
    def expectation(values, probabilities):
        """
        Compute the expectation (expected value) of a random variable.

        Args:
            values (list): List of possible values of the random variable.
            probabilities (list): List of corresponding probabilities.

        Returns:
            float: Expectation (expected value) of the random variable.
        """
        pairs = zip(values, probabilities)
        return sum([value * prob for value, prob in pairs])



class Uniform:
    """
    Class representing the Uniform distribution.
    """
    def __init__(self, a, b):
        """
        Initialize the Uniform distribution.

        Args:
            a: Lower bound of the distribution.
            b: Upper bound of the distribution.
        """
        self.a = a
        self.b = b
        self.mean = (a + b) / 2
        self.median = (a + b) / 2
        self.variance = (b - a)**2 / 12
        self.std = self.variance**0.5
        self.skewness = 0
        self.kurtosis = -1.2

    def pmf(self, x):
        """
        Probability Mass Function (PMF) of the Uniform distribution.

        Args:
            x: Input value.

        Returns:
            float: PMF value at the input value.
        """
        if x < self.a or x > self.b:
            return 0
        return 1 / (self.b - self.a)

    def cdf(self, x):
        """
        Cumulative Distribution Function (CDF) of the Uniform distribution.

        Args:
            x: Input value.

        Returns:
            float: CDF value at the input value.
        """
        if x < self.a:
            return 0
        if x > self.b:
            return 1
        return (x - self.a) / (self.b - self.a)



class Bernoulli:
    """
    Bernoulli distribution class representing a discrete probability distribution
    for a binary random variable with two possible outcomes.

    Args:
        p (float): Probability of success (outcome 1).

    Attributes:
        p (float): Probability of success.
        mean (float): Mean of the distribution.
        mode (float): Mode of the distribution.
        variance (float): Variance of the distribution.
        std (float): Standard deviation of the distribution.
        skewness (float): Skewness of the distribution.
        kurtosis (float): Kurtosis of the distribution.

    Methods:
        pmf(x): Probability mass function for the given value.
        cdf(x): Cumulative distribution function for the given value.
    """

    def __init__(self, p):
        self.p = p
        self.mean = p
        self.mode = 2 * p
        self.variance = p * (1 - p)
        self.std = self.variance**0.5
        self.skewness = (1 - 2*p) / (p*(1 - p))**0.5
        self.kurtosis = (6*(p**2) - 6*p + 1) / (p * (1 - p))

    def pmf(self, x):
        """
        Compute the probability mass function (PMF) for the given value.

        Args:
            x (int): Value for which to compute PMF.

        Returns:
            float: Probability of the given value.
        """
        if x not in range(2):
            return 0
        return self.p**x * ((1 - self.p)**(1-x))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) for the given value.

        Args:
            x (int): Value for which to compute CDF.

        Returns:
            float: Cumulative probability up to the given value.
        """
        if x == 0:
            return 1 - self.p
        if x == 1:
            return self.p
        return 0
    

class Binomial:
    """
    Binomial distribution class representing a discrete probability distribution
    for the number of successes in a fixed number of independent Bernoulli trials.

    Args:
        p (float): Probability of success (outcome 1) in each trial.
        n (int): Number of trials.

    Attributes:
        p (float): Probability of success in each trial.
        n (int): Number of trials.
        mean (float): Mean of the distribution.
        variance (float): Variance of the distribution.
        std (float): Standard deviation of the distribution.

    Methods:
        pdf(x): Probability density function for the given value.
    """

    def __init__(self, p, n):
        self.p = p
        self.n = n
        self.mean = n * p
        self.variance = n * p * (1 - p)
        self.std = self.variance**0.5
        self.C = Probability().combinatorial

    def pdf(self, x):
        """
        Compute the probability density function (PDF) for the given value.

        Args:
            x (int): Number of successes.

        Returns:
            float: Probability of the given number of successes.
        """
        return self.C(self.n, x) * self.p**x * (1 - self.p)**(self.n - x)


class Poisson:
    """
    Poisson distribution class representing a discrete probability distribution
    for the number of events occurring in a fixed interval of time or space.

    Args:
        lamb (float): Average rate of events.

    Attributes:
        lamb (float): Average rate of events.
        mean (float): Mean of the distribution.
        variance (float): Variance of the distribution.
        std (float): Standard deviation of the distribution.
        skewness (float): Skewness of the distribution.
        kurtosis (float): Kurtosis of the distribution.

    Methods:
        pdf(x): Probability mass function for the given value.
        cdf(x): Cumulative distribution function for the given value.
    """

    def __init__(self, lamb):
        self.lamb = lamb
        self.mean = lamb
        self.variance = lamb
        self.std = self.variance**0.5
        self.skewness = 1 / lamb**0.5
        self.kurtosis = 3 + (1 / lamb**0.5)
        self.factorial = Probability.factorial

    def pdf(self, x):
        """
        Compute the probability mass function (PMF) for the given value.

        Args:
            x (int): Number of events.

        Returns:
            float: Probability of the given number of events.
        """
        return 2.718218**-self.lamb * self.lamb**x / self.factorial(x)

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) for the given value.

        Args:
            x (int): Number of events.

        Returns:
            float: Cumulative probability up to the given number of events.
        """
        sums = 0
        for i in range(x):
            sums += (self.lamb**i / self.factorial(i))
        return 2.718218**-self.lamb * sums


class Normal:
    """
    Class representing the Normal distribution.
    """
    def __init__(self, mean, std):
        """
        Initialize the Normal distribution.

        Args:
            mean: Mean of the distribution.
            std: Standard deviation of the distribution.
        """
        self.mean = mean
        self.std = std
        self.variance = std**0.5

    def pdf(self, x):
        """
        Probability Density Function (PDF) of the Normal distribution.

        Args:
            x: Input value.

        Returns:
            float: PDF value at the input value.
        """
        pi = 3.142
        e = 2.218218
        normalised_x = (x - self.mean)/self.std
        return (1 / (self.std * (self.std * pi)**0.5)) *\
               (e**-(normalised_x)**2 / 2)

    def cdf(self, x):
        """
        Cumulative Distribution Function (CDF) of the Normal distribution.

        Args:
            x: Input value.

        Returns:
            float: CDF value at the input value.
        """
        pass  # Add implementation for the CDF