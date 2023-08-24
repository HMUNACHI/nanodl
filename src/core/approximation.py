def derivative(f, x, h=0.0001):
    return (f(x+h) - f(x)) / h

def integral(f, a, b, h=0.0001):
    x = a
    y_prime = 0
    while x <= b:
        y_prime += h * f(x)
        x += h
    return y_prime