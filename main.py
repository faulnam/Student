
def bisection(f, a, b, tol, max_iter):
    if f(a) * f(b) >= 0:
        print("Bisection method may not converge because f(a) and f(b) have the same sign.")
        return None

    iteration = 0
    while (b - a) / 2.0 > tol and iteration < max_iter:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iteration += 1

    return (a + b) / 2.0

# Example usage:
def example_function(x):
    return (x*x*x*x)-(x*x*x) + (2*x*x)-(2*x)-12

a = -2.0 # Left endpoint of the interval
b = 0.0  # Right endpoint of the interval
tolerance = 0.001 # Tolerance for stopping criteria
max_iterations = 13 # Maximum number of iterations

root = bisection(example_function, a, b, tolerance, max_iterations)

if root is not None:
    print(f"Approximate root: {root:.6f}")
else:
    print("Bisection method did not converge.")