import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x**5 + 5*np.exp(-3*x)

# Define the first and second derivatives of f(x)
def f_prime(x):
    return 5*x**4 - 15*np.exp(-3*x)

def f_double_prime(x):
    return 20*x**3 + 45*np.exp(-3*x)

# Define the Taylor series approximations
def zero_order_approx(x, x0):
    return f(x0) * np.ones_like(x)

def first_order_approx(x, x0):
    return f(x0) + f_prime(x0)*(x - x0)

def second_order_approx(x, x0):
    return f(x0) + f_prime(x0)*(x - x0) + 0.5*f_double_prime(x0)*(x - x0)**2

# Generate x values
x_values = np.linspace(-1, 3, 400)

# Compute f(x) values
f_values = f(x_values)

# Compute Taylor series approximations for x0 = 1
x0_1 = 1
zero_order_approx_1 = zero_order_approx(x_values, x0_1)
first_order_approx_1 = first_order_approx(x_values, x0_1)
second_order_approx_1 = second_order_approx(x_values, x0_1)

# Compute Taylor series approximations for x0 = 2
x0_2 = 2
zero_order_approx_2 = zero_order_approx(x_values, x0_2)
first_order_approx_2 = first_order_approx(x_values, x0_2)
second_order_approx_2 = second_order_approx(x_values, x0_2)

# Plot the functions and their Taylor series approximations for x0 = 1
plt.figure(figsize=(12, 6))
plt.plot(x_values, f_values, label='f(x)', color='blue')
plt.plot(x_values, zero_order_approx_1, label='Zero Order Approximation', linestyle='--', color='red')
plt.plot(x_values, first_order_approx_1, label='First Order Approximation', linestyle='--', color='green')
plt.plot(x_values, second_order_approx_1, label='Second Order Approximation', linestyle='--', color='orange')
plt.title('Taylor Series Approximations around x0 = 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Plot the functions and their Taylor series approximations for x0 = 2
plt.figure(figsize=(12, 6))
plt.plot(x_values, f_values, label='f(x)', color='blue')
plt.plot(x_values, zero_order_approx_2, label='Zero Order Approximation', linestyle='--', color='red')
plt.plot(x_values, first_order_approx_2, label='First Order Approximation', linestyle='--', color='green')
plt.plot(x_values, second_order_approx_2, label='Second Order Approximation', linestyle='--', color='orange')
plt.title('Taylor Series Approximations around x0 = 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()