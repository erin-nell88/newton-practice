def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x))/h

#def f(x):
    #return x**2
    
#print(derivative(f, 3))

def derivative2(f, x, h=(1e-5)):
    return derivative(lambda z: derivative(f,z,h), x, h)

#print(derivative2(f, 3)

def newton(f, x0, tolerance = 1e-5, iterations = 100):
    x = x0
    for i in range(iterations):
        first = derivative(f , x)
        second = derivative2(f ,x)

        if abs(first) < tolerance:
            break
            
        x = x - first/second

    return x

def f(x):
    return x**2


min = newton(f, x0=0)
print(min)
print(f(min))