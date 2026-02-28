def divide(a, b):
    return a / b

def greet(name):
    print("Hello, " + name)

def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

def first_element(lst):
    return lst[0]

if __name__ == "__main__":
    print(divide(10, 0))
    print(average([]))
    print(first_element([]))
