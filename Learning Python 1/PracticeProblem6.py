"""
Задача: Дано натуральное число. Выведите его последнюю цифру.
"""

number = int(input("Введите натуральное число: "))

print(f"Его последняя цифра: {number % 10}")