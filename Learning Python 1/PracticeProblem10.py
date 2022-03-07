"""
Задача: Дано число n. С начала суток прошло n минут.
Определите, сколько часов и минут будут показывать электронные часы в этот момент.
Программа должна вывести два числа: количество часов (от 0 до 23) и количество минут (от 0 до 59).
Учтите, что число n может быть больше, чем количество минут в сутках.
"""

minutes = int(input("Введите количество минут: "))
hours = 0

while minutes >= 60:
    hours += 1
    minutes -= 60

while hours >= 24:
    hours -= 24

print(f"Минуты: {minutes}; Часы: {hours}")