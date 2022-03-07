"""
Задача: Электронные часы показывают время в формате h:mm:ss, то есть сначала записывается количество часов,
потом обязательно двузначное количество минут, затем обязательно двузначное количество секунд.
Количество минут и секунд при необходимости дополняются до двузначного числа нулями.
С начала суток прошло n секунд. Выведите, что покажут часы.
"""

seconds = int(input("Введите количество секунд: "))
minutes = 0
hours = 0

while seconds >= 60:
    minutes += 1
    seconds -= 60
while minutes >= 60:
    hours += 1
    minutes -= 60
while hours >= 24:
    hours -= 24

if minutes < 10 and seconds < 10:
    print(f"h:mm:ss - {hours}:0{minutes}:0{seconds}")
elif minutes < 10 and seconds >= 10:
    print(f"h:mm:ss - {hours}:0{minutes}:{seconds}")
elif minutes >= 10 and seconds >= 10:
    print(f"h:mm:ss - {hours}:{minutes}:{seconds}")
else:  # minutes >= 10 and seconds < 10
    print(f"h:mm:ss - {hours}:{minutes}:0{seconds}")