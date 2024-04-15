from random import randint

df = open('data.txt', 'a', encoding='utf-8')

for _ in range(1000):
    df.write(str(randint(0, 36)) + "\n")

df.close()
