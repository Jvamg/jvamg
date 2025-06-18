import random

while True:
    low = input('what is the min number? ')
    if low.isdigit():
        low = int(low)
        break


while True:
    high = input('what is the max number? ')
    if high.isdigit():
        high = int(high)
        break

number = random.randint(low, high)

while True:
    guess = input('what is your guess? ')
    if guess.isdigit():
        guess = int(guess)
        break

while guess != number:
    print('You guessed wrong, try again')
    if guess > high or guess < low:
        print('out of range')
        print()
    elif guess > number:
        print('too high')
        print()
    else:
        print('too low')
        print()
    guess = int(input('what is your new guess? '))
print('You guessed right!')
