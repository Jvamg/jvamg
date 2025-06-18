menu = {'popcorn': 5.00,
        'chocolate': 2.00,
        'fries': 3.00,
        'soda': 1.50,
        'water': 1.00, }

cart = []
total = 0

print('------- MENU -------')
for key, value in menu.items():
    print(f'{key:10}: ${value:.2f}')
print('---------------------')

while True:
    food = input('Select an item (q to quit): ')
    if food.lower() == 'q':
        break
    elif food.lower() in menu.keys():
        cart.append(food.lower())
        total += menu[food.lower()]
    else:
        print('This food is not on the menu.')

print('------- RECEIPT -------')
for item in cart:
    print(f'{item:10}: ${menu[item]:.2f}')
print('------------------------')
print(f'Total: ${total:.2f}')
print('------------------------')

print('What is the payment method?\n1. Cash\n2. Credit Card\n3. Debit Card\n4. Pix')
payment_method = input('Choose an option: ')
if payment_method == '1':
    print('Thank you for your payment.')
elif payment_method == '2':
    print('Thank you for your payment.')
elif payment_method == '3':
    print('Thank you for your payment.')
elif payment_method == '4':
    print('Thank you for your payment.')
else:
    print('Invalid payment method.')
