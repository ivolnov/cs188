from foo import Foo


"""
To run this script, type

  python buyLotsOfFruit.py

Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""

fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75,
              'limes':0.75, 'strawberries':1.00}



def main():
    o = Foo('alloha')
    print(o.getName())
    #print buyLotsOfFruits([('apples', 2.0), ('pears', 3.0), ('limes', 4.0)])

def buyLotsOfFruits(orderList):
    cost = 0
    for fruit, amount in orderList:
        cost += amount * fruitPrices[fruit]
    return 'Cost of ' + str(orderList) + ' is ' + str(cost)

#def quickSort(list):
 #   return [e for e in list if e < list[0]] + [e for e in list e >= list[0]]

if __name__ == '__main__':
    main()
