# This is a sample Python script.
from gekko import GEKKO

import PSO as pso


# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Author: Paul Benavides
# Derechos Reservados

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'PSO, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def modelo_prey_predator():
    m = GEKKO()
    b1 = m.Param(1)
    b2 = m.Param(1)


def PSO_test():
    pop = 300
    gen = 100
    x_min = [-10, 10]
    x_max = [10, 10]
    t_data = [0, 0.1, 0.2, 0.4, 0.8, 1]
    x_data = [2.0, 1.6, 1.2, 0.7, 0.3, 0.15]
    pso.PSO(pop, gen, x_min, x_max)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
