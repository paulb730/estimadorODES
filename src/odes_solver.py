from scipy.integrate import odeint,solve_ivp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src import PSO as pso
from scipy.optimize import minimize,least_squares

# Author: Paul Benavides
# Derechos Reservados


def fitting_model_graph(model,z, t, thetha,index=0):
    ym_tofit = odeint(model, z, t, args=tuple(thetha))
    solucion_requerida=ym_tofit[:,index]
    return solucion_requerida

def PSO_test(objective_function, pop, gen, xmin, xmax, c1, c2, w):
    """"
    :param objective_function: espacio de búsqueda de las partículas  
    :param pop: población de partículas
    :param gen: iteraciones 
    :param xmin: xmin de busqueda
    :param xmax: xmax de busqueda
    :param c1: coeficiente de coercitividad
    :param c2: coeficiente social 
    :param w: peso inercia 
    :return: optimización via PSO 
    """""
    optimizador = pso.PSO(pop, gen, xmin, xmax, objective_function, c1, c2, w)
    fitlist, best_pos = optimizador.done()
    vectthetha = []
    for pos in best_pos:

        vectthetha.append("{:e}".format(pos))

    return [fitlist, vectthetha]





