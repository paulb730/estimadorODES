import numpy as np

from scipy.integrate import odeint
import PSO as pso


# Author: Paul Benavides
# Derechos Reservados


def objective_function(thetha, model, z, t, ymeasured):
    """
    :param thetha: vector de parámetros
    :param model: definición de modelo
    :param z: condiciones iniciales
    :param t: tiempo
    :param ymeasured: mediciones experimentales
    :param ymodelo: y del modelo
    :return: función objetivo
    """
    ymodelo = odeint(model, z, t, args=tuple(thetha))  # integración del modelo
    return np.sum(np.power((ymodelo[:, 1] - ymeasured), 2))


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
        vectthetha.append(format(pos, '.4E'))

    return [fitlist, vectthetha]


def fitting_model_graph(thetha, t, z, model):
    ymfitted = odeint(model, z, t, args=tuple(thetha))
    return ymfitted


"""
# def the model
t_data = [0, 10, 20, 40, 80, 160, 320]
exp_1 = [0, 0.192, 0.144, 0.423, 0.421, 0.407, 0.271]
exp_2 = [0, 0.140, 0.240, 0.308, 0.405, 0.464, 0.223]

vect_prom = []
A = np.zeros(2)
y2_prom = prom(exp_1, exp_2, vect_prom)  # promedio de las observaciones iniciales
df_1 = pd.DataFrame(y2_prom, columns=['PROM_EXP'])
df_1.head()
df_1.iplot()
b1 = 0
b2 = 0
A = (b1, b2)
pso_optimizer = PSO_test(objective_function, 50, 100, A, [1, 1], 1, 1, 1)

fitlist, best_pos = pso_optimizer.done()

print("best", format(best_pos[0], '.4E'), format(best_pos[1], '.4e'))

A0 = [best_pos[0], best_pos[1]]

z = [1, 0]

ymodel1 = odeint(model_kinetic_chemistry, z, t_data, args=tuple(A0))  # integración del modelo
print("objective_function", format(np.sum(np.power((ymodel1[:, 1] - y2_prom), 2)), '.4e'))

# plot results
plt.plot(t_data, ymodel1[:, 0], '-r')
plt.plot(t_data, y2_prom, 'bo')
plt.plot(t_data, ymodel1[:, 1], '-g')
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()



"""
