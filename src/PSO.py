from __future__ import division
import numpy as np


# Author: Paul Benavides
# Tema: Estimación de parámetros en ecuaciones diferenciales ordinarias. Aplicación a la inteligencia artificial


class PSO:
    def __init__(self, pop, generation, x_min, x_max, objective_function, c1=0.1, c2=0.1, w=1):
        """
        :param pop: número de partículas usadas en el enjambre
        :param generation: número de iteraciones a realizar
        :param x_min: valores mínimos de límitacion en el espacio de búsqueda
        :param x_max: valores máximos de limitación en el espacio de búsqueda
        :param objective_function: función a minimizar ⅀|| y - ym ||  y de las observaciones obtenidas menos las y del modelo simulado
        :param c1: coeficiente de cognitividad
        :param c2: coeficiente social
        :param w: inertia de la partícula
        """

        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.pop = pop
        self.x_min = np.array(x_min)
        self.x_max = np.array(x_max)
        self.generation = generation
        self.max_v = (self.x_max - self.x_min) * 0.05  # vector de valores máximos
        self.min_v = -(self.x_max - self.x_min) * 0.05  # vector de valores mínimos
        self.objective_function = objective_function

        # Iniciar la población de partículas

        self.particals = [Partical(self.x_min, self.x_max, self.max_v, self.min_v, self.objective_function) for i in
                          range(self.pop)]

        # Inicializar el vector de global best
        self.gbest = np.zeros(len(x_min))
        self.gbestFit = float('Inf')

        self.fitness_list = []  # Vector de los parámetros que se van ajustando a la solución

    def init_gbest(self):
        for part in self.particals:
            if part.getBestFit() < self.gbestFit:
                self.gbestFit = part.getBestFit()
                self.gbest = part.getPbest()

    def done(self):
        for i in range(self.generation):
            for part in (self.particals):
                part.update(self.w, self.c1, self.c2, self.gbest)
                if part.getBestFit() < self.gbestFit:
                    self.gbestFit = part.getBestFit()
                    self.gbest = part.getPbest()
            self.fitness_list.append(self.gbest)
        return self.fitness_list, self.gbest
class Partical:

    def __init__(self, x_min, x_max, max_v, min_v, fitness):
        """

        :param x_min:
        :param x_max:
        :param max_v:
        :param min_v:
        :param fitness: objective function
        """

        self.dim = len(x_min)  # Obtener el número de variables
        self.max_v = max_v
        self.min_v = min_v
        self.x_min = x_min
        self.x_max = x_max

        """
        Para evitar que las diferentes restricciones de 
        variables sean diferentes , toda las que se pasan
        son matrices
        """

        self.pos = np.zeros(self.dim)  # la posicion inicial de la particula
        self.pbest = np.zeros(self.dim)  # inicialización del vector  personal best
        self.initPos(x_min, x_max)

        self._v = np.zeros(self.dim)
        self.initV(min_v, max_v)  # Velocidad inicial de la particula

        self.fitness = fitness
        #Inicializar la particula en la funcion objetivo
        self.bestFitness = fitness(self.pos)

    def _updateFit(self):
        if self.fitness(self.pos) < self.bestFitness:
            self.bestFitness = self.fitness(self.pos)
            self.pbest = self.pos

    def _updatePos(self):

        self.pos = self.pos + self._v

        for i in range(self.dim):
            self.pos[i] = min(self.pos[i], self.x_max[i])
            self.pos[i] = max(self.pos[i], self.x_min[i])

    def _updateV(self, w, c1, c2, gbest):
        """

        :param w:
        :param c1:
        :param c2:
        :param gbest:
        :return:
        """
        #Vector de velocidad PSO
        self._v = w * self._v + c1 * np.random.random() * (self.pbest - self.pos) + c2 * np.random.random() * (gbest - self.pos)

        for i in range(self.dim):
            self._v[i] = min(self._v[i], self.max_v[i])
            self._v[i] = max(self._v[i], self.min_v[i])

    def initPos(self, x_min, x_max):
        for i in range(self.dim):
            self.pos[i] = np.random.uniform(x_min[i], x_max[i])
            self.pbest[i] = self.pos[i]

    def initV(self, min_v, max_v):
        for i in range(self.dim):
            self._v[i] = np.random.uniform(min_v[i], max_v[i])

    def getPbest(self):
        return self.pbest

    def getBestFit(self):
        return self.bestFitness

    def update(self, w, c1, c2, gbest):

        """

        :param w:
        :param c1:
        :param c2:
        :param gbest:
        :return:
        """
        self._updateV(w, c1, c2, gbest)
        self._updatePos()
        self._updateFit()
