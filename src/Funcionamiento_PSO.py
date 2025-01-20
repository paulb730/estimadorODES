import pygmo as pg

import numpy as np


class Parameter:
    def __init__(self, x_min, x_max):
        """

        :param x_min: Umbral mínimo de búsqueda
        :param x_max: Umbral máximo de búsqueda
        Constructor class Parameter
        """

        self.x_min = x_min
        self.x_max = x_max
    def get_param(self):
        """

        :return: Vector que limita el espacio de búsqueda
        """
        return (self.x_min,self.x_max)

class Modelos_objective:
    def __init__(self ,get_ymodel,get_y_data,x_min,x_max,case:int):
        """
        Constructor  class.
        """
        self.param_1=Parameter(x_min,x_max)
        self.y_modelo=get_ymodel
        self.y_data=get_y_data
        self.Vb=self.param_1.get_param()
        self.case=case

    def get_bounds(self):
        """
        Defines the boundaries of the search space.
        """
        return (self.Vb[0], self.Vb[1])

    def fitness(self,thetha:list):
        if self.case=='1':
            function_value = np.sum((self.y_modelo(thetha) - self.y_data()) ** 2)
        if self.case=='2':
            function_value =(1/2)*np.log10((np.sum(((self.y_modelo(thetha)[:, 2] - self.y_data()[0]) ** 2)) + np.sum((self.y_modelo(thetha)[:, 3] - self.y_data()[1]) ** 2)))
        if self.case=='3':
            function_value=(1/3)*(np.sum((self.y_modelo(thetha)[:,0] - self.y_data()[0]) ** 2)+np.sum((self.y_modelo(thetha)[:,1] - self.y_data()[1]) ** 2)+np.sum((self.y_modelo(thetha)[:,2] - self.y_data()[2]) ** 2))
        if self.case=='4':
            function_value= (1/len(self.y_data()))*np.sum((np.log10(self.y_modelo(thetha)[:,2]) - self.y_data()) ** 2)
        if self.case=='5':
            function_value= np.sum(np.power((self.y_modelo(thetha)[:,1] - self.y_data()[1]),2))


        return [function_value]






class Modelos_MultiObjective:
    def __init__(self, get_ymodel, get_y_data, x_min, x_max, case: int):
        """
        Constructor  class.
        """
        self.param_1 = Parameter(x_min, x_max)
        self.y_modelo = get_ymodel
        self.y_data = get_y_data
        self.Vb = self.param_1.get_param()
        self.case = case



        # Return number of objectives

    def fitness(self,thetha:list):
        if self.case=='2':
            f1= (1 / self.y_data()[0] ** 2)* np.sum(self.y_modelo(thetha)[:, 2] - self.y_data()[0]) ** 2
            f2=(1 / self.y_data()[1] ** 2)* np.sum(self.y_modelo(thetha)[:, 3] - self.y_data()[1]) ** 2

        return[f1,f2]
        # Define objectives

    def get_nobj (self):
        return 2


        # Return bounds of decision variables

    def get_bounds(self):
        return (self.Vb[0], self.Vb[1])





#INICIALIZACION DE ALGORITMOS PARA CUALQUIER METODO

def init_algorithm_pso(ymodel, ymeas, lowerbounds, upperbounds, gen: int, pop, c1, c2, w, verbosity: int,case):
    thethaVector = []
    gVector = []
    modelo_obj = Modelos_objective(ymodel, ymeas, lowerbounds, upperbounds,case)
    prob = pg.problem(modelo_obj)
    PSO=pg.pso(gen=gen, omega=w, eta1=c1, eta2=c2)
    algo = pg.algorithm(PSO)
    algo.set_verbosity(verbosity)
    swarm = pg.population(prob, size=pop)
    swarm = algo.evolve(swarm)
    thethaVector.append(swarm.get_x()[swarm.best_idx()])
    gVector.append(swarm.get_f()[swarm.best_idx()])


    return swarm.champion_x,swarm.champion_f,swarm


def init_nspso(ymodel, ymeas, lowerbounds, upperbounds,gen:int,pop:int,verbosity,case):
    thethaVector = []
    gVector = []
    modelo_obj = Modelos_objective(ymodel, ymeas, lowerbounds, upperbounds, case)
    prob = pg.problem(modelo_obj)

    swarm = pg.population(prob, size=pop)
    MC= pg.gwo(gen=gen)
    print(MC)
    algo = pg.algorithm(MC)
    algo.set_verbosity(verbosity)

    swarm = algo.evolve(swarm)
    thethaVector.append(swarm.get_x()[swarm.best_idx()])
    gVector.append(swarm.get_f()[swarm.best_idx()])

    return swarm.champion_x,swarm.champion_f,swarm

def init_bee_colony(ymodel, ymeas, lowerbounds, upperbounds,gen:int,pop:int,limit:int,verbosity,case):
    thethaVector = []
    gVector = []
    modelo_obj = Modelos_objective(ymodel, ymeas, lowerbounds, upperbounds, case)
    prob = pg.problem(modelo_obj)
    BC = pg.bee_colony(gen=gen,limit=limit)
    algo = pg.algorithm(BC)
    algo.set_verbosity(verbosity)
    swarm = pg.population(prob, size=pop)
    swarm = algo.evolve(swarm)
    thethaVector.append(swarm.get_x()[swarm.best_idx()])
    gVector.append(swarm.get_f()[swarm.best_idx()])
    return swarm.champion_x, swarm.champion_f, swarm


def init_diif_e(ymodel, ymeas, lowerbounds, upperbounds,gen:int,pop:int,f,cr,case,evolutions:int):
    thethaVector = []
    gVector = []
    modelo_obj=Modelos_objective(ymodel, ymeas, lowerbounds, upperbounds,case)
    prob = pg.problem(modelo_obj)
    DE= pg.core.de(gen=gen, F=f, CR=cr, variant=2, ftol=1e-06, xtol=1e-06)

    algo=pg.algorithm(DE)

    swarm = pg.population(prob, size=pop,seed=171015)

    for i in range(evolutions):
        swarm = algo.evolve(swarm)
        thethaVector.append(swarm.get_x()[swarm.best_idx()])
        gVector.append(swarm.get_f()[swarm.best_idx()])

    return swarm.champion_x,swarm.champion_f,swarm

def init_genetic_algorithm(ymodel, ymeas, lowerbounds, upperbounds,gen:int,pop:int,m,cr,case):
    thethaVector = []
    gVector = []
    modelo_obj = Modelos_objective(ymodel, ymeas, lowerbounds, upperbounds, case)
    prob = pg.problem(modelo_obj)
    GA = pg.core.sga(gen=gen, cr=cr, m=m)
    algo = pg.algorithm(GA)
    swarm = pg.population(prob, size=pop)
    swarm = algo.evolve(swarm)
    thethaVector.append(swarm.get_x()[swarm.best_idx()])
    gVector.append(swarm.get_f()[swarm.best_idx()])
    return thethaVector,gVector,swarm



# optimization_enzimatic=init_algorithm_pso(ymodel_enzi,get_y_data_1,[0.1, 0.1, 0.1, 0.1],[1, 3, 1, 0.3],100,100,2.05,2.05,0.7298,1)


#print("Eureka",optimization_enzimatic)
"""
#instancia de modelo enzimatico
modelo_enzi_obj=Modelos_objective(ymodel_enzi,get_y_data_1,[0.1, 0.1, 0.1, 0.1],[1, 3, 1, 0.3])
#variable cambio
udp=modelo_enzi_obj
#inicializar problema de optimizacion
prob=pg.problem(udp)
#elegir algoritmo PSO inidicaciones
algo=algorithm(pg.pso(gen=100))
#informacion backend
algo.set_verbosity(1)
#numero de particulas
pop_size=100
#poblacion
pop = pg.population(prob, size=pop_size,current_seed = 171015)
#tabla de parametros
individuals_list = []
#valor de funcion residual /objetivo
fitness_list = []
#iniciar optimizacion
pop = algo.evolve(pop)
#vector de parametros
individuals_list.append(pop.get_x()[pop.best_idx()])
#vector de residuales
fitness_list.append(pop.get_f()[pop.best_idx()])

print("parametros",individuals_list)


"""




"""
udp_1 = HimmelblauOptimization(-5.0, 5.0, -5.0, 5.0)
prob_1 = pg.problem(udp)
algo = algorithm(pg.pso(gen = 500))
algo.set_verbosity(50)
# Set population size
pop_size = 1000
# Set seed
current_seed = 171015
# Create population
pop = pg.population(prob, size=pop_size, seed=current_seed)
# Initialize empty containers
individuals_list = []
fitness_list = []
# Evolve population multiple times
pop = algo.evolve(pop)
individuals_list.append(pop.get_x()[pop.best_idx()])
fitness_list.append(pop.get_f()[pop.best_idx()])

"""







