from pyomo.environ import *

if __name__ == '__main__':
    m = ConcreteModel()

    m.T = Set(initialize=[1, 2])
    m.G = Set(initialize=['a', 'b'])

    m.P = Param(m.T, m.G, initialize=0, mutable=True)

    update = {(1, 'a'): 10, (1, 'b'): 10, (2, 'a'): 10}

    # m.P = update