from spider_monkey import SMA, chunk
import math


def fitness_calc(value):
    x1,x2 = value

    return  math.sin(x1 + x2) + (x1 - x2)**2 - (3/2)*x1 + (5/2)*x2 +1

sma = SMA(pop_size=100, max_group_size=10,local_leader_limit=5, global_leader_limit=5,pr=0.8,
          fitness_func=fitness_calc,dir_min_max=[(-1.5,4),(-3,3)], max_iterations=10, minimize=True)

sma.run()

best = min(sma.population, key=lambda sm: sm.fitness)
x1,x2 = best.pos
print("x1 = {}\nx2 = {}\nf(X*) = {}".format(x1,x2,best.fitness),sma.global_leader.fitness)