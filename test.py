from spider_monkey import SMA, chunk
import math


def fitness_calc(value):
    x1,x2 = value

    return  math.sin(x1 + x2) + (x1 - x2)**2 - (3/2)*x1 + (5/2)*x2 +1

def cond_func(it,vars,value,sma):
    print('#{}'.format(it), vars, value)
    print()
    for sm in sma.population:
        print("   "+str(sm))
    print()
    return False if it == 100 else True

sma = SMA(pop_size=10, max_group_size=5,local_leader_limit=5, global_leader_limit=5,pr=0.8,
          fitness_func=fitness_calc,dir_min_max=[(-1.5,4),(-3,3)], conditional_func=cond_func ,minimize=True)

sma.run()

best = min(sma.population, key=lambda sm: sm.fitness)
x1,x2 = best.pos
print("x1 = {}\nx2 = {}\nf(X*) = {}".format(x1,x2,best.fitness),sma.global_leader.fitness, sma.global_leader == best)