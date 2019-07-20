from spider_monkey import SMA


def fitness_calc(value):
    print("Calculando fitness...",value)
    return value[0]+value[1]

sma = SMA(pop_size=10, local_leader_limit=5, global_leader_limit=5,pr=0.2,
          fitness_func=fitness_calc,dir_min_max=[(0,50),(0,50)])

sma.init_pop()
sma.global_learning()
sma.local_learning()

sma.groups[0].members_pos_update()
sma.local_leader_phase()
sma.global_leader_phase()
print("Global Leader",sma.global_leader)
print("Local leader g0", sma.groups[0].local_leader)