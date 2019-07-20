import random

class SMA:

    def __init__(self,pop_size, local_leader_limit, global_leader_limit,
                 pr, fitness_func, dir_min_max):
        self.groups = []
        self.global_leader_count = 0
        self.population = []
        self.pop_size = pop_size
        self.global_leader = None
        self.local_leader_limit = local_leader_limit
        self.global_leader_limit = global_leader_limit
        self.pr = pr
        self.fitness_func = fitness_func
        self.dir_min_max = dir_min_max


    def global_learning(self):
        best = max(self.population,key = lambda sm: sm.fitness)

        if self.global_leader == None:
            self.global_leader = best
            return

        if  best.fitness > self.global_leader.fitness:
            self.global_leader.pos = best.pos
        else:
            self.global_leader_count += 1


    def init_pop(self):
        gp = SMG(self)
        self.groups.append(gp)
        for _ in range(self.pop_size):
            sm = SM(gp)
            gp.add(sm)
            sm.calc_fitness()
            self.population.append(sm)


    def local_learning(self):
        for gp in self.groups:
            gp.local_learning()

    def global_leader_phase(self):
        for gp in self.groups:
            gp.calc_probs()

        for sm in self.population:
            if random.random() < sm.prob:
                new_pos = sm.pos[:]
                while True:
                    other = random.choice(self.population)
                    if other != sm:
                        break
                i = random.choice(range(len(self.dir_min_max)))
                new_pos[i] = sm.pos[i] + random.uniform(0,1) * (self.global_leader.pos[i] - sm.pos[i]) \
                             + random.uniform(-1,1) * (other.pos[i] - sm.pos[i])

                new_pos_fitness = self.fitness_func(new_pos)
                if new_pos_fitness > sm.fitness:
                    sm.pos = new_pos


class SMG:
    def __init__(self,sma):
        self.sma = sma
        self.local_leader = None
        self.members = []
        self.local_leader_count = 0

    def add(self,sm):
        sm.group = self
        self.members.append(sm)

    def local_learning(self):
        best = max(self.members,key = lambda sm: sm.fitness)
        if self.local_leader == None:
            self.local_leader = best
            return

        if best.fitness > self.local_leader.fitness:
            self.local_leader.pos = best.pos
        else:
            self.local_leader_count += 1

    def members_pos_update(self):
        for sm in self.members:
            new_pos = [0] * len(self.sma.dir_min_max)
            while True:
                other = random.choice(self.members)
                if other != sm:
                    break

            for i in range(len(self.sma.dir_min_max)):
                if random.uniform(0,1) >= self.sma.pr:
                    new_pos[i] = sm.pos[i] + random.uniform(0,1) * (self.local_leader.pos[i] - sm.pos[i]) \
                             + random.uniform(-1,1)  * (other.pos[i] - sm.pos[i])
                else:
                    new_pos[i] = sm.pos[i]

            new_pos_fitness = self.sma.fitness_func(new_pos)
            if new_pos_fitness > sm.fitness:
                sm.pos = new_pos

    def calc_probs(self):
        max_fitness = max(self.members, key=lambda sm: sm.fitness).fitness
        for sm in self.members:
            sm.prob = 0.9 * (sm.fitness / max_fitness) + 0.1


class SM:
    def __init__(self,group):
        self.group = group
        self.fitness = None
        self.prob = 0
        self.rand_pos()

    def rand_pos(self):
        self.pos = []
        for min,max in self.group.sma.dir_min_max:
            pos = min + random.uniform(0,1) * (max  - min)
            self.pos.append(pos)


    def calc_fitness(self):
        self.fitness = self.group.sma.fitness_func(self.pos)




