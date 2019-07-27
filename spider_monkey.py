import random
import numpy as np

def chunk(l, n):
    return np.array_split(np.array(l),n)
class SMA:

    def __init__(self,pop_size, max_group_size,local_leader_limit, global_leader_limit,
                 pr, fitness_func, dir_min_max, conditional_func, minimize=False):
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
        self.max_group_size = max_group_size
        self.conditional_func = conditional_func
        self.iter_count = 0
        self.minimize = minimize
        self.min_max_f = min if self.minimize else max

    @staticmethod
    def clip(value,min_max):
        if value < min_max[0]:
            return min_max[0]

        if value > min_max[1]:
            return min_max[1]

        return value

    def fitness_cmp(self,new,old):
        if not self.minimize:
            return new  > old
        else:
            return new < old

    def global_learning(self):
        best = self.min_max_f(self.population,key = lambda sm: sm.fitness)

        if self.global_leader == None:
            self.global_leader = best
            return

        if self.fitness_cmp(best.fitness,self.global_leader.fitness):
            self.global_leader.pos = best.pos[:]
            self.global_leader.fitness = best.fitness
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


    def local_leader_phase(self):
        for gp in self.groups:
            gp.members_pos_update()

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

                new_pos[i] = self.clip(new_pos[i],self.dir_min_max[i])

                new_pos_fitness = self.fitness_func(new_pos)
                if self.fitness_cmp(new_pos_fitness,sm.fitness):
                    sm.pos = new_pos[:]
                    sm.fitness = new_pos_fitness

    def global_leader_decision(self):

        if self.global_leader_count > self.global_leader_limit:
            self.global_leader_count = 0
            size = len(self.groups)
            self.groups = []
            if size < self.max_group_size:

                qt = size + 1
                for p in chunk(self.population,qt):
                    gp = SMG(self)
                    self.groups.append(gp)
                    for sm in p:
                        gp.add(sm)
                    gp.local_learning()

            else:
                gp = SMG(self)
                self.groups.append(gp)
                for p in self.population:
                    gp.add(p)
                gp.local_learning()

    def local_leader_decision(self):
        for gp in self.groups:
            gp.local_leader_decision()

    def run(self):
        self.init_pop()

        self.local_learning()
        self.global_learning()

        while True:

            self.local_leader_phase()
            self.global_leader_phase()
            self.local_learning()
            self.global_learning()
            self.local_leader_decision()
            self.global_leader_decision()

            self.iter_count += 1
            if not self.conditional_func(self.iter_count,self.global_leader.pos,self.global_leader.fitness,self):
                return






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
        best = self.sma.min_max_f(self.members,key = lambda sm: sm.fitness)
        if self.local_leader == None:
            self.local_leader = best
            return

        if self.sma.fitness_cmp(best.fitness,self.local_leader.fitness):
            self.local_leader.pos = best.pos[:]
            self.local_leader.fitness = best.fitness
        else:
            self.local_leader_count += 1

    def members_pos_update(self):
        if len(self.members) == 1:
            return
        for sm in self.members:
            new_pos = [0] * len(self.sma.dir_min_max)
            while True:
                other = random.choice(self.members)
                if other != sm:
                    break

            for i,min_max in enumerate(self.sma.dir_min_max):
                if random.uniform(0,1) >= self.sma.pr:
                    new_pos[i] = sm.pos[i] + random.uniform(0,1) * (self.local_leader.pos[i] - sm.pos[i]) \
                             + random.uniform(-1,1) * (other.pos[i] - sm.pos[i])
                    new_pos[i] = self.sma.clip(new_pos[i],min_max)

                else:
                    new_pos[i] = sm.pos[i]

            new_pos_fitness = self.sma.fitness_func(new_pos)
            if self.sma.fitness_cmp(new_pos_fitness,sm.fitness):
                sm.pos = new_pos[:]
                sm.fitness = new_pos_fitness

    def local_leader_decision(self):
        if self.local_leader_count > self.sma.local_leader_limit:
            self.local_leader_count = 0
            for sm in self.members:
                for i,min_max in enumerate(self.sma.dir_min_max):
                    min,max = min_max
                    if random.uniform(0,1) >= self.sma.pr:
                        sm.pos[i] = SM.calc_rand_pos(min,max)
                    else:
                        sm.pos[i] = sm.pos[i] + random.uniform(0,1) * (self.sma.global_leader.pos[i] - sm.pos[i]) \
                            + random.uniform(0,1) * (sm.pos[i] - self.local_leader.pos[i])
                        sm.pos[i] = self.sma.clip(sm.pos[i],min_max)
                sm.calc_fitness()




    def calc_probs(self):
        max_fitness = max(self.members, key=lambda sm: sm.fitness).fitness
        for sm in self.members:
            sm.prob = 0.9 * (sm.fitness / max_fitness) + 0.1

    def print_leader(self):
        print(self.local_leader)


class SM:
    def __init__(self,group=None):
        self.group = group
        self.fitness = None
        self.prob = 0
        self.rand_pos()

    def rand_pos(self):
        self.pos = []
        for min,max in self.group.sma.dir_min_max:
            pos = self.calc_rand_pos(min,max)
            self.pos.append(pos)

    @staticmethod
    def calc_rand_pos(min,max):
        return min + random.uniform(0, 1) * (max - min)

    def calc_fitness(self):
        self.fitness = self.group.sma.fitness_func(self.pos)

    def __str__(self):
        return "{} -> {}".format(self.pos, self.fitness)



