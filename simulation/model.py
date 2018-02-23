import numpy as np
import math
cpu_speed = 1.0

def leaky_relu(x):
    return x if x >= 0 else 0.3*x

class App(object):


    def __init__(self, name, candidate_models,  alpha=0.05, beta=0.001, acc_min=0, lag_max=0, freeze_model=False):
        self.name = name
        self.can_models = candidate_models

        self.acc_min = np.random.normal(acc_min, acc_min*0.001)
        self.freeze_model = freeze_model
        #self.latency_max = np.random.normal(candidate_models[0].infer_time, 10)
        self.latency_max = np.random.normal(lag_max,10)
        self.nb_switches = 0
        self.alpha = alpha
        self.beta = beta

        self.model = None
        self.load_model_time = 0
        self.infer_remain_time = 0
        self.ellapse = 0
        self.nb_infers = 0
        self.infer_accs = []
        self.last_time = 0
        self.ellapse_times = []
        self.sum_load_model_time = 0
        self.sim_cpu = 0
        self.cpu = 0


    def load_model(self, model):

        if self.load_model_time > 0:
            return

        if self.model is not None and self.model.name == model.name:
            pass
        else:
                if self.model is None:
                    self.load_model_time = model.load_time
                else:
                    if self.freeze_model:
                        # go bigger:
                        if model.size > self.model.size:
                            self.load_model_time = np.abs(self.model.load_time - model.load_time)
                        else:
                            self.load_model_time = 0
                    else:
                        self.load_model_time = model.load_time

                self.model = model
                self.nb_switches += 1
                self.infer_remain_time = self.model.infer_time




    def compute_cost(self, sim_model, sim_cpu, print_cost=False):

        #acc_cost = max( self.acc_min - sim_model.acc, 0)
        acc_cost = self.acc_min - sim_model.acc
        #latency_cost = max( sim_model.infer_time /sim_cpu - self.latency_max  , 0)
        latency_cost = sim_model.infer_time / sim_cpu - self.latency_max
        #latency_cost = sim_model.infer_time * sim_cpu

        #compute load cost
        #if already load, then no cost.
        if self.model is not None and self.model.name == sim_model.name:
            load_cost = 0
        else:
            if self.model is None:
                load_cost = sim_model.load_time
            else:
                ##load model cost
                if self.freeze_model:
                    if sim_model.size > self.model.size:
                        load_cost = np.abs(sim_model.load_time - self.model.load_time)
                    else:
                        load_cost = 0
                else:
                    load_cost = sim_model.load_time

        if print_cost:
            print('acc:{:.3f}, lag:{:.3f}, load:{:.3f}'.format(acc_cost,self.alpha * latency_cost,self.beta * load_cost,
                  acc_cost + self.alpha * latency_cost + self.beta * load_cost))
        return acc_cost + self.alpha * latency_cost + self.beta * load_cost




    def run_model(self):

        #load model
        if self.load_model_time > 0:
            #just loaded
            if self.load_model_time == self.model.load_time:
                self.last_time = self.ellapse
            self.load_model_time -= 1
            self.sum_load_model_time += 1

        ## finish loading model, inference
        else:
            new_remain_time = self.infer_remain_time - self.cpu
            # inference finished
            if new_remain_time <= 0:
                self.ellapse_times.append(self.ellapse - self.last_time)
                self.last_time = self.ellapse

                #fire a inference
                self.nb_infers = self.nb_infers + 1
                self.infer_accs.append(self.model.acc)
                self.infer_remain_time = self.model.infer_time + new_remain_time
            else:
                self.infer_remain_time = new_remain_time

        self.ellapse = self.ellapse + 1

    def print_sum(self):
        print(self.name + "\t{}, Run for {} times, switch {} times, mean acc {:.2f}/{:.2f}, average lag:{:.2f}/{:.2f}".format(
            self.model.name,
            self.nb_infers, self.nb_switches, np.mean(self.infer_accs),self.acc_min, np.mean(self.ellapse_times), self.latency_max / cpu_speed))

    def print_sim_2(self):
        fps_list = 1000./np.array(self.ellapse_times)
        fps = np.mean(fps_list)
        print("Run {} times, switch {} times, sum_load_model_time{}, fps{}".
              format(self.nb_infers, self.nb_switches, self.sum_load_model_time,fps) )

    def get_mem_cost(self):
        return 0 if self.model is None else self.model.size

    # def get_Gflops(self):
    #     return 0 if self.model is None else self.model.Gflops

class Model(object):

    def __init__(self, arch, name, acc, Gflops, load_time, infer_time, size):

        self.name = name
        self.arch = arch
        self.acc = acc
        #self.Gflops = Gflops
        self.load_time = load_time
        self.infer_time = infer_time
        self.size = size


    @classmethod
    def init_from_list(cls, arch, config):
        return cls(arch, config[0],config[3],config[1],config[2],config[4],config[5])
