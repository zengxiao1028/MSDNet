import numpy as np

cpu_speed = 1.0

class App(object):

    freeze_model = False

    def __init__(self, name, candidate_models,  alpha=0.05, beta=0.001):
        self.name = name
        self.can_models = candidate_models

        self.acc_min = np.random.normal(candidate_models[0].acc, candidate_models[0].acc*0.001)

        self.latency_max = np.random.normal(candidate_models[0].infer_time, 10)

        self.load_times = 0
        self.alpha = alpha
        self.beta = beta

        self.model = None
        self.load_model_time = 0
        self.infer_remain_time = 0
        self.ellapse = 0
        self.infer_times = 0
        self.infer_accs = []
        self.last_time = 0
        self.ellapse_times = []
        self.sim_model = None

        self.sim_cpu = 0
        self.cpu = 0


    def load_model(self, model):

        if self.model is not None and self.model.name == model.name:
            pass
        else:
                self.model = model
                self.load_times += 1
                self.infer_remain_time = self.model.infer_time

                old_load_time = 0 if self.model is None else self.model.load_time
                if self.freeze_model:
                    # go bigger:
                    if model.size > self.model.size:
                        self.load_model_time = np.abs(old_load_time - model.load_time)
                else:
                    self.load_model_time = model.load_time


    ## load sim model for computing cost
    def sim_load_model(self, model):
        self.sim_model = model


    def compute_cost(self, sim_model, sim_cpu, print_cost=False):

        acc_cost = max( self.acc_min - sim_model.acc, 0)
        latency_cost = max( sim_model.infer_time /sim_cpu - self.latency_max  , 0)
        #latency_cost = sim_model.infer_time * sim_cpu


        if self.model is not None and self.model.name == sim_model.name:
            load_cost = 0
        else:

            #old_load_time = 0 if self.model is None else self.model.load_time
            #load model cost
            #if self.freeze_model:
            #    load_cost = np.abs(sim_model.load_time - old_load_time)
            #else:
                load_cost = sim_model.load_time

        if print_cost:
            print('acc:{:.3f}, lag:{:.3f}, load:{:.3f}'.format(acc_cost,self.alpha * latency_cost,self.beta * load_cost,
                  acc_cost + self.alpha * latency_cost + self.beta * load_cost))
        return acc_cost + self.alpha * latency_cost + self.beta * load_cost

    # def compute_cost(self, running_apps):
    #     if self.sim_model is None:
    #         return 10000000
    #
    #     acc_cost = self.acc_min -  self.sim_model.acc
    #     latency_cost = max( (self.sim_model.infer_time - self.latency_max )/self.sim_cpu , 0)
    #
    #     if self.model is None or self.model.name != self.sim_model.name:
    #         load_cost = self.sim_model.load_time
    #     else:
    #         load_cost = 0

    #     return acc_cost + self.alpha * latency_cost + self.beta * load_cost

    def run_model(self):
        #acc, flops(latency)
        #allocated_consumed_time = self.get_Gflops() * cpu_speed / np.sum([app.get_Gflops() for app in running_apps])
        allocated_consumed_time = self.cpu
        #allocated_consumed_time = cpu_speed
        new_remain_time = self.infer_remain_time - allocated_consumed_time

        #load model
        if self.load_model_time > 0:
            self.load_model_time -= 1

        ## finish loading model, inference
        else:
            # inference finished
            if new_remain_time <= 0:
                self.ellapse_times.append(self.ellapse - self.last_time)
                self.last_time = self.ellapse

                #fire a inference
                self.infer_times = self.infer_times + 1
                self.infer_accs.append(self.model.acc)
                self.infer_remain_time = self.model.infer_time + new_remain_time
            else:
                self.infer_remain_time = new_remain_time

        self.ellapse = self.ellapse + 1

    def print_sum(self):
        print(self.name + "Run for {} times, mean acc {:.2f}/{:.2f}, average lag:{:.2f}/{:.2f}".format(
            self.infer_times, np.mean(self.infer_accs),self.acc_min, np.mean(self.ellapse_times), self.latency_max/cpu_speed))

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
