import numpy as np

cpu_speed = 3.07/486. #(Gflops/millisec)

class App(object):



    def __init__(self, name = 'app', acc_min = 0, latency_max = 0, priority = 0):
        self.name = name

        self.acc_min = acc_min
        self.latency_max = latency_max
        #print('creating {}, acc_min:{:.2f}, latency_min:{:.2f}'.format(name, acc_min, latency_max))
        self.priority = priority
        self.model = None
        self.load_model_time = 0
        self.infer_remain_flops = 0
        self.ellapse = 0
        self.infer_times = 0
        self.infer_accs = []
        self.last_time = 0
        self.ellapse_times = []
        self.sim_model = None


    def load_model(self, model):

          if self.model is None or self.model.name != model.name:
                self.model = model
                self.load_model_time = np.random.normal(model.load_time, 10)
                self.infer_remain_flops = np.random.normal(self.model.Gflops, 0.05)

    ## load sim model for computing cost
    def sim_load_model(self, model):
        self.sim_model = model

    def compute_cost(self, running_apps, alpha=1.0, beta=0.001):
        if self.sim_model is None:
            return 10000000

        acc_cost = self.acc_min -  self.sim_model.acc
        latency_cost = max( (self.sim_model.Gflops - self.latency_max ) /cpu_speed , 0)

        if self.model is None or self.model.name != self.sim_model.name:
            load_cost = self.sim_model.load_time
        else:
            load_cost = 0


        return acc_cost + alpha * latency_cost + beta * load_cost

    def run_model(self, running_apps):
        #acc, flops(latency)
        #allocated_flops = self.get_Gflops() * cpu_speed / np.sum([app.get_Gflops() for app in running_apps])
        allocated_flops = cpu_speed / len(running_apps)
        #allocated_flops = cpu_speed
        new_remain_flops = self.infer_remain_flops - allocated_flops

        #load model
        if self.load_model_time > 0:
            self.load_model_time -= 1

        ## finish loading model, inference
        else:
            # inference finished
            if new_remain_flops <= 0:
                self.ellapse_times.append(self.ellapse - self.last_time)
                self.last_time = self.ellapse

                self.infer_times = self.infer_times + 1
                self.infer_accs.append(np.random.normal(self.model.acc, 0.05))
                self.infer_remain_flops = self.model.Gflops + new_remain_flops
            else:
                self.infer_remain_flops = new_remain_flops


        self.ellapse = self.ellapse + 1

    def print_sum(self):
        print("Run for {} times, mean acc {:.2f}/{:.2f}, average lag:{:.2f}/{:.2f}".format(
            self.infer_times, np.mean(self.infer_accs),self.acc_min, np.mean(self.ellapse_times), self.latency_max/cpu_speed))

    def get_mem_cost(self):
        return 0 if self.model is None else self.model.size

    def get_Gflops(self):
        return 0 if self.model is None else self.model.Gflops

class Model(object):

    def __init__(self, arch, name, acc, Gflops, load_time, infer_time, size):

        self.name = name
        self.arch = arch
        self.acc = acc
        self.Gflops = Gflops
        self.load_time = load_time
        self.infer_time = infer_time
        self.size = size


    @classmethod
    def init_from_list(cls, arch, config):
        return cls(arch, config[0],config[3],config[1],config[2],config[4],config[5])
