import numpy as np
import copy
from simulation.model import App, Model, cpu_speed
import itertools
from collections import defaultdict


S_max = 25
                    #name, GFlops, load time, acc, inference time, model size
resnet50_imagenet50_configs = [
                   #('100', 5.32, 240, ),
                   #('90', 4, 144),
                   ('imagenet50 80', 3.07, 103, 0.8935, 301, 10.606),
                   #('70', 2.33, 65),
                   #('60', 2.03, 42),
                   ('imagenet50 50', 1.79, 32, 0.8661, 193, 3.12),
                   #('40', 1.05, 18),
                   #('30', 0.81, 12, 0),
                   ('imagenet50 20', 0.74, 6, 0.801, 97, 0.53),
                   #('10', 0.66, 4, 0),
                   #('0', 0.6, 4, 0),
                   #('b0', 0.36, 3, 0),
                   ('imagenet50 b10', 0.31, 2, 0.6363, 65, 0.149),
                   ('imagenet50 b20', 0.27, 2, 0.625, 55, 0.117)
                   ]

            #name, GFlops, load_time, acc, inference_time, model_size
resnet50_imagenet100_configs = [
                   ('imagenet100 100', 5.32, 240,    0.9008,    502,    23.793),
                   ('imagenet100 90',  4,    144,    0.8909,    378,    15.61),
                   ('imagenet100 80', 3.07,  103,    0.879 ,    301,    10.606),
                   ('imagenet100 70', 2.33,  65,     0.8622,    255,    6.467 ),
                   ('imagenet100 60', 2.03,  42,     0.7192,    217,    4.407 ),
                   ]

                    # name,     GFlops, load_time, acc, inference_time, model_size
VGG512_cifar10_configs = [
                 ('cifar10 VGG16-E40p', 7.34,     96,    0.9074,    605,    9.636),
                 ('cifar10 VGG16-E30p',  2.79,    48,    0.8710,    318,    4.554),
                 ('cifar10 VGG16-E25p', 2.14,     33,    0.8651 ,    226,    3.216),
                 ('cifar10 VGG16-E12p', 1.51,     20,     0.8510,    169,    2.06),
                 ('cifar10 VGG16-E05p', 0.81,     10,     0.7547,    112,    0.982),
                 ]


VGG512_GTSRB_configs = [
                   ('GTSRB VGG16-E25p', 2.14,     33,     0.9853,    226,    3.216),
                   ('GTSRB VGG16-E05p',  0.81,    10,    0.9777,    112,    0.982),
                   ('GTSRB VGG16-E01p', 0.11,     3,    0.9591,    52,    0.343),
                   ('GTSRB VGG16-E00p', 0.06,     3,     0.9520,    48,    0.233),
                   ('GTSRB VGG16-E005p', 0.04,     3,     0.9443,    42,    0.18),
                   ]


resnet50_imagenet50_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet50_configs]

resnet50_imagenet100_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet100_configs]

vgg512_cifar10_Models = [Model.init_from_list('vgg512', config) for config in VGG512_cifar10_configs]

vgg512_GTSRB_Models = [Model.init_from_list('vgg512', config) for config in VGG512_GTSRB_configs]

model_types = [(resnet50_imagenet50_Models,  (0.05,0.001)  ),
               (resnet50_imagenet100_Models, (0.05,0.001)  ),
               (vgg512_cifar10_Models,        (0.05,0.001)  ),
               (vgg512_GTSRB_Models,          (0.01,0.001)  )
]

cpu_allocations = [ x/10. for x in range(1, 10)]

def optimize(running_apps):

    model_product = itertools.product(*[app.can_models for app in running_apps])
    models_schemes = [each for each in model_product]

    cpu_product = itertools.product(cpu_allocations,repeat=len(running_apps)-1) #the last one is determined by preceding ones
    cpu_schemes = [cpu_scheme + (1-np.sum(cpu_scheme),) for cpu_scheme in cpu_product if np.sum(cpu_scheme) < 1.0 ]


    ## brutal search for the optimal solution ##
    profiles = []
    for i in range(len(models_schemes)):
        for idx, app in enumerate(running_apps):
            app.sim_load_model(models_schemes[i][idx])

        #try every possible cpu schemes
        for j in range(len(cpu_schemes)):
            for idx, app in enumerate(running_apps):
                app.sim_cpu = cpu_schemes[j][idx]

            cost = compute_sum_cost(running_apps)
            profiles.append((models_schemes[i], cpu_schemes[j], cost))
    profiles = sorted(profiles, key=lambda profile: profile[-1])


    ### Smax requirement ####
    best_profile = None
    for profile in profiles:
        sum_mem_cost = compute_sum_mem(profile[0])
        if sum_mem_cost <= S_max:
            best_profile = profile
            break

    if best_profile is None:
        print('No optimal solution found')
    else:
        ### load (switch) best profile ###
        for idx, app in enumerate(running_apps):
            app.load_model(best_profile[0][idx])
            app.cpu = best_profile[1][idx]

def main():
    optimize_now = False
    running_apps = []
    for i in range(4):
        app_model_type = model_types[i]
        app = App('app'+str(i+1), app_model_type[0], *app_model_type[1])
        running_apps.append(app)


    finished_apps = []
    np.random.seed(1024)
    for t in range(int(1e5)):

        #random add apps
        if np.random.uniform()>0.999 and len(running_apps) < 6:
            app_model_type =  model_types[t%len(model_types)]
            app = App('app' + str(i + 1), app_model_type[0], *app_model_type[1])
            running_apps.append(app)
            optimize_now = True

        # random delete apps
        if np.random.uniform()>0.9995 and len(running_apps) > 2:
            remove_index = 0

            #if running_apps[remove_index].infer_times > 5:
            app = running_apps.pop(remove_index)

            app.print_sum()
            finished_apps.append(app)
            optimize_now = True


        if t % 1000==0 or optimize_now:

            optimize(running_apps)

            optimize_now = False

        if t%1000 == 0:
            print(t,len(running_apps), [app.model.name for app in running_apps])

        for idx, app in enumerate(running_apps):

            app.run_model(running_apps)

    delta_acc_list = []
    delta_latency_list = []
    for app in finished_apps:
        if app.infer_times == 0:
            print('App exit before inference finished.')
        else:
            delta_acc_list.append(np.mean(app.infer_accs) - app.acc_min),
            delta_latency_list.append(np.array(app.ellapse_times) - app.latency_max/cpu_speed)

    inferences = np.hstack(delta_latency_list).flatten()
    on_time_inferences = inferences<= 0
    print('Delta acc: {:.2f}, on time rate {:.2f}, average_latency:{:.2f}'.format(np.mean(delta_acc_list),
                                                           np.sum(on_time_inferences.astype(int))/len(inferences),
                                                            np.mean(inferences)) )

def compute_sum_cost(running_apps):
    return  np.sum([app.compute_cost(running_apps) for app in running_apps])

def compute_sum_mem(models):
    return  np.sum([model.size for model in models])

if __name__ == '__main__':
    main()