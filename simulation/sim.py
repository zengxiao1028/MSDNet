import numpy as np
import copy
from simulation.model import App, Model, cpu_speed
import itertools
import time
from collections import defaultdict
import multiprocessing
PRINT_COST=True
REVERSE_SEARCH=False

fair_allocation = False
minTotalCost = False



S_max = 100
costs = []
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

#name GFlops, load time, acc, inference time, model size
resnet50_cifar10_configs = [
                   #('100', 5.32, 240, ),
                   #('90', 4, 144),
                   ('cifar10 80', 3.07, 103, 0.906, 301, 10.606),
                   #('70', 2.33, 65),
                   #('60', 2.03, 42),
                   ('cifar10 50', 1.79, 32, 0.898, 193, 3.12),
                   #('40', 1.05, 18),
                   #('30', 0.81, 12, 0),
                   ('cifar10 20', 0.74, 6, 0.8783, 97, 0.53),
                   #('10', 0.66, 4, 0),
                   #('0', 0.6, 4, 0),
                   #('b0', 0.36, 3, 0),
                   ('cifar10 b10', 0.31, 2, 0.8204, 65, 0.149),
                   ('cifar10 b20', 0.27, 2, 0.808, 55, 0.117)
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

resnet50_cifar10_Models = [Model.init_from_list('resnet', config) for config in resnet50_cifar10_configs]

resnet50_imagenet100_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet100_configs]

vgg512_cifar10_Models = [Model.init_from_list('vgg512', config) for config in VGG512_cifar10_configs]

vgg512_GTSRB_Models = [Model.init_from_list('vgg512', config) for config in VGG512_GTSRB_configs]

model_types = [(resnet50_imagenet50_Models,  (1e-3,1e-3), 'imagenet50 resnet50'  ),
               (resnet50_cifar10_Models, (1e-3,1e-3), 'cifar10 resnet50'  ),
               (resnet50_imagenet100_Models, (1e-3,1e-3), 'imagenet100 resnet50'  ),
               (vgg512_cifar10_Models,        (1e-3,1e-3), 'cifar10 vgg512'  ),
               (vgg512_GTSRB_Models,          (1e-4,1e-4), 'GTSRB vgg512'  )
]

results_dict={each[2]:[] for each in model_types}

cpu_allocations = [ x/100. for x in range(10, 100, 10)]

def compute_scheme_cost(cpu_scheme, models_schemes, running_apps):
    profiles = []
    for model_scheme in models_schemes:
        # try every possible cpu schemes
            if minTotalCost:
                cost = compute_sum_cost(running_apps, model_scheme, cpu_scheme)
            else:
                cost = compute_max_cost(running_apps, model_scheme, cpu_scheme)
            profiles.append((model_scheme, cpu_scheme, cost))


    profiles = sorted(profiles, key=lambda profile: profile[-1],reverse=REVERSE_SEARCH)


    return profiles[0]

def optimize(running_apps):

    model_product = itertools.product(*[app.can_models for app in running_apps])
    models_schemes = [each for each in model_product if compute_sum_mem(each) <= S_max ]

    cpu_product = itertools.product(cpu_allocations, repeat = len(running_apps)-1) #the last one is determined by preceding ones

    if fair_allocation:
        cpu_schemes = ([1. / len(running_apps) for x in running_apps],)
    else:
        cpu_schemes = [cpu_scheme + (1-np.sum(cpu_scheme),) for cpu_scheme in cpu_product if np.sum(cpu_scheme) < 1.0 ]

    schemes = []
    for cpu_scheme in cpu_schemes:
        schemes.append((cpu_scheme,models_schemes,running_apps))

    with multiprocessing.Pool(processes=12) as pool:
        results = pool.starmap(compute_scheme_cost, schemes)

    for result in results:
        compute_sum_cost(running_apps,result[0],result[1],print_cost=False)


    ## brutal search for the optimal solution ##
    sorted_results = sorted(results, key=lambda profile: profile[-1],reverse=REVERSE_SEARCH)
    best_profile = sorted_results[0]
    compute_sum_cost(running_apps, best_profile[0], best_profile[1], print_cost=PRINT_COST)



    if best_profile is None:
        print('No optimal solution found')
    else:
        ### load (switch) best profile ###
        cost =  compute_sum_cost(running_apps, best_profile[0], best_profile[1])
        costs.append(cost)
        for idx, app in enumerate(running_apps):
            app.load_model(best_profile[0][idx])
            app.cpu = best_profile[1][idx]

def main():
    begin_time = time.time()
    np.random.seed(1023)
    optimize_now = False
    running_apps = []
    for i in range(len(model_types)):
        app_model_type = model_types[i]
        app = App(app_model_type[2], app_model_type[0], *app_model_type[1])
        running_apps.append(app)


    for t in range(int(1e5)):

        #random add apps
        if np.random.uniform()>0.99 and len(running_apps) < 6:
            app_model_type =  model_types[t%len(model_types)]
            app = App(app_model_type[2], app_model_type[0], *app_model_type[1])
            running_apps.append(app)
            optimize_now = True

        # random delete apps
        if np.random.uniform()>0.999 and len(running_apps) > 2:
            remove_index = 0

            if running_apps[remove_index].infer_times > 1:
                app = running_apps.pop(remove_index)

                app.print_sum()
                results_dict[app.name].append(app)
                optimize_now = True


        if t == 0 or optimize_now:

            optimize(running_apps)
            optimize_now = False

        if t % 1000 == 0:
            print(t,len(running_apps), [app.model.name for app in running_apps])

        for idx, app in enumerate(running_apps):

            app.run_model()

    infers = 0
    on_time_infers = 0
    for k in sorted(results_dict.keys()):
        v = stat_apps(results_dict[k])
        on_time_infers += v[0]
        infers += v[1]

    print('on_time_infers',on_time_infers/infers)
    print('cost', np.sum(costs))
    print('Used time:{:f}'.format( time.time() - begin_time))


def stat_apps(finished_apps):
    delta_acc_list = []
    delta_latency_list = []
    latency_list = []
    sum_nb_infers = 0
    sum_nb_switches = 0
    for app in finished_apps:
        if app.infer_times == 0:
            #print('App exit before inference finished.')
            pass
        else:
            delta_acc_list.append(np.array(app.infer_accs) - app.acc_min),
            delta_latency_list.append(np.array(app.ellapse_times) - app.latency_max)
            latency_list = latency_list + app.ellapse_times

            sum_nb_infers += app.nb_infers
            sum_nb_switches += app.nb_switches

    one_by_one_fps = 1/np.array(latency_list)
    mean_fps = np.sum(latency_list) / sum_nb_infers
    delta_accuracies = np.hstack(delta_acc_list).flatten()
    delta_latencies = np.hstack(delta_latency_list).flatten()
    assert len(delta_latencies) == sum_nb_infers
    on_time_inferences = delta_latencies <= 0
    print(finished_apps[0].name)
    print('Delta acc: {:.2f}, '
          'on time rate {:.2f}, '
          'average_latency:{:.2f}, '
          'ontime_inferences:{}, '
          'infer_times:{:d}, '
          'load_times:{:d}, '
          'mean_fps:{:.2f}, '
          'one_by_one_fps:{:.2f}'.format(np.mean(delta_accuracies),
                                  np.sum(on_time_inferences.astype(int)) / len(delta_latencies),
                                  np.mean(delta_latencies),
                                   np.sum( on_time_inferences.astype(int)),
                                  np.sum(sum_nb_infers),
                                  np.sum(sum_nb_switches),
                                     one_by_one_fps,
                                         mean_fps))

    print('{:.3f},{:.3f},{:.3f},{},{:d},{:d}'.format(
            np.mean(delta_accuracies),
            np.sum(on_time_inferences.astype(
                int)) / len(delta_latencies),
            np.mean(delta_latencies),
            np.sum(on_time_inferences.astype(int)),
            np.sum(sum_nb_infers),
            np.sum(sum_nb_switches),
            one_by_one_fps,
            mean_fps))
    return ( np.sum(on_time_inferences.astype(int)), len(delta_latencies) )


def compute_sum_cost(running_apps, model_scheme, cpu_scheme, print_cost=False):
    return  np.sum([app.compute_cost(model_scheme[idx],cpu_scheme[idx],print_cost) for idx,app in enumerate(running_apps)])

def compute_max_cost(running_apps, model_scheme, cpu_scheme, print_cost=False):
    return  np.max([app.compute_cost(model_scheme[idx],cpu_scheme[idx],print_cost) for idx,app in enumerate(running_apps)])

def compute_sum_mem(models):
    return  np.sum([model.size for model in models])

if __name__ == '__main__':
    main()