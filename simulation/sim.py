import numpy as np
from simulation.model import App, Model, cpu_speed
import itertools
import time
import multiprocessing

PRINT_COST = False
REVERSE_SEARCH = False


minTotalCost = True
s_alpha = 0.00026


# minTotalCost = False
# s_alpha = 0.00016


gen_pro = 0.99
exit_pro = 0.99



use_baseline = False
fair_allocation = True
if use_baseline:
    from simulation.baseline_model_configs import *
    freeze_model = False
else:
    from simulation.model_configs import *
    freeze_model = True


costs = []

resnet50_imagenet50_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet50_configs]
resnet50_scene_Models = [Model.init_from_list('resnet', config) for config in resnet50_scene_configs]
resnet50_imagenet100_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet100_configs]
vgg4096_cifar10_Models = [Model.init_from_list('vgg4096', config) for config in VGG4096_cifar10_configs]
vgg512_GTSRB_Models = [Model.init_from_list('vgg512', config) for config in VGG512_GTSRB_configs]
vgg2048_gender_Models = [Model.init_from_list('vgg2048', config) for config in VGG2048_gender_configs]


cpu_allocations = [x / 100. for x in range(10, 100, 10)]


def compute_scheme_cost(cpu_scheme, models_schemes, running_apps):
    profiles = []
    for model_scheme in models_schemes:
        # try every possible cpu schemes
        if minTotalCost:
            cost = compute_sum_cost(running_apps, model_scheme, cpu_scheme)
        else:
            cost = compute_max_cost(running_apps, model_scheme, cpu_scheme)
        profiles.append((model_scheme, cpu_scheme, cost))

    profiles = sorted(profiles, key=lambda profile: profile[-1], reverse=REVERSE_SEARCH)

    return profiles[0]


def optimize(running_apps, S_max, predefine_model_scheme):
    #model_product = itertools.product(*[app.can_models for app in running_apps])
    #models_schemes = [each for each in model_product if compute_sum_mem(each) <= S_max]

    #second
    models_schemes = []
    for idx,app in enumerate(running_apps):
        models_schemes.append(app.can_models[predefine_model_scheme[idx]])
    models_schemes = [models_schemes]
    if len(models_schemes) == 0:
        return
    cpu_product = itertools.product(cpu_allocations,
                                    repeat=len(running_apps) - 1)  # the last one is determined by preceding ones
    if fair_allocation:
        cpu_schemes = ([1. / len(running_apps) for x in running_apps],)
    else:
        cpu_schemes = [cpu_scheme + (1 - np.sum(cpu_scheme),) for cpu_scheme in cpu_product if np.sum(cpu_scheme) < 1.0]

    schemes = []
    for cpu_scheme in cpu_schemes:
        schemes.append((cpu_scheme, models_schemes, running_apps))

    with multiprocessing.Pool(processes=12) as pool:
        results = pool.starmap(compute_scheme_cost, schemes)

    ## brutal search for the optimal solution ##
    sorted_results = sorted(results, key=lambda profile: profile[-1], reverse=REVERSE_SEARCH)
    best_profile = sorted_results[0]


    if best_profile is None:
        print('No optimal solution found')
    else:
        ### load (switch) best profile ###
        #print('Fair cost break down:')
        cost = compute_sum_cost(running_apps, best_profile[0], best_profile[1],print_cost=False)
        costs.append(cost)
        for idx, app in enumerate(running_apps):
            if app.model is None:
                app.load_model(best_profile[0][idx])
            elif app.load_model_time == 0 and app.nb_infers > app.nb_switches:
                app.load_model(best_profile[0][idx])

            app.cpu = best_profile[1][idx]

    #optimize_2(running_apps)

def optimize_1(running_apps, S_max, predefine_model_scheme):

    #second
    models_schemes = []
    for idx,app in enumerate(running_apps):
        models_schemes.append(app.can_models[predefine_model_scheme[idx]])

    cpu_product = itertools.product(cpu_allocations, repeat=len(running_apps) - 1)  # the last one is determined by preceding ones
    cpu_schemes = [cpu_scheme + (1 - np.sum(cpu_scheme),) for cpu_scheme in cpu_product if np.sum(cpu_scheme) < 1.0]

    ### load (switch) best profile ###
    #print('Fair cost break down:')
    cost = compute_sum_cost(running_apps, models_schemes, cpu_schemes[0],print_cost=False)
    costs.append(cost)
    for idx, app in enumerate(running_apps):
        if app.model is None:
            app.load_model(models_schemes[idx])
        elif app.load_model_time == 0 and app.nb_infers>=1:
            app.load_model(models_schemes[idx])

        app.cpu = cpu_schemes[0][idx]


def optimize_2(running_apps,S_max):
    model_product = itertools.product(*[app.can_models for app in running_apps])
    models_schemes = [each for each in model_product if compute_sum_mem(each) <= S_max]
    models_scheme = list(models_schemes[0])

    cpu_scheme = [0.01 for x in running_apps]

    while np.sum(cpu_scheme) < 1:
        delta_cpu = 0.01

        # select app with highest cost:
        app_idx = select_highest_cost_app(running_apps, models_scheme, cpu_scheme)

        # assign cpu resource to it
        cpu_scheme[app_idx] += delta_cpu

        # select model for it
        model_idx = select_lowest_cost_model(running_apps[app_idx], cpu_scheme[app_idx])
        models_scheme[app_idx] = running_apps[app_idx].can_models[model_idx]

    for idx, app in enumerate(running_apps):
        app.load_model(models_scheme[idx])
        app.cpu = cpu_scheme[idx]

    #print('MinMax cost break down:')
    compute_sum_cost(running_apps,models_scheme,cpu_scheme,False)





def main(alpha, beta):
    # alpha,     beta,   acc_min, latency_max
    model_types = [(resnet50_imagenet50_Models,     (alpha, beta, 0.542, 300), 'imagenet50 resnet50'),
                   (resnet50_scene_Models,          (alpha, beta, 0.6487, 700), 'scene resnet50'),
                   (resnet50_imagenet100_Models,    (alpha, beta, 0.7026, 1000), 'imagenet100 resnet50'),
                   (vgg4096_cifar10_Models,         (alpha, beta, 0.72, 200), 'cifar10 vgg4096'),
                   (vgg512_GTSRB_Models,            (alpha, beta, 0.9046, 100), 'GTSRB vgg512'),
                   (vgg2048_gender_Models,          (alpha, beta, 0.6765, 200), 'gender vgg2048')
                   ]

    results_dict = {each[2]: [] for each in model_types}


    np.random.seed(1023)

    running_scheme = np.random.randint(0,5,(int(1e5),6))

    S_max = 100
    optimize_now = False
    running_apps = []
    for i in range(len(model_types)):
        app_model_type = model_types[i]
        app = App(app_model_type[2], app_model_type[0], *app_model_type[1], freeze_model)
        running_apps.append(app)

    for t in range(int(1e4)):

        # random add apps
        if np.random.uniform() > gen_pro and len(running_apps) < 6:
        #if np.random.uniform() > 0.9 and len(running_apps) < 6:
            app_model_type = model_types[t % len(model_types)]
            app = App(app_model_type[2], app_model_type[0], *app_model_type[1],freeze_model)
            running_apps.append(app)
            optimize_now = True

        # random delete apps
        if np.random.uniform() > exit_pro and len(running_apps) > 2:
        #if np.random.uniform() > 0.9 and len(running_apps) > 2:
            remove_index = 0

            if running_apps[remove_index].nb_infers >= 1:
                app = running_apps.pop(remove_index)

                app.print_sum()
                results_dict[app.name].append(app)
                optimize_now = True

        if (t+1)%100 == 0:

            optimize_now=True

        if t == 0 or optimize_now:

            if fair_allocation:
                optimize_1(running_apps, S_max, running_scheme[t])
            else:
                optimize_2(running_apps,S_max)
            optimize_now = False


        # if t % 1000 == 0:
        #     print('########################')
        #     for each in running_apps:
        #         each.print_sum()

        for idx, app in enumerate(running_apps):
            app.run_model()

    infers = 0
    on_time_infers = 0
    sum_acc = 0
    sum_fps = 0
    sum_finished_apps = 0
    sum_load_time = 0
    sum_infer_time =0
    for k in sorted(results_dict.keys()):
        v = stat_apps(results_dict[k])
        on_time_infers += v[0]
        infers += v[1]
        sum_acc += v[2]
        sum_fps += v[3]
        sum_finished_apps += v[4]
        sum_load_time += v[5]
        sum_infer_time += v[6]
        sum_finished_apps += len(results_dict[k])
    print('{:.4f},{},{:.4f},{},{},{}'.format(sum_acc/infers, infers, sum_fps/infers, sum_finished_apps, sum_load_time,
                                          sum_infer_time))
    #print('on_time_infers', on_time_infers / infers)
    #print('cost', np.sum(costs))
    #print('Used time:{:f}'.format(time.time() - begin_time))


def stat_apps(finished_apps):
    delta_acc_list = []
    delta_latency_list = []
    latency_list = []
    sum_nb_infers = 0
    sum_nb_switches = 0
    sum_load_model_time = 0
    for app in finished_apps:
        if app.nb_infers == 0:
            pass
        else:
            delta_acc_list.append(np.array(app.infer_accs) - app.acc_min),
            delta_latency_list.append(np.array(app.ellapse_times) - app.latency_max)
            latency_list.append(app.ellapse_times)

            sum_nb_infers += app.nb_infers
            sum_nb_switches += app.nb_switches
            sum_load_model_time +=app.sum_load_model_time
    latencies = np.hstack(latency_list).flatten()
    one_by_one_fps = np.mean(1000. / np.array(latencies))
    mean_fps = sum_nb_infers / np.sum(latencies) * 1000.

    delta_accuracies = np.hstack(delta_acc_list).flatten()
    delta_latencies = np.hstack(delta_latency_list).flatten()
    assert len(delta_latencies) == sum_nb_infers
    on_time_inferences = delta_latencies <= 0
    # print(finished_apps[0].name,app.latency_max)
    # print('Delta acc: {:.2f}, '
    #       'on time rate {:.2f}, '
    #       'average_latency:{:.2f}, '
    #       'ontime_inferences:{}, '
    #       'nb_infers:{:d}, '
    #       'nb_switches:{:d}, '
    #       'mean_fps:{:.2f}, '
    #       'one_by_one_fps:{:.2f}'.format(np.mean(delta_accuracies),
    #                                      np.sum(on_time_inferences.astype(int)) / len(delta_latencies),
    #                                      np.mean(delta_latencies),
    #                                      np.sum(on_time_inferences.astype(int)),
    #                                      np.sum(sum_nb_infers),
    #                                      np.sum(sum_nb_switches),
    #                                      one_by_one_fps,
    #                                      mean_fps))

    print('{:.3f},{:.3f},{:.3f},{},{:d},{:d},{:.2f},{:.2f}'.format(
        np.mean(delta_accuracies),
        np.sum(on_time_inferences.astype(int)) / len(delta_latencies),
        np.mean(delta_latencies),
        np.sum(on_time_inferences.astype(int)),
        np.sum(sum_nb_infers),
        np.sum(sum_nb_switches),
        one_by_one_fps,
        mean_fps))

    return (np.sum(on_time_inferences.astype(int)), len(delta_latencies),
            np.sum(delta_accuracies), sum_nb_infers*one_by_one_fps, sum_nb_switches,sum_load_model_time,
            np.sum(latencies))


def compute_sum_cost(running_apps, model_scheme, cpu_scheme, print_cost=False):
    return np.sum(
        [app.compute_cost(model_scheme[idx], cpu_scheme[idx], print_cost) for idx, app in enumerate(running_apps)])


def select_highest_cost_app(running_apps, model_scheme, cpu_scheme, print_cost=False):
    costs_list = [app.compute_cost(model_scheme[idx], cpu_scheme[idx], print_cost) for idx, app in
                  enumerate(running_apps)]
    return np.argmax(costs_list)


def select_lowest_cost_model(app, cpu_resouce, print_cost=False):
    return np.argmin(
        [app.compute_cost(can_model, cpu_resouce, print_cost) for can_model in app.can_models])


def compute_max_cost(running_apps, model_scheme, cpu_scheme, print_cost=False):
    return np.max(
        [app.compute_cost(model_scheme[idx], cpu_scheme[idx], print_cost) for idx, app in enumerate(running_apps)])


def compute_sum_mem(models):
    return np.sum([model.size for model in models])

def main_2():

    def run(app_model_type,freeze):
        run_time = 1000000
        app1 = App(app_model_type[2], app_model_type[0], *app_model_type[1], freeze)
        app1.load_model(app1.can_models[0])
        app1.cpu = 1
        multiplier = 2
        for i in range(run_time):
            app1.run_model()
            if app1.nb_infers == 1*multiplier:
                app1.load_model(app1.can_models[1])
            elif app1.nb_infers == 2*multiplier:
                app1.load_model(app1.can_models[2])
            elif app1.nb_infers == 3*multiplier:
                app1.load_model(app1.can_models[3])
            elif app1.nb_infers == 4*multiplier:
                app1.load_model(app1.can_models[4])
            elif app1.nb_infers == 5*multiplier:
                app1.load_model(app1.can_models[3])
            elif app1.nb_infers == 6*multiplier:
                app1.load_model(app1.can_models[2])
            elif app1.nb_infers == 7*multiplier:
                app1.load_model(app1.can_models[1])
            elif app1.nb_infers == 8*multiplier:
                app1.load_model(app1.can_models[0])
            elif app1.nb_infers >8*multiplier:
                break
        app1.print_sim_2()

    alpha = 0.00016
    beta = 0
    model_types = [(resnet50_imagenet50_Models,     (alpha, beta, 0.542, 300), 'imagenet50 resnet50'),
                   (resnet50_scene_Models,          (alpha, beta, 0.6487, 700), 'scene resnet50'),
                   (resnet50_imagenet100_Models,    (alpha, beta, 0.7026, 1000), 'imagenet100 resnet50'),
                   (vgg4096_cifar10_Models,         (alpha, beta, 0.72, 200), 'cifar10 vgg4096'),
                   (vgg512_GTSRB_Models,            (alpha, beta, 0.9046, 100), 'GTSRB vgg512'),
                   (vgg2048_gender_Models,          (alpha, beta, 0.6765, 200), 'gender vgg2048')
                   ]
    for model_type in model_types:
        print(model_type[2])
        run(model_type,True)
        run(model_type,False)



if __name__ == '__main__':

    main_2()

    #main(alpha=s_alpha, beta=0)

    # alpha_list = [1,0.001,0.0008,0.0007,0.00065,0.0005,0.0003,0.00029,0.00026,0.00023,
    #               0.0002,0.00019,0.00016,0.00013,0.0001,0.00005,0.00003,0.00002,0.000017,
    #               0.000015,0.000013,0.00001,0]
    # for a in alpha_list:
    #     alpha=a
    #     main(alpha, beta=0)



    # beta_list = [1,0.1,0.001,0.0001,0.00001,0]
    # for b in beta_list:
    #     beta = b
    #     main(0.0005, beta)
