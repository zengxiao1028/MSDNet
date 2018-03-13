import numpy as np
import copy
from simulation.model import App, Model, cpu_speed
import itertools
import time
import multiprocessing
import matplotlib.pyplot as plt
PRINT_COST = False
REVERSE_SEARCH = False


use_baseline = False
fair_allocation = True

minTotalCost = False



if use_baseline:
    from simulation.baseline_model_configs import *
    freeze_model = False
else:
    from simulation.model_configs import *
    freeze_model = True

S_max = 100
costs = []

resnet50_imagenet50_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet50_configs]
resnet50_scene_Models = [Model.init_from_list('resnet', config) for config in resnet50_scene_configs]
resnet50_imagenet100_Models = [Model.init_from_list('resnet', config) for config in resnet50_imagenet100_configs]
vgg4096_cifar10_Models = [Model.init_from_list('vgg4096', config) for config in VGG4096_cifar10_configs]
vgg512_GTSRB_Models = [Model.init_from_list('vgg512', config) for config in VGG512_GTSRB_configs]
vgg2048_gender_Models = [Model.init_from_list('vgg2048', config) for config in VGG2048_gender_configs]

if use_baseline:
    sel = 1
    resnet50_imagenet50_Models = [resnet50_imagenet50_Models[sel] ]
    resnet50_scene_Models = [resnet50_scene_Models[sel]]
    resnet50_imagenet100_Models = [resnet50_imagenet100_Models[sel]]
    vgg4096_cifar10_Models = [vgg4096_cifar10_Models[sel]]
    vgg512_GTSRB_Models = [vgg512_GTSRB_Models[sel] ]
    vgg2048_gender_Models = [vgg2048_gender_Models[sel]]


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


def optimize(running_apps):
    model_product = itertools.product(*[app.can_models for app in running_apps])
    models_schemes = [each for each in model_product if compute_sum_mem(each) <= S_max]

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
            app.load_model(best_profile[0][idx])
            app.cpu = best_profile[1][idx]

    #optimize_2(running_apps)


def optimize_2(running_apps):
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

    begin_time = time.time()
    np.random.seed(1023)
    optimize_now = False
    running_apps = []
    benchmark = []
    for i in range(len(model_types)):
        app_model_type = model_types[i]
        app = App(app_model_type[2], app_model_type[0], *app_model_type[1], freeze_model)
        running_apps.append(app)

    for t in range(int(1e5)):

        # random add apps
        if np.random.uniform() > 0.99 and len(running_apps) < 6:
        #if np.random.uniform() > 0.9 and len(running_apps) < 6:
            app_model_type = model_types[t % len(model_types)]
            app = App(app_model_type[2], app_model_type[0], *app_model_type[1],freeze_model)
            running_apps.append(app)
            optimize_now = True

        # random delete apps
        if np.random.uniform() > 0.999 and len(running_apps) > 2:
        #if np.random.uniform() > 0.9 and len(running_apps) > 2:
            remove_index = 0

            if running_apps[remove_index].nb_infers > 1:
                app = running_apps.pop(remove_index)

                #app.print_sum()
                results_dict[app.name].append(app)
                optimize_now = True

        if t == 0 or optimize_now:
            if fair_allocation:
                optimize(running_apps)
            else:
                optimize_2(running_apps)
            optimize_now = False

        #if t % 1000 == 0:
        #    print(t, len(running_apps), [app.model.name for app in running_apps])

        for idx, app in enumerate(running_apps):
            app.run_model()

        # record for benchmark
        benchmark.append(len(running_apps))

    # plot benchmark
    # benchmark = np.array(benchmark)
    # hist = np.histogram(benchmark, bins=[1,2,3,4,5,6,7])
    # hist_horm = np.histogram(benchmark, bins=[1, 2, 3, 4, 5, 6, 7], density=True)
    # print(hist)
    # print(hist_horm)
    # plt.hist(benchmark,bins=[1,2,3,4,5,6,7])
    # plt.title("Avg # of running APPs in the experiment")
    # plt.show()
    infers = 0
    on_time_infers = 0
    sum_acc = 0
    sum_fps = 0

    for k in sorted(results_dict.keys()):
        v = stat_apps(results_dict[k])
        on_time_infers += v[0]
        infers += v[1]
        sum_acc += v[2]
        sum_fps += v[3]
    print('{:.4f},{},{:.4f}'.format(sum_acc/infers,infers,sum_fps/infers))

    #print('on_time_infers', on_time_infers / infers)
    #print('cost', np.sum(costs))
    #print('Used time:{:f}'.format(time.time() - begin_time))

    # sum_run_time = []
    # sum_gen_apps = []
    # app_group_names = []
    # for k in sorted(results_dict.keys()):
    #     v = stat_apps_benchmark(results_dict[k])
    #     sum_run_time.append(v[0])
    #     sum_gen_apps.append(v[1])
    #     app_group_names.append(v[2])

    # ind = np.arange(6)  # the x locations for the groups
    # width = 0.35  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind, sum_run_time, width, color='r')
    # rects2 = ax.bar(ind + width, sum_gen_apps, width, color='y')
    #
    # # add some text for labels, title and axes ticks
    # ax.set_ylabel('time')
    # ax.set_title('avg inference time and #switches for each APP')
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(app_group_names)
    #
    # ax.legend((rects1[0], rects2[0]), ('Infer Time', '#switches'))
    #
    # def autolabel(rects):
    #     """
    #     Attach a text label above each bar displaying its height
    #     """
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
    #                 '%d' % int(height),
    #                 ha='center', va='bottom')
    #
    # autolabel(rects1)
    # autolabel(rects2)
    #
    # plt.show()



def stat_apps_benchmark(finished_apps):
    delta_acc_list = []
    delta_latency_list = []
    latency_list = []
    sum_nb_infers = 0
    sum_nb_switches = 0
    for app in finished_apps:
        if app.nb_infers == 0:
            pass
        else:
            delta_acc_list.append(np.array(app.infer_accs) - app.acc_min),
            delta_latency_list.append(np.array(app.ellapse_times) - app.latency_max)
            latency_list.append(app.ellapse_times)

            sum_nb_infers += app.nb_infers
            sum_nb_switches += app.nb_switches

    latencies = np.hstack(latency_list).flatten()
    delta_latencies = np.hstack(delta_latency_list).flatten()
    assert len(delta_latencies) == sum_nb_infers

    print(finished_apps[0].name, len(finished_apps), np.sum(latencies)/1000.)
    return len(finished_apps), np.sum(latencies)/10000., finished_apps[0].name


def stat_apps(finished_apps):
    delta_acc_list = []
    delta_latency_list = []
    latency_list = []
    sum_nb_infers = 0
    sum_nb_switches = 0
    for app in finished_apps:
        if app.nb_infers == 0:
            pass
        else:
            ##accuracy gain
            delta_acc_list.append(np.array(app.infer_accs) - app.acc_min),
            ##absolute
            #delta_acc_list.append(np.array(app.infer_accs)),
            delta_latency_list.append(np.array(app.ellapse_times) - app.latency_max)
            latency_list.append(app.ellapse_times)

            sum_nb_infers += app.nb_infers
            sum_nb_switches += app.nb_switches

    latencies = np.hstack(latency_list).flatten()
    one_by_one_fps = np.mean(1000. / np.array(latencies))
    mean_fps = sum_nb_infers / np.sum(latencies) * 1000.

    delta_accuracies = np.hstack(delta_acc_list).flatten()
    delta_latencies = np.hstack(delta_latency_list).flatten()
    assert len(delta_latencies) == sum_nb_infers
    on_time_inferences = delta_latencies <= 0
    #print(finished_apps[0].name,len(finished_apps))
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

    # print('acc{:.3f},on_time_rate{:.3f},avg_latency{:.3f},on_time_infers{},nb_infers{:d},nb_swithces{:d},fps{:.2f},fps{:.2f}'.format(
    #     np.mean(delta_accuracies),
    #     np.sum(on_time_inferences.astype(int)) / len(delta_latencies),
    #     np.mean(delta_latencies),
    #     np.sum(on_time_inferences.astype(int)),
    #     np.sum(sum_nb_infers),
    #     np.sum(sum_nb_switches),
    #     one_by_one_fps,
    #     mean_fps))

    return (np.sum(on_time_inferences.astype(int)), len(delta_latencies), np.sum(delta_accuracies), sum_nb_infers*mean_fps )


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


if __name__ == '__main__':


    alpha_list = [1,0.001,0.0008,0.0007,0.00065,0.0005,0.0003,0.00029,0.00026,0.00023,
                  0.0002,0.00019,0.00016,0.00013,0.0001,0.00005,0.00003,0.00002,0.000017,
                  0.000015,0.000013,0.00001,0]
    for a in alpha_list:
        alpha=a
        main(alpha, beta=0)



    # beta_list = [1,0.1,0.001,0.0001,0.00001,0]
    # for b in beta_list:
    #     beta = b
    #     main(0.0005, beta)
