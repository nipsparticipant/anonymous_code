from utils.utility import *
from experiments import *
from vaegp import *


def read_log(logname, maxlen=None):
    f = open(logname, 'r')
    rlines = f.readlines()
    lines = []
    for l in rlines:
        if 'rmse' in l:
            lines.append(l)
    rmse = np.array([float(eval(l)['rmse']) for l in lines])
    iter = np.array([float(eval(l)['iter']) for l in lines])
    time = np.array([float(eval(l)['time']) for l in lines])
    if maxlen is None:
        return rmse, time, iter
    else:
        return rmse[:maxlen], time[:maxlen], iter[:maxlen]


def create_std_plot():
    plt.figure()
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# x is list of np.array
def compute_mean_std(x):
    x = np.array(x)
    mean = np.sum(x, axis=0) / x.shape[0]
    std = (np.sum((x - mean) ** 2, axis=0) / x.shape[0]) ** 0.5
    return mean, std

def main():
    prefix = './results_to_be_processed/gas10/'
    output_prefix = './performance_plot/gas10/'
    seed = ['s1', 's2', 's3', 's4', 's5']
    samp = ['16', '32', '64', '128']

    rmse_full = []
    time_full = []
    for s in seed:
        full_logname = prefix + '_'.join(['full', s]) + '.log'
        print(full_logname)
        rmse, time, iter = read_log(full_logname)
        rmse_full.append(rmse)
        time_full.append(time)
    
    mean_rmse_full, std_rmse_full = compute_mean_std(rmse_full)
    mean_time_full, std_time_full = compute_mean_std(time_full)

    for p in samp:
        rmse_vaegp = []
        rmse_ssgp = []
        time_vaegp = []
        time_ssgp = []
        iter_vaegp = []
        iter_ssgp = []
        for s in seed:
            vaegp_logname = prefix + '_'.join(['vaegp', p, s]) + '.log'
            rmse, time, iter = read_log(vaegp_logname)
            rmse_vaegp.append(rmse)
            time_vaegp.append(time)
            iter_vaegp.append(iter)

            ssgp_logname = prefix + '_'.join(['ssgp', p, s]) + '.log'
            rmse, time, iter = read_log(ssgp_logname)
            rmse_ssgp.append(rmse)
            time_ssgp.append(time)
            iter_ssgp.append(iter)

        mean_rmse_vaegp, std_rmse_vaegp = compute_mean_std(rmse_vaegp)
        mean_rmse_ssgp, std_rmse_ssgp = compute_mean_std(rmse_ssgp)
        mean_time_vaegp, std_time_vaegp = compute_mean_std(time_vaegp)
        mean_time_ssgp, std_time_ssgp = compute_mean_std(time_ssgp)
        iter, _ = compute_mean_std(iter_vaegp)

        l1 = 'Revisited SSGP (p=' + p + ')'
        l2 = 'SSGP (p=' + p + ')'
        l3 = 'FGP'

        create_std_plot()
        plt.xlabel('No. Iterations')
        plt.ylabel('RMSE of CO concentration (ppm)')
        try:
            plt.errorbar(iter[0: -1: 5], mean_rmse_vaegp[0: -1: 5], 0.5 * std_rmse_vaegp[0: -1: 5],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l1)
            plt.errorbar(iter[0: -1: 5], mean_rmse_ssgp[0: -1: 5], 0.5 * std_rmse_ssgp[0: -1: 5],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l2)
            plt.errorbar(iter[0: -1: 5], mean_rmse_full[0: -1: 5], 0.5 * std_rmse_full[0: -1: 5],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l3)
            plt.legend(loc="upper right")
            plt.savefig(output_prefix + '_'.join(['rmse', p]) + '.png')
        except Exception as e:
            print(e)
            print(mean_rmse_vaegp)
            print(std_rmse_vaegp)
            print(iter)

        create_std_plot()
        plt.xlabel('No. Iterations')
        plt.ylabel('Cumulative Runtime (ms)')
        try:
            plt.errorbar(iter[0: -1: 5], mean_time_vaegp[0: -1: 5], 0.5 * std_time_vaegp[0: -1: 5],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l1)
            plt.errorbar(iter[0: -1: 5], mean_time_ssgp[0: -1: 5], 0.5 * std_time_ssgp[0: -1: 5],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l2)
            plt.errorbar(iter[0: -1: 5], mean_time_full[0: -1: 5], 0.5 * std_time_full[0: -1: 5],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l3)
            plt.legend(loc="upper right")
            plt.savefig(output_prefix + '_'.join(['time', p]) + '.png')
        except Exception as e:
            print(e)
            print(mean_rmse_vaegp)
            print(std_rmse_vaegp)
            print(iter)


def main2():
    prefix = './results_to_be_processed/gas500/'
    output_prefix = './performance_plot/gas500/'
    seed = ['s1', 's2', 's3', 's4', 's5']
    samp = ['16', '32', '64', '128']
    '''
    rmse_full = []
    time_full = []
    for s in seed:
        full_logname = prefix + '_'.join(['full', s]) + '.log'
        print(full_logname)
        rmse, time, iter = read_log(full_logname)
        rmse_full.append(rmse)
        time_full.append(time)

    mean_rmse_full, std_rmse_full = compute_mean_std(rmse_full)
    mean_time_full, std_time_full = compute_mean_std(time_full)
    '''
    for p in samp:
        rmse_vaegp = []
        rmse_ssgp = []
        time_vaegp = []
        time_ssgp = []
        iter_vaegp = []
        iter_ssgp = []
        for s in seed:
            vaegp_logname = prefix + '_'.join(['vaegp', p, s]) + '.log'
            rmse, time, iter = read_log(vaegp_logname)
            rmse_vaegp.append(rmse)
            time_vaegp.append(time)
            iter_vaegp.append(iter)

            ssgp_logname = prefix + '_'.join(['ssgp', p, s]) + '.log'
            rmse, time, iter = read_log(ssgp_logname, maxlen=10)
            rmse_ssgp.append(rmse)
            time_ssgp.append(time)
            iter_ssgp.append(iter)

        mean_rmse_vaegp, std_rmse_vaegp = compute_mean_std(rmse_vaegp)
        mean_rmse_ssgp, std_rmse_ssgp = compute_mean_std(rmse_ssgp)
        mean_time_vaegp, std_time_vaegp = compute_mean_std(time_vaegp)
        mean_time_ssgp, std_time_ssgp = compute_mean_std(time_ssgp)
        iter, _ = compute_mean_std(iter_vaegp)

        l1 = 'Revisited SSGP (p=' + p + ')'
        l2 = 'SSGP (p=' + p + ')'
        l3 = 'FGP'

        create_std_plot()
        plt.xlabel('No. Iterations')
        plt.ylabel('RMSE of CO concentration (ppm)')
        try:
            plt.errorbar(iter[0: -1: 1], mean_rmse_vaegp[0: -1: 1], std_rmse_vaegp[0: -1: 1],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l1)
            plt.errorbar(iter[0: -1: 1], mean_rmse_ssgp[0: -1: 1], 0.5 * std_rmse_ssgp[0: -1: 1],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l2)
            '''
            plt.errorbar(iter[0: -1: 5], mean_rmse_full[0: -1: 5], 0.5 * std_rmse_full[0: -1: 5],
                         linewidth=2, markersize=12, label=l3)
            '''
            plt.legend(loc="upper right")
            plt.savefig(output_prefix + '_'.join(['rmse', p]) + '.png')
        except Exception as e:
            print(e)
            print(mean_rmse_vaegp)
            print(std_rmse_vaegp)
            print(iter)

        create_std_plot()
        plt.xlabel('No. Iterations')
        plt.ylabel('Cumulative Runtime (ms)')
        try:
            plt.errorbar(iter[0: -1: 1], mean_time_vaegp[0: -1: 1], std_time_vaegp[0: -1: 1],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l1)
            plt.errorbar(iter[0: -1: 1], mean_time_ssgp[0: -1: 1], 0.5 * std_time_ssgp[0: -1: 1],
                         linestyle='--', marker='^', linewidth=2, markersize=12, label=l2)
            '''
            plt.errorbar(iter[0: -1: 5], mean_time_full[0: -1: 5], 0.5 * std_time_full[0: -1: 5],
                         linewidth=2, markersize=12, label=l3)
            '''
            plt.legend(loc="upper right")
            plt.savefig(output_prefix + '_'.join(['time', p]) + '.png')
        except Exception as e:
            print(e)
            print(mean_rmse_vaegp)
            print(std_rmse_vaegp)
            print(iter)


if __name__ == '__main__':
    main()
    #main2()
