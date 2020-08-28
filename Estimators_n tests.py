# Parameters independent of the setting:
alpha = 0.95
max_trials = 100
min_sample_size = 300
max_sample_size = 2000
samples_number = 50
sample_sizes = linspace(min_sample_size, max_sample_size, samples_number).astype(int)
string_sample_sizes = [str(a) for a in sample_sizes]


# Defining the settings :
def setting_parameters(setting):
    if setting == 1:
        return [1.5, 1.15, 4.1, 3]
    if setting == 2:
        return [2.2, 1.75, 6, 3]
    if setting == 3:
        return [2.4, 1.95, 6.6, 3]


for i in range(1, 4):

    setting_number = str(i)

    # defining the parameters of the setting "i"___
    parameters = setting_parameters(i)
    sigma_normal = parameters[0]
    sigma_lognormal = parameters[1]
    x_m = parameters[2]
    a = parameters[3]
    # ___________________________________________

    for string_estimator in dic:
        print(string_estimator, setting_number)
        # for each estimator (method), we run the function setting to obtain results and store them in a csv file
        if string_estimator == 'random_trunc':
            results = setting(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_sample_size,
                              max_sample_size, samples_number, 0.95, u='empirical_2nd_moment')

            df = pd.DataFrame(results,
                              columns=[''] + [a for a in string_sample_sizes])
            df.to_csv('/results/' + setting_number + '_random_trunc_empirical.csv', index=False)

            results = setting(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_sample_size,
                              max_sample_size, samples_number, 0.95, u='true_2nd_moment')

            df = pd.DataFrame(results,
                              columns=[''] + [a for a in string_sample_sizes])
            df.to_csv('/results/' + setting_number + '_random_trunc_true_variance.csv', index=False)

        else:
            results = setting(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_sample_size,
                              max_sample_size, samples_number, 0.95, u=0)

            df = pd.DataFrame(results,
                              columns=[''] + [a for a in string_sample_sizes])
            df.to_csv('/results/' + setting_number + '_' + string_estimator + '.csv', index=False)
