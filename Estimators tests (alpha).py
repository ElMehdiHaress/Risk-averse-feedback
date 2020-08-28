# Parameters independent of the setting :
n = 400
max_trials = 100
min_alpha = 0.5
max_alpha = 0.97
alpha_size = 50
alpha_line = linspace(min_alpha, max_alpha, alpha_size)
string_alpha_line = [str(a) for a in alpha_line]


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
    # ____________________________________________

    for string_estimator in dic:
        # for each estimator (method), we run the function setting to obtain results and store them in a csv file
        print(string_estimator, setting_number)
        if string_estimator == 'random_trunc':
            results = setting_alpha(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_alpha,
                                    max_alpha, alpha_size, n, u='empirical_2nd_moment')

            df = pd.DataFrame(results,
                              columns=[''] + [a for a in string_alpha_line])
            df.to_csv('/results/' + setting_number + '_random_trunc_empirical.csv', index=False)

            results = setting_alpha(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_alpha,
                                    max_alpha, alpha_size, n, u='true_2nd_moment')

            df = pd.DataFrame(results,
                              columns=[''] + [a for a in string_alpha_line])
            df.to_csv('/results/' + setting_number + '_random_trunc_true_variance.csv', index=False)

        else:
            results = setting_alpha(string_estimator, sigma_normal, sigma_lognormal, x_m, a, max_trials, min_alpha,
                                    max_alpha, alpha_size, n, u=0)

            df = pd.DataFrame(results,
                              columns=[''] + [a for a in string_alpha_line])
            df.to_csv('/results/' + setting_number + '_' + string_estimator + '.csv', index=False)
