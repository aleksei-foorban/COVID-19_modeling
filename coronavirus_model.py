import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def logistic_model(x, a, b, c):
    return c / (1 + np.exp(-(x - b) / a))


def logistic_model_predefined_b(x, a, c):
    return c / (1 + np.exp(-(x - 90.7) / a))


def estimate_growth_model(df, title, func, p0, param_bounds,
                          response_column='deceduti', start_point=0):

    df = df.loc[:, ['data', response_column]]
    df = df.iloc[start_point:, :]
    FMT = '%Y-%m-%d %H:%M:%S'
    date = df['data']
    df['data'] = date.map(lambda x: (x - datetime.strptime("2020-01-01 00:00:00", FMT)).days)

    # Fitting the curve
    x = list(df.iloc[:, 0])
    y = list(df.iloc[:, 1])
    fit = curve_fit(func, x, y, p0=p0, bounds=param_bounds)
    param_est = fit[0]
    if func == logistic_model:
        sol = fsolve(lambda x: logistic_model(x, *param_est) - param_est[2], param_est[1])
    errors = [np.sqrt(fit[1][i][i]) for i in range(len(param_est))]
    y_pred_func = [func(i, *param_est) for i in x]
    print("Evaluated parameters:")
    print(param_est)
    if func == logistic_model:
        print("Pandemic end date: " + str(sol))
    print("Errors of the parameters evaluation:")
    print(errors)
    print("MSE: " + str(mean_squared_error(y, y_pred_func)))

    pred_x = list(range(max(x)+1, 140))

    # Plotting the results
    plt.subplots(2, 1, sharex=True, squeeze=True)
    ax1 = plt.subplot(211)
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(10)

    ax1.set_title(title)
    days_since_1_jan = x + pred_x
    num_points = len(x)
    jan_1st = datetime(2020, 1, 1)
    time_axis = [jan_1st + timedelta(d) for d in days_since_1_jan]

    # Plotting real data
    plt.scatter(time_axis[:len(y)], y, label="Real data",color="red")
    # plt.scatter(list(df_test.iloc[1:, 0]), list(df_test.iloc[1:, 1]), label="Test set data", color="violet")

    # Plotting predicted logistic curve
    y_pred = [func(i, *param_est) for i in days_since_1_jan]
    plt.plot(time_axis, y_pred, label="Logistic model")
    ax1.label_outer()

    # Bootstrapping prediction intervals
    residuals = np.subtract(y, y_pred_func)
    bootstrap_pred_array = []

    for i in range(10000):
        residuals_sample = np.random.choice(residuals, len(residuals))
        y_bootstrap = np.add(residuals_sample, y_pred_func)
        try:
            fit = curve_fit(func, x, y_bootstrap, p0=p0, bounds=param_bounds)
            param_est = fit[0]
        except:
            pass
        y_pred_bootstrap = [func(i, *param_est) for i in days_since_1_jan]
        n_daily_deaths_bootstrap = np.diff(y_pred_bootstrap)
        if len(bootstrap_pred_array) > 0:
            bootstrap_pred_array = np.vstack((bootstrap_pred_array, y_pred_bootstrap))
            bootstrap_daily_deaths_array = np.vstack((bootstrap_daily_deaths_array, n_daily_deaths_bootstrap))
        else:
            bootstrap_pred_array = y_pred_bootstrap
            bootstrap_daily_deaths_array = n_daily_deaths_bootstrap

    y_bootstrap_pred_int_low = np.percentile(bootstrap_pred_array, 2.5, axis=0)
    y_bootstrap_pred_int_high = np.percentile(bootstrap_pred_array, 97.5, axis=0)
    daily_deaths_int_low = np.percentile(bootstrap_daily_deaths_array, 2.5, axis=0)
    daily_deaths_int_high = np.percentile(bootstrap_daily_deaths_array, 97.5, axis=0)

    # Plotting bootstrapped prediction intervals
    plt.plot(time_axis[len(y):], y_bootstrap_pred_int_low[len(y):], label="Bootstrap prediction interval", linestyle='dashed',
             color='blue', alpha=0.4)
    plt.plot(time_axis[len(y):], y_bootstrap_pred_int_high[len(y):], linestyle='dashed', color='blue', alpha=0.4)
    plt.legend()
    plt.ylabel(response_column + " (cumulative)")
    plt.xlim(min(time_axis), max(time_axis))
    plt.ylim((min(y)*0.9, max(y_bootstrap_pred_int_high)*1.1))

    # Plotting daily data
    ax2 = plt.subplot(212, sharex=ax1)
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(10)
    daily_deaths = np.diff(y)
    daily_deaths_pred = np.diff(y_pred)
    daily_deaths_pred = np.around(daily_deaths_pred).astype(int)

    n_obs = len(daily_deaths)
    diff_len = len(daily_deaths_pred) - len(daily_deaths)
    nones = [None] * diff_len
    daily_deaths = np.append(daily_deaths, nones)

    d = {'date': time_axis[1:], 'daily_deaths': daily_deaths, 'daily_deaths_pred': daily_deaths_pred,
         'daily_deaths_pred_low': daily_deaths_int_low, 'daily_deaths_pred_high': daily_deaths_int_high}
    df_deaths = pd.DataFrame(d)

    plt.bar(x=time_axis[1:(n_obs+1)], height=daily_deaths[:n_obs], color='red', alpha=0.8)
    plt.plot(time_axis[1:], daily_deaths_pred, color='blue', alpha=0.5)
    for i_x, i_y, i_y_low, i_y_high in zip(time_axis[num_points::5], daily_deaths_pred[num_points-1::5],
                                           daily_deaths_int_low[num_points-1::5], daily_deaths_int_high[num_points-1::5]):
        plt.text(i_x, i_y, '{}: {}'.format(str(i_x)[:10], i_y), fontsize=8)
        plt.text(i_x+timedelta(hours=17), i_y-40, '({}, {})'.format(int(i_y_low), int(i_y_high)), fontsize=8)
    # plt.bar(x=df_test.iloc[1:, 0], height=np.diff(list(df_test.iloc[:, 1])), color='violet', alpha=0.8)
    plt.ylabel(response_column + " (daily)")
    plt.show()

    return df_deaths


def growth_model_sliding_data(df_data, func, p0, param_bounds, title,
                              response_column='deceduti', moving=False, window=10):

    list_date = []
    list_param = []

    FMT = '%Y-%m-%d %H:%M:%S'
    df_init = df_data.loc[:, ['data', response_column]]
    date = df_init['data']
    df_init['data'] = date.map(lambda x: (x - datetime.strptime("2020-01-01 00:00:00", FMT)).days)
    x = list(df_init.iloc[:, 0])
    y = list(df_init.iloc[:, 1])
    fit_init = curve_fit(func, x, y, p0=p0, bounds=param_bounds, maxfev=5000)
    param_init = fit_init[0]

    for i in range(df_data.shape[0] - window):
        df = df_data.loc[:, ['data', response_column]]
        if moving == True:
            start_point = i
        else:
            start_point = 0
        df = df.iloc[start_point:(i+window+1), :]
        date = df['data']
        df['data'] = date.map(lambda x: (x - datetime.strptime("2020-01-01 00:00:00", FMT)).days)

        # Fitting the curve
        x = list(df.iloc[:, 0])
        y = list(df.iloc[:, 1])
        try:
            fit = curve_fit(func, x, y, p0=param_init, bounds=param_bounds, maxfev=5000)
            param_est = fit[0]
        except:
            pass
        pred_x = list(range(max(x) + 1, 100))
        days_since_1_jan = x + pred_x
        jan_1st = datetime(2020, 1, 1)
        time_axis = [jan_1st + timedelta(d + 1) for d in days_since_1_jan]

        # Plotting predicted logistic curve
        y_pred = [func(i, *param_est) for i in days_since_1_jan]

        plt.plot(time_axis, y_pred, alpha= (i + 1) / (df_data.shape[0] - window), color="blue", linestyle="dashed")

        list_date.append(jan_1st + timedelta(days=max(x) + 1))
        list_param.append(param_est)


    # Plotting real data
    y_common = list(df_data[response_column])
    x_common = list(df_data['data'])
    plt.scatter(x_common, y_common, label="Real data", color="red")
    axes = plt.axes()
    x_extended = [max(x_common) + timedelta(days=d+1) for d in range((max(time_axis) - max(x_common)).days)]
    x_extended = x_common + x_extended
    axes.set_xticks(x_extended[1::7])
    axes.set_title(title)
    plt.show()

    daily_deaths = np.diff(y_common)
    n_obs = len(daily_deaths)

    fig, axes = plt.subplots(nrows=len(param_init)+1, ncols=1, sharex=True)
    ax1 = plt.subplot(411)
    ax1.set_title('Daily new deaths (Italy)')
    plt.bar(x=x_common[1:(n_obs + 1)], height=daily_deaths, color='red', alpha=0.8)
    ax1.label_outer()

    df_result = pd.DataFrame(list_param)
    df_result['date'] = pd.Series(list_date)

    for ax, i in zip(axes.flatten()[1:], range(len(param_init))):
        ax.set_title('parameter {} moving window estimation'.format(i+1))
        ax.plot(df_result.date, df_result.iloc[:, i], label='moving window estimate')
        ax.axhline(y=param_init[i], xmin=0, xmax=1000, linestyle='--', color='green', alpha=0.5, label='estimate over all data')
        if i == 0:
            ax.legend(['moving window estimate', 'estimate over all data'])
        ax1.get_shared_x_axes().join(ax1, ax)
        ax.label_outer()

    plt.show()

    return df_result


if __name__ == '__main__':

    url_italy = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    url_regions = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
    url_world = "https://covid.ourworldindata.org/data/full_data.csv"
    url_s_korea = "https://raw.githubusercontent.com/jihoo-kim/Data-Science-for-COVID-19-old/master/dataset/Time/Time.csv"
    df_italy = pd.read_csv(url_italy)
    df_italy['positive_test_share'] = df_italy.totale_casi / df_italy.tamponi
    df_italy['data'] = pd.to_datetime(df_italy['data'])
    df_italy['report_date'] = df_italy.data.dt.date

    fig_1, _ = plt.subplots(2, 1)
    ax1 = plt.subplot(211)
    ax1.set_title("New cases of COVID-19 in Italy per day")
    plt.bar(x=df_italy.data[1:], height=np.diff(df_italy.totale_casi), color='orange', alpha=0.8)
    plt.ylabel('number of cases')
    ax1.label_outer()
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.set_title("COVID-19 deaths in Italy per day")
    plt.bar(x=df_italy.data[1:], height=np.diff(df_italy.deceduti), color='red', alpha=0.8)
    ax2.set_xticks(df_italy.data[1::3])
    plt.ylabel('number of deaths')
    fig_1.tight_layout()
    plt.show()

    df_italy.iloc[12, 9] = (df_italy.iloc[11, 9] + df_italy.iloc[13, 9]) / 2
    df_italy.iloc[19, 9] = (df_italy.iloc[18, 9] + df_italy.iloc[20, 9]) / 2
    df_italy.iloc[31, 9] = 8215

    df_regions = pd.read_csv(url_regions)
    df_regions['positive_test_share'] = df_regions.totale_casi / df_regions.tamponi
    df_regions['data'] = pd.to_datetime(df_regions['data'])
    df_lombardy = df_regions[df_regions.denominazione_regione == "Lombardia"].copy()
    df_lombardy.iloc[22, 0] = datetime.strptime("2020-03-17 18:00:00", '%Y-%m-%d %H:%M:%S')
    df_lombardy.iloc[12, 13] = (df_lombardy.iloc[11, 13] + df_lombardy.iloc[13, 13]) / 2
    df_lombardy.iloc[19, 13] = (df_lombardy.iloc[18, 13] + df_lombardy.iloc[20, 13]) / 2
    df_emrom = df_regions[df_regions.denominazione_regione == "Emilia Romagna"]
    df_piemonte = df_regions[df_regions.denominazione_regione == "Piemonte"]
    df_veneto = df_regions[df_regions.denominazione_regione == "Veneto"]

    df_world = pd.read_csv(url_world)
    df_world["data"] = pd.to_datetime(df_world.date)
    df_china = df_world[df_world.location == "China"].copy()

    df_china.iloc[23, 3] = (df_china.iloc[23, 3] + df_china.iloc[24, 3]) / 2
    df_china.iloc[23, 5] = df_china.iloc[22, 5] + df_china.iloc[23, 3]
    df_china.iloc[24, 3] = df_china.iloc[23, 3]
    df_china.iloc[24, 5] = df_china.iloc[23, 5] + df_china.iloc[24, 3]

    fig_2, _ = plt.subplots(2, 1)
    ax1 = plt.subplot(211)
    ax1.set_title("New cases of COVID-19 in China per day")
    plt.bar(x=df_china.data[1:], height=df_china.new_cases[1:], color='orange', alpha=0.8)
    plt.ylabel('number of cases')
    ax1.label_outer()
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.set_title("COVID-19 deaths in China per day")
    plt.bar(x=df_china.data[1:], height=df_china.new_deaths[1:], color='red', alpha=0.8)
    ax2.set_xticks(df_china.data[1::5])
    plt.ylabel('number of deaths')
    fig_1.tight_layout()
    plt.show()

    df_s_korea = pd.read_csv(url_s_korea)
    df_s_korea['positive_test_share'] = df_s_korea.confirmed / df_s_korea.test
    df_s_korea.rename(columns={"date": "report_date", "test": "tamponi", "confirmed": "totale_casi"}, inplace=True)

    def plot_positive_tests(df, region):

        df = df.copy()
        df['tamponi_daily'] = df['tamponi'].diff()
        df['new_cases_daily'] = df['totale_casi'].diff()
        plt.bar(x=df.report_date[1:], height=df.tamponi_daily[1:], color='green', alpha=0.8, label="daily new swabs")
        plt.bar(x=df.report_date[1:], height=df.new_cases_daily[1:], color='orange', alpha=0.8, label="daily new cases")
        plt.legend(loc='upper left', bbox_to_anchor=(0, 0.95))
        ax2 = plt.twinx()
        sns.lineplot(x='report_date', y='positive_test_share', data=df, label="cumulative share of positive cases")
        ax2.lines[0].set_linestyle("--")
        ax2.grid(None)
        ax2.set_xticks(df.report_date[1::7])
        ax2.set_title("COVID-19 testing dynamics ({})".format(region))
        plt.show()

    plot_positive_tests(df_italy, "Italy")
    plot_positive_tests(df_s_korea.iloc[3:,:], "South Korea")

    df_regions.sort_values(by='data', inplace=True)

    regions = list(set(list(df_regions["denominazione_regione"])))
    _ = sns.lineplot(x='data', y='positive_test_share',
                 data=df_regions[df_regions["denominazione_regione"].isin(["Lombardia", "Piemonte", "Emilia Romagna",
                                                                           "Veneto", "Marche"])],
                 style="denominazione_regione", hue="denominazione_regione", dashes=False, markers=True)
    _.set_title("Cumulative share of positive tests by Italian region")
    plt.axes().set_xticks(df_italy.data[1::4])
    plt.show()
    _ = sns.lineplot(x='data', y='positive_test_share',
                 data=df_regions[df_regions["denominazione_regione"].isin(["Friuli Venezia Giulia", "Liguria",
                                                                           "Valle d'Aosta", "P.A. Trento"])],
                 style="denominazione_regione", hue="denominazione_regione", dashes=False, markers=True)
    _.set_title("Cumulative share of positive tests by Italian region")
    plt.axes().set_xticks(df_italy.data[1::4])
    plt.show()
    _ = sns.lineplot(x='data', y='positive_test_share',
                 data=df_regions[df_regions["denominazione_regione"].isin(["Toscana", "Umbria",
                                                                           "Lazio", "Abruzzo"])],
                 style="denominazione_regione", hue="denominazione_regione", dashes=False, markers=True,)
    _.set_title("Cumulative share of positive tests by Italian region")
    plt.axes().set_xticks(df_italy.data[1::4])
    plt.show()
    _ = sns.lineplot(x='data', y='positive_test_share',
                 data=df_regions[df_regions["denominazione_regione"].isin(["Campania", "Molise",
                                                                           "Basilicata", "Puglia"])],
                 style="denominazione_regione", hue="denominazione_regione", dashes=False, markers=True)
    _.set_title("Cumulative share of positive tests by Italian region")
    plt.axes().set_xticks(df_italy.data[1::4])
    plt.show()
    _ = sns.lineplot(x='data', y='positive_test_share',
                 data=df_regions[df_regions["denominazione_regione"].isin(["Calabria", "Sicilia",
                                                                           "Sardegna"])],
                 style="denominazione_regione", hue="denominazione_regione", dashes=False, markers=True)
    _.set_title("Cumulative share of positive tests by Italian region")
    plt.axes().set_xticks(df_italy.data[1::4])
    plt.show()

    df_deaths = \
        estimate_growth_model(
            df=df_italy,
            func=logistic_model_predefined_b,
            p0=[6, 10000],
            param_bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
            title='Training the logistic growth model on the Italian dataset (fixed b=90.7)',
            response_column='deceduti',
            start_point=0
        )

    df_deaths.to_csv('outbreak_predictions.csv')

    growth_model_sliding_data(
        df_data=df_italy,
        func=logistic_model,
        p0=[6, 88, 10000],
        param_bounds=([-np.inf, 85, -np.inf], [np.inf, 91, np.inf]),
        title='Logistic growth model moving window estimations for the number of deaths in Italy (85<=b<=91)',
        response_column='deceduti',
        moving=True,
        window=15
    )
