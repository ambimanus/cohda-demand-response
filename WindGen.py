from __future__ import division

import numpy as np


def WindGen(Wind):
    # Calculated Electrical Energy Generated from a Wind Turbine according to
    # section 4.2.1 of Simon's thesis. Input wind speed in in [m/s].
    # Scale = 3 / 1.5      # Scaled turbine to 1.5 MW
    Scale = 1
    P_rt = 1.5           # Rated turbine power [MW]
    D = 82.5             # Rotor diameter [m]
    rho = 1.225          # air density [kg/m^3]
    cpm = 0.300          # Max coefficient of performance
    cpr = 0.250          # coefficient of performance at r
    u_co = 26            # cutout speed [m/s]
    u_ci = 4             # cut-in speed [m/s]
    u_r = 12             # Turbine rated speed [m/s]

    S = 3                # Stages of Gearbox
    n_e = 0.95           # Electrical System Efficiency

    F1 = 1.2521          # calculated empirical constant
    F2 = 0.7010          # calculated empirical constant
    u_m = 7              # Wind speed at maximum Cp [m/s]


    k = len(Wind)
    P_m = np.zeros(k)
    P = np.zeros(k)
    C_p = np.zeros(k)
    n_g = np.zeros(k)

    for i in range(k):
        if Wind[i] < u_ci:
            C_p[i] = 0

        if Wind[i] >= u_ci and Wind[i] <= u_r:
            C_p[i] = cpm * (
                1 - F1 * (((u_m / Wind[i]) - 1) ** 2) -
                    F2 * (((u_m / Wind[i]) - 1) ** 3)
            )

        if Wind[i] >= u_r and Wind[i] <= u_co:
            C_p[i] = cpr * ((u_r / Wind[i]) ** 3)

        if Wind[i] > u_co:
            C_p[i] = 0

        P_m[i] = (1 / 8) * C_p[i] * rho * np.pi * (D ** 2) * (Wind[i] ** 3)

        if P_m[i] <= (S * P_rt) / 90:
            n_g[i] = 0.1

        if P_m[i] > (S * P_rt) / 90:
            n_g[i] = 1 - ((0.01 * S * P_rt) / P_m[i])

        P[i] = Scale * n_e * n_g[i] * P_m[i] / 1000  # Report power generated in kW

    return P


def import_M2(fn, it=1440):
    assert it <= 1440
    import matplotlib.dates as mdates
    import datetime

    dd = ['object', 'object', 'float', 'float']
    data = np.genfromtxt(fn, delimiter=',', skip_header=1, dtype=dd)

    dim = data.shape[-1]
    x = np.empty(dim)
    for i in range(dim):
        date = datetime.datetime.strptime(data['f0'][i], '%m/%d/%Y').date()
        time = datetime.datetime.strptime(data['f1'][i], '%H:%M').time()
        combined = datetime.datetime.combine(date, time)
        x[i] = mdates.date2num(combined)


    lim_a, lim_b = dim - (3 * 1440), dim - (2 * 1440)

    datetimes = x[lim_a : lim_b]
    temperature = data['f2'][lim_a : lim_b]
    windspeed = data['f3'][lim_a : lim_b]
    windpower = WindGen(windspeed)

    return datetimes[:it], temperature[:it], windspeed[:it], windpower[:it]


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    # http://www.nrel.gov/midc/nwtc_m2/
    # http://www.nrel.gov/midc/apps/data_api.pl?site=NWTC&begin=20140301&end=20140331
    # http://www.nrel.gov/midc/apps/data_api.pl?site=NWTC&begin=20140329&end=20140329

    fn = sys.argv[1]
    datetimes, temperature, windspeed, windpower = import_M2(fn)
    # windspeed_2 = (windspeed + np.random.normal(windspeed.mean(), windspeed.std() / 5, len(windspeed))) / 1.6
    # windpower_2 = WindGen(windspeed_2)

    # import pdb
    # pdb.set_trace()

    plt.plot_date(datetimes, temperature, '-', label='Temperature [\\textdegree{}C]')
    plt.plot_date(datetimes, windspeed, '-', label='Wind Speed [m/s]')
    # plt.plot_date(datetimes, windspeed_2, '-', label='Wind Speed 2 [m/s]')
    plt.plot_date(datetimes, windpower / 1000, '-', label='Wind Power [MW]')
    # plt.plot_date(datetimes, windpower_2 / 1000, '-', label='Wind Power 2 [MW]')
    plt.legend()
    plt.show()
    sys.exit(0)

    np.save('.'.join((fn.split('.')[0], 'npy')), windpower)
