# coding=utf8

from __future__ import division

import sys
import os
import copy
import pickle
import datetime

import numpy as np
import scipy as sp
# import scipy.io as sio

import configuration
# import stats
# import analyze
from WindGen import import_M2
from cohda import uvic as cohda
from cohda import util


# Emulate Matlab's dot-style syntax for dicts
class DotDict(dict):
    def __init__(self, other={}):
        for k in other.keys():
            self[k] = copy.deepcopy(other[k])
    def __getattr__(self, attr):
        return self.get(attr, None)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def InitHeat(n, it, temperature):
    # Initializes Heat structure with thermodynamic parameters. Simulates full
    # period of operation to ensure load diversity for first time step
    # temperature and device states.

    delta = 1   # deadband
    T = 1 / 60  # time step
    T_set = 20  # setpoint temperature
    q_fan = 0.5 # fan power
    Ca = 2.5    # indoor air thermal mass share
    Cm = 7.5    # indoor contents thermall mass share
    Rao = 2     # building envelope resistance (outdoor air temperature <-> indoor thermal masses)
    Rma = 0.5   # heat transfer resistance (Ca <-> Cm)
    T_design = -5   # temperature design point for heatpump sizing
    # COP curve Parameters
    X = [x for x in range(-10, 20, 5)]
    Y = [2, 2.5, 3, 3.5, 3.9, 4.35]

    House = DotDict()

    # Deadband, Time step, and Number of Load entities
    House.N = n
    House.delta = delta
    House.T = T

    # Generic structure elements for all loads
    House.n = np.zeros((n, it))         # heatpump activation states (0, 1)
    House.n[:, 0] = np.around(np.random.rand(n))
    House.e = np.zeros((n, it))         # sampled end-use state comparison: T_a - T_s


    House.P0 = np.zeros(it)
    House.Pmin = np.zeros(it)
    House.Pmax = np.zeros(it)
    House.P_target = np.zeros(it)

    House.message_counter = np.zeros(it)


    # Load specific structure elements
    House.T_a = np.zeros((n, it))       # indoor air temperature
    House.T_b = np.zeros((n, it))       # indoor mass temperature
    House.P_r = np.zeros((n, it))       # total rated power of device (heatpump + fan)

    # Initialize temperatures for time step 1:
    House.T_a[:, 0] = (21 - 19) * np.random.rand(n) + 19
    House.T_b[:, 0] = House.T_a[:, 0]


    House.P = np.zeros(n)
    House.q_fan = np.random.normal(q_fan, 0.1, n)
    House.T_set = np.repeat(T_set, n)           # setpoint temperatures
    House.P_h = np.zeros(n)                     # rated power of heatpump component
    House.Ca = np.random.normal(Ca, 0.2, n)     # indoor air thermal mass share
    House.Cm = np.random.normal(Cm, 0.4, n)     # indoor contents thermall mass share
    House.Rao = np.random.normal(Rao, 0.05, n)  # building envelope resistance (outdoor air temperature <-> indoor thermal masses)
    House.Rma = np.random.normal(Rma, 0.02, n)  # heat transfer resistance (Ca <-> Cm)
    House.COP_curve = np.polyfit(X, Y, 2)       # COP curve

    House.oversize = np.random.normal(1.5, 0.2, n)  # oversizing parameters
    House.T_design = np.random.normal(T_design, 0.5, n) # temperature design points for heatpump sizing

    print 'Initialization'
    # Calculate heatpump sizes based on parameters above
    HeatPumpSizing(House)
    House_init = DotDict(other=House)

    # Calculate state- and input-transition matrices, Eq 4.2, 4.4
    Omega, Gamma = HeatPumpMatrixCalc(House)

    # Do some simulation steps to initialize sane values for T_a and T_b
    k = max(temperature.shape) #// 2
    progress = util.PBar(k * n).start()
    for i in range(k - 1):
        HeatPumpInit(House_init, temperature, i, progress, Omega, Gamma)
    progress.finish()
    print
    House.T_a[:, 0] = House_init.T_a[:, k - 1]
    House.T_b[:, 0] = House_init.T_b[:, k - 1]
    House.e[:, 0] = House_init.e[:, k - 1]
    House.n[:, 0] = House_init.n[:, k - 1]

    House_uncontrolled = DotDict(other=House)

    # return House_init, Omega, Gamma
    return House, Omega, Gamma, House_uncontrolled


def HeatPumpSizing(House):
    # This function sizes heat pumps for N houses based on the house design
    # temperature, the heat pump Coefficient of performance curve, the heat pump
    # oversizing factor (gamma), and assuming that the design temperature
    # setpoint is the same as the simulation temperature setpoint.

    # Verified complete & working 30 Oct 2013

    # allocation of constant terms needed from HouseStatic object
    k = House.N

    # Heat pump sizing
    for i in range(k):
        q_d = House.oversize[i] * ((House.T_set[i] - House.T_design[i]) / House.Rao[i])     # Eq 4.6
        eff_d = np.polyval(House.COP_curve, House.T_design[i])
        P_h = q_d / eff_d                                                               # Eq 4.8
        House.P_h[i] = np.around(2 * P_h) / 2                                                 # Rounding to nearest 0.5kW rating
        House.P[i] = House.P_h[i] + House.q_fan[i]                                      # Eq 4.11


def HeatPumpInit(House, temperature, it, progress, Omega, Gamma):
    eset = House.delta / 2

    House.e[:, it] = House.T_a[:, it] - House.T_set

    House.n[:, it] = np.logical_or(
        np.logical_and(House.n[:, it] == 0, House.e[:, it] <= -eset),
        np.logical_and(House.n[:, it] == 1, House.e[:, it] < eset)
    )

    for n in range(House.N):
        eff = np.polyval(House.COP_curve, House.T_a[n, it])   # heat pump efficiency, based on curve
        q_h = eff * House.P_h[n]                           # heat provided by heatpump, Eq 4.9
        q_dot = House.n[n, it] * (q_h + House.q_fan[n])       # heat provided by heatpump + fan, Eq 4.10
        House.P_r[n, it] = House.n[n, it] * House.P[n]       # Eq 4.11
        U = np.zeros(2)
        U[0] = temperature[it]                            # Eq 4.2
        U[1] = q_dot                                     # Eq 4.2
        theta = np.array([House.T_a[n, it], House.T_b[n, it]])  # Eq 4.2
        # e = np.std(temp)
        # e = np.random.normal(0, 0.1, (1, 2))

        theta_next = np.dot(Omega[n], theta) + np.dot(Gamma[n], U) # +e    # Eq 4.4
        House.T_a[n, it + 1] = theta_next[0]
        House.T_b[n, it + 1] = theta_next[1]

        progress.update()

    House.n[:, it + 1] = House.n[:, it]


def HeatPumpMatrixCalc(House):
    # Calculate the matrix coefficients of Eq 4.2
    Rma_Ca = 1 / (House.Rma * House.Ca)
    Rao_Ca = 1 / (House.Rao * House.Ca)
    Rma_Cm = 1 / (House.Rma * House.Cm)
    Ca = 1 / House.Ca

    A, B = np.zeros((House.N, 2, 2)), np.zeros((House.N, 2, 2))
    A[:, 0, 0] = -(Rma_Ca + Rao_Ca)
    A[:, 0, 1] = Rma_Ca
    A[:, 1, 0] = Rma_Cm
    A[:, 1, 1] = -Rma_Cm
    B[:, 0, 0] = Rao_Ca
    B[:, 0, 1] = Ca

    # Calculate state- and input-transition matrices of Eq 4.4
    T = House.T # time step
    Omega = [sp.linalg.expm(A[i] * T) for i in range(House.N)]
    Gamma = [sp.linalg.lstsq(A[i].T, (np.dot(B[i], Omega[i]) - B[i]).T)[0].T
             for i in range(House.N)]

    return Omega, Gamma


def COHDA_Interface(House, temperature, Target, Omega, Gamma, relative=False):
    # This program will run a full simulation based on the number of time steps
    # in the Enviro.air (Outdoor air temperature) data series. Wind power
    # generation is provided for the duration of the simulation in the Wind
    # input variable (kW).
    #
    # For each time-step, the program will determine the state (e) of each house
    # based on the temperature set-point and internal air temperature based on a
    # normalized scale from -1 to 1. This information is then sent to the
    # OptimizerCOHDA module, which determines if units can be switched on or off
    # based on this state data and the limitations of our control authority to
    # 1/2 the deadband width. The results of the possible states for each house,
    # along with a probability factor derived from the current state of the
    # heat pump (n=1,0) and the state (e) of the house to discourage rapid
    # cycling of units. This data is summarized into a vector, Z, which is sent
    # to the COHDA algorithm using a python command line call. The results of
    # the COHDA algorithm are then returned to this program, and used to set the
    # heat pump state for the next time step (n=1,0). Then the thermodynamic
    # module is called to determine the air and building mass temperatures after
    # the time step is completed.
    #
    # Finally the Grid module is run to summarize
    # the power system status at each time step, making up defecits in renewable
    # generation with a conventional generator operating as a slack bus (if
    # excess renewable energy is generated, it can also be curtailed at this
    # slack bus.
    #
    # The program then iterates for k iterations to evaluate the simulation for
    # the desired time period.
    #
    # Inputs:
    #        House - structure with home thermodynamic data
    #        Power - structure with power generation placeholders
    #        Enviro - Outdoor air temperature [*C]
    #        Wind - Wind generation data [kW]
    #
    # Outputs:
    #        House - structure with simulation results for individual houses
    #        Power - structure with simulation results for agglomerated power

    eset1 = House.delta / 2
    k = max(temperature.shape)

    # Check target for it=0, assign u for it=0
    House.e[:, 0] = House.T_a[:, 0] - House.T_set
    House.n[:, 0] = np.logical_or(
        np.logical_and(House.n[:, 0] == 0, House.e[:, 0] <= -eset1),
        np.logical_and(House.n[:, 0] == 1, House.e[:, 0] <= eset1)
    )

    for it in range(k):
        # House.n[:, it] = np.logical_or(
        #     np.logical_and(House.n[:, it] == 0, House.e[:, it] <= -eset1),
        #     np.logical_and(House.n[:, it] == 1, House.e[:, it] <= eset1)
        # )
        House = HeatPump(House, temperature, it, Omega, Gamma)
        House.e[:, it + 1] = House.T_a[:, it + 1] - House.T_set
        # Define target for total power - running average.
        House, Pmin, Pmax = OptimizerCOHDA(House, Target, it, relative=relative)

        Prob = np.zeros(House.N) #((min(max(House.e[:, it], -1), 1)) + 1) / 2
        # Prob = np.logical_or(
        #     np.logical_and(House.n[:, it] == 1, Prob),
        #     np.logical_and(House.n[:, it] == 0, 1 - Prob) # probability of discarding change in state
        # )

        seed = 0
        target = House.P_target[it + 1]
        states, opt_w = {}, {}
        for uid in range(House.N):
            states[uid] = House.n[uid, it] * House.P[uid] # Current Power of unit
            opt_w[uid] = map(float, [Pmin[uid], Pmax[uid]])
        # states = {uid: House.n[uid, it] * House.P[uid] for uid in range(House.N)}
        # opt_w = {uid: map(float, [Pmin[uid], Pmax[uid]]) for uid in range(House.N)}
        stats = cohda.run(seed, target, states, Prob, opt_w)

        House.n[:, it + 1] = np.array(stats.solution.values()) / House.P
        House.message_counter[it + 1] = stats.message_counter

    # Simulate last time step
    House = HeatPump(House, temperature, k, Omega, Gamma)

    print

    return House


def HeatPump(House, temperature, it, Omega, Gamma, progress=None):
    # Equivalent thermodynamic parameter model for houses using air source heat
    # pumps. Considers both building air temperature, and building mass
    # temperature for more accurate thermal modeling. Error term, e() accounts
    # for solar radiation and other random thermal effects (currently disabled).

    for n in range(House.N):
        eff = np.polyval(House.COP_curve, House.T_a[n, it])   # heat pump efficiency, based on curve
        q_h = eff * House.P_h[n]                           # heat provided by heatpump, Eq 4.9
        q_dot = House.n[n, it] * (q_h + House.q_fan[n])       # heat provided by heatpump + fan, Eq 4.10
        House.P_r[n, it] = House.n[n, it] * House.P[n]       # Eq 4.11

        if it < House.P_r.shape[-1] - 1:
            U = np.zeros(2)
            U[0] = temperature[it]                            # Eq 4.2
            U[1] = q_dot                                     # Eq 4.2
            theta = np.array([House.T_a[n, it], House.T_b[n, it]])  # Eq 4.2
            # e = np.std(temp)
            # e = np.random.normal(0, 0.1, (1, 2))

            theta_next = np.dot(Omega[n], theta) + np.dot(Gamma[n], U) # +e    # Eq 4.4
            House.T_a[n, it + 1] = theta_next[0]
            House.T_b[n, it + 1] = theta_next[1]

        if progress is not None:
            progress.update()

    # Initially set the next time steps heat pump state based on persistence
    # model. The controller can override these settings to achieve it's target
    if it < House.P_r.shape[-1] - 1:
        House.n[:, it + 1] = House.n[:, it]

    return House


def HeatPump_uncontrolled(House, temperature, it, Omega, Gamma, progress=None):
    # Equivalent thermodynamic parameter model for houses using air source heat
    # pumps. Considers both building air temperature, and building mass
    # temperature for more accurate thermal modeling. Error term, e() accounts
    # for solar radiation and other random thermal effects (currently disabled).

    for n in range(House.N):
        eff = np.polyval(House.COP_curve, House.T_a[n, it])   # heat pump efficiency, based on curve
        q_h = eff * House.P_h[n]                           # heat provided by heatpump, Eq 4.9
        q_dot = House.n[n, it] * (q_h + House.q_fan[n])       # heat provided by heatpump + fan, Eq 4.10
        House.P_r[n, it] = House.n[n, it] * House.P[n]       # Eq 4.11

        if it < House.P_r.shape[-1] - 1:
            U = np.zeros(2)
            U[0] = temperature[it]                            # Eq 4.2
            U[1] = q_dot                                     # Eq 4.2
            theta = np.array([House.T_a[n, it], House.T_b[n, it]])  # Eq 4.2
            # e = np.std(temp)
            # e = np.random.normal(0, 0.1, (1, 2))

            theta_next = np.dot(Omega[n], theta) + np.dot(Gamma[n], U) # +e    # Eq 4.4
            House.T_a[n, it + 1] = theta_next[0]
            House.T_b[n, it + 1] = theta_next[1]

        if progress is not None:
            progress.update()

    if it < House.P_r.shape[-1] - 1:
        House.e[:, it + 1] = House.T_a[:, it + 1] - House.T_set
        eset1 = House.delta / 2
        House.n[:, it + 1] = np.logical_or(
            # Devices that are off, but have to be switched on
            np.logical_and(House.n[:, it] == 0, House.e[:, it + 1] <= -eset1),
            # Devices that are on, and are allowed to stay on
            np.logical_and(House.n[:, it] == 1, House.e[:, it + 1] <= eset1)
        )

    return House


def OptimizerCOHDA(House, Target, it, relative=False):
    # This function determines the possible states that each heat pump are
    # feasible given our user-defined temperature constraints - 1/2 the deadband
    # range. It also calculates a target power for the responsive load
    # population using a 2 minute moving average, and verifies that this target
    # falls within the range of feasible values for the load population (and
    # alters the target to an appropriate value at the upper or lower bound if
    # it is outside the range of feasible values for the load population at the
    # current timestep).

    # House.e[:, it + 1] = House.T_a[:, it + 1] - House.T_set   # already set outside this function
    eset1 = House.delta / 2

    # From Parkinson et al., "Comfort-constrained distributed heat pump management":
    # Accurately scheduling the heat pump load in real-time
    # will initially require determination of the control signal
    # (set-point change) corresponding to the objective. To
    # prevent customer-side disruption, set-point modulations are
    # constrained to remain within the quarter-deadband width, as
    # it is unlikely individuals will notice changes of this
    # magnitude at end-use.
    # |u(k)| ≤ δ/4
    # Constraining set-point changes to this magnitude provides
    # further benefits, as it will only involve loads traversing the
    # final quarter-trajectory of their current operating state,
    # thereby preventing rapid-cycling.
    u = House.delta / 4     # Eq 3.22

    # Max Power ratings for each unit for COHDA:
    # Collect devices that are *allowed* to be on
    Pmax = np.logical_or(
        # Devices that are off, but may be switched on
        np.logical_and(House.n[:, it + 1] == 0, House.e[:, it + 1] <= (-eset1 + u)),
        # Devices that are on, and are allowed to stay on
        np.logical_and(House.n[:, it + 1] == 1, House.e[:, it + 1] <= (eset1 + u))
    ) * House.P
    House.Pmax[it + 1] = np.sum(Pmax)

    # Min Power ratings for each unit for COHDA:
    # Collect devices that *must* be on
    Pmin = np.logical_or(
        # Devices that are off, but have to be switched on
        np.logical_and(House.n[:, it + 1] == 0, House.e[:, it + 1] <= (-eset1 - u)),
        # Devices that are on, and have to stay on
        np.logical_and(House.n[:, it + 1] == 1, House.e[:, it + 1] <= (eset1 - u))
    ) * House.P
    House.Pmin[it + 1] = np.sum(Pmin)

    # Uncontrolled responsive load trajectory (uncontrolled capacity factor P0)
    # u = 0                   # Eq 3.25
    # House.P0[it + 1] = np.sum(np.logical_or(
    #     # Devices that are off, but have to be switched on
    #     np.logical_and(House.n[:, it + 1] == 0, House.e[:, it + 1] <= -eset1),
    #     # Devices that are on, and are allowed to stay on
    #     np.logical_and(House.n[:, it + 1] == 1, House.e[:, it + 1] <= eset1)
    # ) * House.P)

    # Target power output from responsive load community
    # House.P_target[it + 1] = House.P0[it] + Target[it + 1]
    if relative:
        House.P_target[it + 1] = House.P_r[:, it].sum(0) + Target[it + 1]
    else:
        House.P_target[it + 1] = Target[it + 1]

    # print new line every simulated hour
    hr = it / 60 + 1
    if it == 0:
        print 'Simulation progress, splitted by hour:'
        print '(< means target below feasible range, > means above, . is ok)'
        print ('%2d: ' % hr),
    elif it % 60 == 0:
        print ('\n%2d: ' % hr),

    # Verify target is feasible, and modify if not.
    if House.P_target[it + 1] < House.Pmin[it + 1]:
        print '<',
        House.P_target[it + 1] = House.Pmin[it + 1]
    elif House.P_target[it + 1] > House.Pmax[it + 1]:
        print '>',
        House.P_target[it + 1] = House.Pmax[it + 1]
    else:
        print '.', # simply to show that the program is progressing correctly.

    sys.stdout.flush()

    return House, Pmin, Pmax


def main(cfg):
    # This is the main script for running the COHDA optimizer on a given set of
    # scenario data files (demand response program attempt to follow a wind
    # generation profile).

    cfg.ts_start = datetime.datetime.now()#.isoformat()

    np.random.seed(cfg.seed)

    # Enviro.mat - Comment from Adam:
    # The Enviro.mat data was simply an arbitrarily scaled temperature profile from
    # the NREL Met 2 wind tower - I don't know the exact dates, and the absolute
    # temperature values were scaled into a more useful temperature range to
    # represent the Pacific Northwest (In subsequent simulations I have used raw
    # temperature data from the NREL wind tower without any scaling - but at the
    # time we were working I just used a representative day and scaled it as needed
    # to represent the average temperature desired, without changing the variation
    # of temperature. I can provide new Enviro.mat data if you'd rather run the
    # simulations again with a specific dates temperature info, or you could
    # replace the file with any 1-minute resolution temperature data (*C) you have
    # available).
    # Enviro = sio.loadmat('Enviro.mat')['Enviro']

    # UnresponsiveLoad.mat - Comment from Adam:
    # The Unresponsive_Load.mat data was provided by Simon Parkinson, and was based
    # on a single substation feeder in the electrical grid. I believe the data was
    # generated from GridLab-D to represent a load community in the Pacific
    # NorthWest of North America again. I actually don't use this data for any of
    # my thesis simulations, as I've settled on a more direct method of evaluating
    # system performance. This unresponsive load data is used by the 'Grid' module
    # to determine how much additional generation is required to satisfy community
    # loads with or without demand response.
    # UnresponsiveLoad = sio.loadmat('UnresponsiveLoad.mat')

    # Wind.mat - Comment from Adam:
    # The Wind.mat data is wind generation from a wind turbine (I believe I used a
    # 3 kW turbine when building this data) based on environmental wind speed data.
    # The wind speed data was from the same dates as the Enviro.mat temperature
    # data, and unscaled. Again, I have been using NREL Met 2 wind tower data to
    # produce wind generation profiles for my thesis, but any 1-minute resolution
    # wind speed data (m/s) would be suitable to produce the wind power profile
    # using the attached script (Simple conversion of mechanical energy extracted
    # from wind by a turbine converted to electrical power via a gearbox and fixed
    # electrical conversion efficiency).
    # Wind = sio.loadmat('Wind.mat')

    # n = number of units. Comment from Adam:
    # In this scenario we used a population of responsive heat pumps as the
    # participating loads in the demand response scenario, as you stated above. We
    # used a population of 100 homes, mostly due to computer processing time, as
    # larger populations required significantly more time to run.
    n = cfg.n

    # it = number of simulation steps.
    it = cfg.it

    # scale_w = n / 2000
    # scale_u = -5 * n / 2000
    scale_w = n / 1500
    # scale_u = -1 * n / 1500

    # Feed-in is positive power, Load is negative power

    # Enviro
    # http://www.nrel.gov/midc/nwtc_m2/
    datetimes, temperature, windspeed, windpower = import_M2(cfg.enviro, it=it - 1)
    # windspeed_2 = (windspeed + np.random.normal(windspeed.mean(), windspeed.std() / 5, len(windspeed))) / 1.6
    # windpower_2 = WindGen(windspeed_2, it=it - 1)
    # windpower = windpower + windpower_2

    # w = Wind['Wind'][0] * scale_w
    w = np.empty(it)
    # w[1:] = np.load('M2_windspeed_80m_2014-04-01.npy') * scale_w
    w[1:] = windpower * scale_w
    w[0] = w[1]

    # Target
    # u = UnresponsiveLoad['P_U'][0] * scale_u
    # Residual = w + u
    # Set target as negative residual load
    # Target = -1 * Residual
    # Set target as negative wind power
    # Target = -1 * w
    # Set target as negative wind power fluctuation
    Target = np.zeros(it)
    # # Target[1:] = -1 * (w[1:] - w[:-1])
    Target[1 + cfg.lag:] = (-1 * (w[1:] - w[:-1]))[:it - 1 - cfg.lag]

    # plt.plot(w, label='Wind')
    # # plt.plot(u, label='Unresponsive Load')
    # # plt.plot(Residual, label='Residual Load')
    # plt.plot(Target, label='Target')
    # plt.legend()
    # plt.show()
    # sys.exit(0)

    # Initialize the data structures for the simulation.
    # (See the documentation of the respective function.)
    House, Omega, Gamma, House_uncontrolled = InitHeat(n, it, temperature)

    # Run simulation for the uncontrolled reference case
    print 'Simulate uncontrolled reference case'
    k = max(temperature.shape) + 1
    progress = util.PBar(k * n).start()
    for i in range(k):
        HeatPump_uncontrolled(House_uncontrolled, temperature, i, Omega, Gamma, progress=progress)
    progress.finish()
    print

    # Run the simulation with the COHDA optimizer.
    # The simulation interprets load as positive power, so scale by -1.
    print 'COHDA'
    House = COHDA_Interface(House, temperature, -1 * Target, Omega, Gamma, relative=True)


    # Store the results
    cfg.House = dict(House)
    cfg.House_uncontrolled = dict(House_uncontrolled)
    cfg.w = w
    cfg.Target = Target
    cfg.ts_end = datetime.datetime.now()#.isoformat()
    return cfg

    # fn1 = str(os.path.join(basepath, '.'.join(('House', ts, 'pickle'))))
    # fn2 = str(os.path.join(basepath, '.'.join(('House_uncontrolled', ts, 'pickle'))))
    # fn3 = str(os.path.join(basepath, '.'.join(('w', ts, 'npy'))))
    # with open(fn1, 'w') as f:
    #     pickle.dump(dict(House), f)
    # with open(fn2, 'w') as f:
    #     pickle.dump(dict(House_uncontrolled), f)
    # np.save(fn3, w)

    # return fn1, fn2, fn3


if __name__ == '__main__':
    cfg = main(configuration.Configuration(n=1500, it=1441, lag=0))
    fn = str(os.path.join(cfg.basepath, '.'.join(
            ('cfg', cfg.title, str(cfg.seed), 'pickle'))))
    with open(fn, 'w') as f:
        pickle.dump(cfg, f)

    # fn1, fn2, fn3 = main()
    # with open(fn1) as f:
    #     d1 = pickle.load(f)
    # with open(fn2) as f:
    #     d2 = pickle.load(f)
    # d3 = np.load(fn3)
    # analyze.plot(DotDict(other=d1), DotDict(other=d2), d3)
