# coding = utf8

from __future__ import division

import sys

import numpy as np
import scipy as sp
import scipy.io as sio
import matplotlib.pyplot as plt

from cohda import uvic as cohda
from cohda import util


# Emulate Matlab's dot-style syntax for dicts
class DotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr, None)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def InitHeat(n, it, Enviro):
    # Initializes Heat structure with thermodynamic parameters. Simulates full
    # period of operation to ensure load diversity for first time step
    # temperature and device states.

    delta = 1
    T = 1 / 60
    T_set = 20
    q_fan = 0.5
    Ca = 2.5
    Cm = 7.5
    Rao = 2
    Rma = 0.5
    T_design = -5
    # COP curve Parameters
    X = [x for x in range(-10, 20, 5)]
    Y = [2, 2.5, 3, 3.5, 3.9, 4.35]

    Load2 = DotDict()

    # Deadband, Time step, and Number of Load entities
    Load2.N = n
    Load2.delta = delta
    Load2.T = T

    # Generic structure elements for all loads
    Load2.n = np.zeros((n, it))
    Load2.n[:, 0] = np.around(np.random.rand(n))
    Load2.e = np.zeros((n, it))


    Load2.P0 = np.zeros(it)
    Load2.Pmin = np.zeros(it)
    Load2.Pmax = np.zeros(it)
    Load2.P_target = np.zeros(it)
    Load2.u = np.zeros(it)
    Load2.P_total = np.zeros(it)
    Load2.M = np.zeros((n, it))
    Load2.ms = np.zeros((1000, it))
    Load2.x = np.zeros((1000, it))
    Load2.m_s = np.zeros(it)
    Load2.cf = np.zeros((1000, it))


    # Load specific structure elements
    Load2.T_a = np.zeros((n, it))
    Load2.T_b = np.zeros((n, it))
    Load2.P_r = np.zeros((n, it))
    Load2.T_a[:, 0] = (21 - 19) * np.random.rand(n) + 19   # Initialize temperatures for time step 1.
    Load2.T_b[:, 0] = Load2.T_a[:, 0]



    Load2.P = np.zeros(n)
    Load2.q_fan = np.random.normal(q_fan, 0.1, n)
    # Load2.T_set = np.random.normal(T_set, 0, n)
    Load2.T_set = np.repeat(T_set, n)
    Load2.P_h = np.zeros(n)
    Load2.Ca = np.random.normal(Ca, 0.2, n)
    Load2.Cm = np.random.normal(Cm, 0.4, n)
    Load2.Rao = np.random.normal(Rao, 0.05, n)
    Load2.Rma = np.random.normal(Rma, 0.02, n)
    Load2.COP_curve = np.polyfit(X, Y, 2)

    Load2.oversize = np.random.normal(1.5, 0.2, n)
    Load2.T_design = np.random.normal(T_design, 0.5, n)
    HeatPumpSizing(Load2)
    TempLoad = Load2

    print 'Initialization'
    k = max(Enviro['air'][0][0][0].shape)
    progress = util.PBar(k * n).start()
    for i in range(k - 1):
        TempLoad = HeatPumpInit(TempLoad, Enviro, i, progress)
    progress.finish()
    print

    Load2.T_a[:, 0] = TempLoad.T_a[:, k - 1]
    Load2.T_b[:, 0] = TempLoad.T_a[:, k - 1]
    Load2.e[:, 0] = TempLoad.e[:, k - 1]
    Load2.n[:, 0] = TempLoad.n[:, k - 1]

    return Load2


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


def HeatPumpInit(House, Enviro, it, progress):
    eset = House.delta / 2

    House.e[:, it] = ((House.T_a[:, it] - House.T_set[:]) / House.delta)

    House.n[:, it] = np.logical_or(
        np.logical_and((House.n[:, it] == 0), (House.e[:, it] <= (-eset + House.u[it]))),
        np.logical_and((House.n[:, it] == 1), (House.e[:, it] < (eset + House.u[it])))
    )

    for n in range(House.N):
        eff = np.polyval(House.COP_curve, House.T_a[n, it])   # heat pump efficiency, based on curve
        q_h = eff * House.P_h[n]                           # Eq 4.9
        q_dot = House.n[n, it] * (q_h + House.q_fan[n])       # Eq 4.10
        House.P_r[n, it] = House.n[n, it] * House.P[n]       # Eq 4.11
        U = np.zeros(2)
        U[0] = Enviro['air'][0][0][0][it]                            # Eq 4.2
        U[1] = q_dot                                     # Eq 4.2
        temp = np.zeros((2, 2))
        temp[0, 0] = House.T_a[n, it]                      # Eq 4.2
        temp[1, 0] = House.T_b[n, it]                      # Eq 4.2
        # e = np.std(temp)
        # e = np.random.normal(0, 0.1, (1, 2))
        Omega, Gamma_b = HeatPumpMatrixCalc(House, n)  # Eq 4.2, 4.4

        temp[:, 1] = np.dot(Omega, temp[:, 0]) + np.dot(Gamma_b, U[:]) # +e[:]    # Eq 4.4
        House.T_a[n, it + 1] = temp[0, 1]
        House.T_b[n, it + 1] = temp[1, 1]

        progress.update()

    House.n[:, it + 1] = House.n[:, it]

    return House


def HeatPumpMatrixCalc(House, n):
    # Function to calculate the matrix coefficients of Eq 4.2 for use in
    # calculations to determine omega and gamma for Eq 4.4

    # Fully functional verified 30 Oct 2013


    Rma = House.Rma[n]
    Rao = House.Rao[n]
    Ca = House.Ca[n]
    Cm = House.Cm[n]
    T = House.T

    A = np.zeros((2, 2))
    A[0, 0] = -((1 / (Rma * Ca)) + (1 / (Rao * Ca)))
    A[0, 1] = (1 / (Rma * Ca))
    A[1, 0] = (1 / (Rma * Cm))
    A[1, 1] = -(1 / (Rma * Cm))

    B = np.zeros((2, 2))
    B[0, 0] = (1 / (Rao * Ca))
    B[0, 1] = (1 / Ca)

    Omega = sp.linalg.expm(np.dot(A, T))

    X = (np.dot(B, Omega) - np.dot(B, sp.linalg.expm(A * 0)))
    Gamma_b = sp.linalg.lstsq(A.T, X.T)[0].T

    return Omega, Gamma_b


def InitPower(n, it):
    # Initializes Power structure for simulations. Must manually add
    # unresponsive load component from input datasets (Power.P_U).
    Power = DotDict()
    Power.Prob = np.zeros((n, it))
    Power.P_L = np.zeros(it)
    Power.P_U = np.zeros(it)
    Power.P_R = np.zeros(it)
    Power.P = np.zeros(it)
    Power.P_T = np.zeros(it)
    Power.HeatPumps = np.zeros(it)
    Power.PHEV = np.zeros(it)
    Power.P_G = np.zeros(it)
    Power.Wind = np.zeros(it)
    Power.P_C = np.zeros(it)
    Power.exitflag = np.zeros(it) # Optimizer ex itflag
    Power.Grid = np.zeros(it)     # Grid optimizer ex itflag
    Power.temp = np.zeros(it)
    Power.P_G1 = np.zeros(it)
    Power.P_G2 = np.zeros(it)

    return Power


def COHDA_Interface(House, Power, Enviro, Target):
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

    # Power.Wind = Wind;
    eset1 = House.delta / 2
    k = max(Enviro['air'][0][0][0].shape)

    # Check target for it=0, assign u for it=0
    House.e[:, 0] = ((House.T_a[:, 0] - House.T_set) / House.delta)
    House.n[:, 0] = np.logical_or(
        np.logical_and((House.n[:, 0] == 0), (House.e[:, 0] <= (-eset1 + House.u[0]))),
        np.logical_and((House.n[:, 0] == 1), (House.e[:, 0] <= (eset1 + House.u[0])))
    )

    for it in range(k - 1):
        # House.n[:, it] = np.logical_or(
        #     np.logical_and((House.n[:, it] == 0), (House.e[:, it] <= (-eset1 + House.u[it]))),
        #     np.logical_and((House.n[:, it] == 1), (House.e[:, it] <= (eset1 + House.u[it])))
        # )
        House = HeatPump(House, Enviro, it)
        Power.HeatPumps[it] = House.P_r[:, it].sum()
        House.e[:, it + 1] = (House.T_a[:, it + 1] - House.T_set) / House.delta
        # Define target for total power - running average.
        House, Power, A, B = OptimizerCOHDA(House, Power, Target, it)

        Prob = np.zeros(House.N) #((min(max(House.e[:, it], -1), 1)) + 1) / 2
        # Prob = np.logical_or(
        #     np.logical_and(House.n[:, it] == 1, Prob),
        #     np.logical_and(House.n[:, it] == 0, 1 - Prob) # probability of discarding change in state
        # )

        seed = 0
        target = House.P_target[it + 1]
        states = {uid: House.n[uid, it] * House.P[uid] for uid in range(House.N)} # Current Power of unit
        opt_w = {uid: map(float, [A[uid], B[uid]]) for uid in range(House.N)}
        uid, Pnew = cohda.run(seed, target, states, Prob, opt_w)

        House.n[:, it + 1] = np.array(Pnew.values()) / House.P[:]

        # # COHDA:
        # #   first line: seed
        # #   second line: target
        # #   remaining lines: [id, sol_init, p_refuse, [feasible states ...]]
        # COHDA = np.zeros((House.N + 2, 5))
        # COHDA[0, :] = [0, -1, -1, -1, -1]   # always zero
        # COHDA[1, :] = [House.P_target[it + 1], -1, -1, -1, -1]
        # COHDA[2 : House.N + 2, 0] = np.arange(0, House.N).T
        # COHDA[2 : House.N + 2, 1] = House.n[:, it] * House.P[:] # Current Power of unit
        # COHDA[2 : House.N + 2, 2] = Prob   # Probability of discarding a change of state - prevents rapid cycling
        # COHDA[2 : House.N + 2, 3] = A     # Minimum power rating of unit
        # COHDA[2 : House.N + 2, 4] = B     # Maximum power rating of unit

        # csvwrite(outfile, COHDA)
        # system(cmd)
        # N = csvread(resultfile)
        # House.n[:, it + 1] = N[:, 1] / House.P[:]


        # Power = Grid(Power, it)
    print

    return House, Power


def HeatPump(House, Enviro, it):
    # Equivalent thermodynamic parameter model for houses using air source heat
    # pumps. Considers both building air temperature, and building mass
    # temperature for more accurate thermal modeling. Error term, e() accounts
    # for solar radiation and other random thermal effects (currently disabled).

    for n in range(House.N):
        eff = np.polyval(House.COP_curve, House.T_a[n, it])   # heat pump efficiency, based on curve
        q_h = eff * House.P_h[n]                           # Eq 4.9
        q_dot = House.n[n, it] * (q_h + House.q_fan[n])       # Eq 4.10
        House.P_r[n, it] = House.n[n, it] * House.P[n]
        U = np.zeros(2)
        U[0] = Enviro['air'][0][0][0][it]
        U[1] = q_dot                                     # Eq 4.2
        temp = np.zeros((2, 2))
        temp[0, 0] = House.T_a[n, it]                      # Eq 4.2
        temp[1, 0] = House.T_b[n, it]                      # Eq 4.2
        # e = np.std(temp)
        # e = np.random.normal(0, 0.1, 1,2)
        Omega, Gamma_b = HeatPumpMatrixCalc(House, n)  # Eq 4.2, 4.4


        temp[:, 1] = np.dot(Omega, temp[:, 0]) + np.dot(Gamma_b, U[:]) # +e[:]    # Eq 4.4
        House.T_a[n, it + 1] = temp[0, 1]
        House.T_b[n, it + 1] = temp[1, 1]

    # Initially set the next time steps heat pump state based on persistence
    # model. The controller can override these settings to achieve it's target
    House.n[:, it + 1] = House.n[:, it]

    return House


def OptimizerCOHDA(Load1, Power, Target, it):
    # This function determines the possible states that each heat pump are
    # feasible given our user-defined temperature constraints - 1/2 the deadband
    # range. It also calculates a target power for the responsive load
    # population using a 2 minute moving average, and verifies that this target
    # falls within the range of feasible values for the load population (and
    # alters the target to an appropriate value at the upper or lower bound if
    # it is outside the range of feasible values for the load population at the
    # current timestep).


    eset1 = Load1.delta / 2
    u1 = Load1.delta / 4

    P1max = np.sum(
        np.logical_and(Load1.n[:, it + 1] == 0, Load1.e[:, it + 1] <= (-eset1 + u1)) * Load1.P[:] +
        np.logical_and(Load1.n[:, it + 1] == 1, Load1.e[:, it + 1] <= (eset1 + u1)) * Load1.P[:]
    )
    Load1.Pmax[it + 1] = P1max
    # Max Power ratings for each unit for COHDA
    B = (
        np.logical_and(Load1.n[:, it + 1] == 0, Load1.e[:, it + 1] <= (-eset1 + u1)) * Load1.P[:] +
        np.logical_and(Load1.n[:, it + 1] == 1, Load1.e[:, it + 1] <= (eset1 + u1)) * Load1.P[:]
    )
    u1 = -u1
    P1min = np.sum(
        np.logical_and(Load1.n[:, it + 1] == 0, Load1.e[:, it + 1] <= (-eset1 + u1)) * Load1.P[:] +
        np.logical_and(Load1.n[:, it + 1] == 1, Load1.e[:, it + 1] <= (eset1 + u1)) * Load1.P[:]
    )
    Load1.Pmin[it + 1] = P1min
    # Min Power ratings for each unit for COHDA
    A = (
        np.logical_and(Load1.n[:, it + 1] == 0, Load1.e[:, it + 1] <= (-eset1 + u1)) * Load1.P[:] +
        np.logical_and(Load1.n[:, it + 1] == 1, Load1.e[:, it + 1] <= (eset1 + u1)) * Load1.P[:]
    )

    u = 0
    P10 = np.sum(
        np.logical_and(Load1.n[:, it + 1] == 0, Load1.e[:, it + 1] <= (-eset1 + u)) * Load1.P[:] +
        np.logical_and(Load1.n[:, it + 1] == 1, Load1.e[:, it + 1] <= (eset1 + u)) * Load1.P[:]
    )
    Load1.P0[it + 1] = P10

    P1max = Load1.Pmax[it]
    P1min = Load1.Pmin[it]
    P10 = Load1.P0[it]

    # Target power output from responsive load community
    Power.P_R[it + 1] = P10 + Target[it + 1]
    Power.P_T[it + 1] = Power.P_R[it + 1]

    # print new line every simulated hour
    hr = (it - 1) / 60 + 1
    if it == 1:
        print 'Simulation progress, splitted by hour:'
        print '(< means target below PTmin, > means above, . is ok)'
        print ('%2d: ' % hr),
    elif it % 60 == 1:
        print ('\n%2d: ' % hr),

    # Verify target is feasible, and modify if not.
    PTmin = P1min
    PTmax = P1max
    if Power.P_T[it + 1] < PTmin:
        print '<',
        Power.P_T[it + 1] = PTmin
    elif Power.P_T[it + 1] > PTmax:
        print '>',
        Power.P_T[it + 1] = PTmax
    else:
        print '.', # simply to show that the program is progressing correctly.

    sys.stdout.flush()

    Load1.P_target[it + 1] = Power.P_T[it + 1]

    return Load1, Power, A, B


if __name__ == '__main__':
    # This is the main script for running the COHDA optimizer on a given set of
    # scenario data files (demand response program attempt to follow a wind
    # generation profile).

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
    Enviro = sio.loadmat('Enviro.mat')['Enviro']

    # UnresponsiveLoad.mat - Comment from Adam:
    # The Unresponsive_Load.mat data was provided by Simon Parkinson, and was based
    # on a single substation feeder in the electrical grid. I believe the data was
    # generated from GridLab-D to represent a load community in the Pacific
    # NorthWest of North America again. I actually don't use this data for any of
    # my thesis simulations, as I've settled on a more direct method of evaluating
    # system performance. This unresponsive load data is used by the 'Grid' module
    # to determine how much additional generation is required to satisfy community
    # loads with or without demand response.
    UnresponsiveLoad = sio.loadmat('UnresponsiveLoad.mat')

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
    Wind = sio.loadmat('Wind.mat')

    # n = number of units. Comment from Adam:
    # In this scenario we used a population of responsive heat pumps as the
    # participating loads in the demand response scenario, as you stated above. We
    # used a population of 100 homes, mostly due to computer processing time, as
    # larger populations required significantly more time to run.
    n = 100

    # it = number of simulation steps.
    it = 1441

    # Initialize the data structures for the simulation.
    # (See the documentation of the respective function.)
    Heat = InitHeat(n, it, Enviro)
    Power = InitPower(n, it)
    # Attach the unresponsive load to the power structure manually.
    # TODO: Make P_U a parameter for the InitPower() function?
    Power.P_U = UnresponsiveLoad['P_U']

    Target = Wind['Wind'][0] / 10

    # Run the simulation with the COHDA optimizer.
    print 'COHDA'
    House, Power = COHDA_Interface(Heat, Power, Enviro, Target)

    # Resample the results to 15 minute resolution
    def resample(d, resolution):
        return (d.reshape(d.shape[0]/resolution, resolution).sum(1)/resolution)

    Target_15 = resample(Target[1: it], 15)
    HPTarget = resample(House.P_target[1 : it] - House.P0[1 : it], 15)
    HPPower = resample(Power.HeatPumps[1: it] - House.P0[1 : it], 15)

    # Target_15 = np.mean(np.reshape(Target[1 : it], 15, size(Target[1 : it], 2)/15));
    # HPTarget = np.mean(np.reshape(House.P_target[1 : it] - House.P0[1 : it], 15, size(Target(2:it), 2)/15));
    # HPPower = np.mean(np.reshape(Power.HeatPumps[1 : it] - House.P0[1 : it], 15, size(Target(2:it), 2)/15));

    # Display the results
    plt.plot(Target_15, label='Target')
    plt.plot(HPTarget, label='Heat Pump Corrected Target')
    plt.plot(HPPower, label='Heat Pump Power Dispatched')
    plt.legend(loc='upper left')
    plt.show()
