function [House, Power, A, B] = OptimizerCOHDA( House, Power, it)
    %This function determines the possible states that each heat pump are
    %feasible given our user-defined temperature constraints - 1/2 the deadband
    %range. It also calculates a target power for the responsive load
    %population using a 2 minute moving average, and verifies that this target
    %falls within the range of feasible values for the load population (and
    %alters the target to an appropriate value at the upper or lower bound if
    %it is outside the range of feasible values for the load population at the
    %current timestep).


    R=1000;             %Resolution of Programmable Thermostats.
    delta1=House.delta; %deadband width of load community

    N=House.N;
    Z=zeros(N,3);
    Z(:,1)=House.n(:,it);
    Z(:,2)=House.e(:,it);
    Z(:,3)=House.P(:,1);

    M=((Z(:,2).*R)./(2*delta1))+(R/2);
    ms=R/2;
    m_min=ms-(R/4);
    m_max=ms+(R/4);
    House.P0(it)=sum(((M<=m_min).*(Z(:,1)==0)).*Z(:,3)+(M<=m_max).*(Z(:,1)==1).*Z(:,3));

    ms=(3/8)*R;
    m_min=ms-(R/4);
    m_max=ms+(R/4);
    House.Pmin(it)=sum(((M<=m_min).*(Z(:,1)==0)).*Z(:,3)+((M<=m_max).*(Z(:,1)==1).*Z(:,3)));
    A=((M<=m_min).*(Z(:,1)==0)).*Z(:,3);%+((M<=m_max).*(Z(:,1)==1).*Z(:,3));

    ms=(5/8)*R;
    m_min=ms-(R/4);
    m_max=ms+(R/4);
    House.Pmax(it)=sum(((M<=m_min).*(Z(:,1)==0)).*Z(:,3)+(M<=m_max).*(Z(:,1)==1).*Z(:,3));
    B=(M<=m_max).*(Z(:,1)==1).*Z(:,3);

    P1max=House.Pmax(it);
    P1min=House.Pmin(it);
    P10=House.P0(it);
    PU=Power.P_U(it+1);     %Assumed to know this accurately using load forecasting
    PW=Power.Wind(it+1);    %Assumed to know this accurately using wind generation forecasting

    %Target power output from responsive load community
    Power.P_R(it)=0;
    if it>2
        PL1=Power.HeatPumps(it-2)+Power.P_U(it-2)-Power.Wind(it-2);
        PL2=Power.HeatPumps(it-1)+Power.P_U(it-1)-Power.Wind(it-1);
        Power.P_R(it)=(0.5*PL1)+(0.5*PL2);  %Target is 2 minute moving average
    end
    if it<=2
        Power.P_R(it)=P10;  %First 2 time-steps we set the target to whatever the load community would do without intervention
    end
    Power.P_T(it)=Power.P_R(it);

    % print new line every simulated hour
    hr = (it - 1) / 60 + 1;
    if it == 1
        fprintf('Simulation progress, splitted by hour:\n')
        fprintf('(< means target below PTmin, > means above, . is ok)\n')
        fprintf('%2d: ', hr)
    elseif mod(it, 60) == 1
        fprintf('\n%2d: ', hr)
    end
    %Verify target is feasible, and modify if not.
    PTmin=P1min+PU-PW;
    PTmax=P1max+PU-PW;
    if Power.P_T(it)<PTmin;
        fprintf('<')
        Power.P_T(it)=PTmin;
    elseif Power.P_T(it)>PTmax
        fprintf('>')
        Power.P_T(it)=PTmax;
    else
        fprintf('.') %simply to show that the program is progressing correctly.
    end


    House.P_target(it+1)=Power.P_T(it)-PU+PW;
end
