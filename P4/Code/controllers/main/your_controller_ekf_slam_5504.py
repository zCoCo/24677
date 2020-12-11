# Fill in the respective function to implement the LQR/EKF SLAM controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
from warnings import warn
from scipy.ndimage import gaussian_filter1d

import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from scipy.spatial.transform import Rotation
from util import *
from ekf_slam import EKF_SLAM

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define Vehicle constants:
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
            
        # B is constant:
        self.B = np.asarray([
            [0, 0],
            [2*self.Ca/self.m, 0],
            [0, 0],
            [2*self.Ca*self.lf/self.Iz, 0]
        ])
        
        ####
        # Vehicle Configuration Constants:
        ####
        self.F_max = 15737 # [N] Maximum vehicle thrust
        self.delta_max = np.pi/6 # [rad] Maximum steering angle (symmetric about 0)
        
        self.decel_max = 9.08 # no brakes: 1.1 # [m/s/s] Maximum capable longitudinal deceleration rate under any condition (empirical from coasting down)
        self.accel_max = 1.01*self.F_max / self.m # [m/s/s] Maximum vehicle acceleration
        
        ####
        # Track Configuration Constants:
        ####
        self.track_length = 1290.385282704354 # [m] Total path length of the track
        
        ####
        # Settings:
        ####
        self.target_time = 40 # [s] Target time to complete loop
        self.max_cornering_speed = 21.5  # [m/s] Maximum Cornering Speed (empirical)
        self.max_cornering_speed = min(self.max_cornering_speed, self.track_length/self.target_time)
        vcm = self.max_cornering_speed # short hand
        
        self.vmax = 90 # [m/s] Maximum allowable instantaneous speed (before system becomes unstable)
        
        # Velocity Waypoints (what speed should the car be going at key points along the track):
        self.vel_waypoints = [
            (0,0),
            (2100,vcm/1.49),#(1700,vcm/1.38),
            (2200,vcm/1.49/1.5),#(1700,vcm/1.38),
            #(2250,vcm),#(1700,vcm/1.38),
            #(2300,vcm/1.5),#(1700,vcm/1.38),
            #(2453,vcm/2.0/1.5),
            #(3236,2*vcm), # <-
            #(5217,vcm),
            (5665,vcm/0.975), # 5685
            #(6574,2*vcm),
            #(7650,vcm/1.1),
            #(7799,vcm/1.42),
            (7800,vcm/1.5),
            (7885,vcm/1.5),
            (8203,self.vmax) # Floor it at the end
        ]
        
        #self.desired_poles = np.asarray([-2.5, -5.3, -0.5+1j, -0.5-1j])
        
        # Base (unadjusted) LQR Matrices for Lateral Control:
        self.Q0 = np.diag(np.asarray([50,10,65,15]) / np.asarray([6,3.21,self.delta_max,np.sqrt(2)/2])**2) # State Cost Weights (settings / max-values^2), rate maxes come from observation
        self.R0 = 110 / (2*self.delta_max)**2 # Input Cost Weights
        
        ####
        # Setup:
        ####
        self.counter = 0
        self.time = 0 # [s] Current time into trajectory execution
        np.random.seed(99)
        
        # Precompute Curvature Vector:
        self.curve = self.computeCurvature()
        
        # Velocity Trajectory (at each position point along the track, what's the target vehicle speed). Used to set xdot depending on track conditions.
        self.vel_traj, time_traj, dist_traj, tf, traj_info = self.s_traj_waypoints(
            pos_traj = self.trajectory,
            a = self.accel_max, 
            d = self.decel_max,
            vel_waypoints = self.vel_waypoints
        )
        """
        self.vel_traj, time_traj, v_ratio = self.fastest_trap_vel_traj(
            pos_traj = self.trajectory,
            vavg = self.track_length / self.target_time,
            amax = self.accel_max,
            dmax = self.decel_max,
            vel_waypoints = self.vel_waypoints
        )
        print(r"Requested Avg. Velocity is {}% of max speed for given profile waypoints.".format(int(100*v_ratio)))
        """
        print(r"Target Time: {}s, Expected Completion Time {}s".format(self.target_time, tf))
        
        if True:
            # Plot Distance- and Time-Parameterized Velocity Trajectory:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(dist_traj,self.vel_traj)
            for t0,s0,v0, tp,sp,vp, tf,sf,vf in traj_info: 
                plt.axvline(s0, linestyle="--", color='blue')
                plt.axvline(sp, linestyle=":", color='black')
                plt.axvline(sf, linestyle="--", color='red')
                
                plt.axhline(v0, linestyle="--", color='blue')
                plt.axhline(vf, linestyle="--", color='red')
                plt.axhline(vp, linestyle=":", color='black')
            plt.xlabel('s')
            plt.ylabel('v(s)')
            plt.show()
            
            plt.figure()
            plt.plot(time_traj,self.vel_traj)
            for t0,s0,v0, tp,sp,vp, tf,sf,vf in traj_info: 
                plt.axvline(t0, linestyle="--", color='blue')
                plt.axvline(tp, linestyle=":", color='black')
                plt.axvline(tf, linestyle="--", color='red')
                
                plt.axhline(v0, linestyle="--", color='blue')
                plt.axhline(vf, linestyle="--", color='red')
                plt.axhline(vp, linestyle=":", color='black')
            plt.xlabel('t')
            plt.ylabel('v(t)')
            plt.show()
        
        self.e1_last = 0
        self.e2_last = 0
        
        ####
        # Control Parameters:
        ####
        # Target Correction Time [s] (used to calibrate PID constants for each control loop)
        self.kt = 1 # note: this is a qualitative trimming factor
        # Core PID Constants (kp, ki, kd), adjusted (trimmed) for each controller via kt.
        #L_long = (12768 + 0.6*(13088-12768))/1000 # time for long controller to reach 10% SS with step input
        #T_long = (85824 + 0.4*(86720-85824))/1000 # time for long controller to reach 90% SS with step input
        Ku_long = 163500 # Lowest long gain causing rapid oscillation in speed under step input
        Tu_long = (0.096/2 + 0.032/2) # Period of that oscillation
        Ku_lat = 0.60#*0.61
        Tu_lat = 6
        self.k_pid_longitudinal = np.asarray([0.60*Ku_long, 1.2*Ku_long/Tu_long, 3*Ku_long*Tu_long/40]).reshape((1,3)) #np.asarray([163500,0,0]).reshape((1,3))#np.asarray([1.2*T_long/L_long, 1.2*T_long/L_long/(2.0*L_long), 1.2*T_long/L_long*0.5*L_long]).reshape((1,3))#np.asarray([0.60*Ku, 1.2*Ku/Tu, 3*Ku*Tu/40]).reshape((1,3))
        self.k_pid_lateral = np.asarray([0.60*Ku_lat, 1.2*Ku_lat/Tu_lat, 3*Ku_lat*Tu_lat/40]).reshape((1,3))#np.asarray([500,0,0]).reshape((1,3))#np.asarray([1,0,0]).reshape((1,3))#
        
        self.look_ahead_multiple = 17.7 # How many car lengths to look ahead and average over when determining desired heading angle
        self.look_ahead_indices = int(self.look_ahead_multiple * 24.0) # Corresponding number of indices (note: car length is approx. equiv. to path length over 24 points)
        
        ### Time of Last Zero Crossing for Each Signal (for determining Ziegler-Nichols Tu):
        self.last_zero_crossing = np.zeros((3,1))
        self.oscillation_period = np.zeros((3,1))
        
        ####
        # Initialize Signals:
        ####
        
        # Initialize Error Signals as an Errors Matrix, E:
        # Rows are: (x,y,psi) = (alongtrack, crosstrack, heading),
        # Cols are: Position, Integral, Derivative
        self.E = np.zeros((3,3))
        
        self.decel_rate = 0
        self.xdot_last = 0

        # For determining max decel rate:
        self.applied_brakes_time = 0
        self.applied_brakes_speed = float('inf')

        # Print Key Settings:
        print((self.max_cornering_speed, self.look_ahead_multiple, self.Q0, self.R0, self.vel_waypoints))

    def getStates(self, timestep, use_slam=False):

        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

        # Initialize the EKF SLAM estimation
        if self.counter == 0:
            # Load the map
            minX, maxX, minY, maxY = -120., 450., -350., 50.
            #minX, maxX, minY, maxY = -120., 450., -500., 50.
            map_x = np.linspace(minX, maxX, 7)
            map_y = np.linspace(minY, maxY, 7)
            map_X, map_Y = np.meshgrid(map_x, map_y)
            map_X = map_X.reshape(-1,1)
            map_Y = map_Y.reshape(-1,1)
            self.map = np.hstack((map_X, map_Y)).reshape((-1))
            
            # Parameters for EKF SLAM
            self.n = int(len(self.map)/2)             
            X_est = X + 0.5
            Y_est = Y - 0.5
            psi_est = psi - 0.02
            mu_est = np.zeros(3+2*self.n)
            mu_est[0:3] = np.array([X_est, Y_est, psi_est])
            mu_est[3:] = np.array(self.map)
            init_P = 1*np.eye(3+2*self.n)
            W = np.zeros((3+2*self.n, 3+2*self.n))
            W[0:3, 0:3] = delT**2 * 0.1 * np.eye(3)
            V = 0.1*np.eye(2*self.n)
            V[self.n:, self.n:] = 0.01*np.eye(self.n)
            # V[self.n:] = 0.01
            print(V)
            
            # Create a SLAM
            self.slam = EKF_SLAM(mu_est, init_P, delT, W, V, self.n)
            self.counter += 1
        else:
            mu = np.zeros(3+2*self.n)
            mu[0:3] = np.array([X, 
                                Y, 
                                psi])
            mu[3:] = self.map
            y = self._compute_measurements(X, Y, psi)
            mu_est, _ = self.slam.predict_and_correct(y, self.previous_u)

        self.previous_u = np.array([xdot, ydot, psidot])

        print("True      X, Y, psi:", X, Y, psi)
        print("Estimated X, Y, psi:", mu_est[0], mu_est[1], mu_est[2])
        print("-------------------------------------------------------")
        
        if use_slam == True:
            return delT, mu_est[0], mu_est[1], xdot, ydot, mu_est[2], psidot
        else:
            return delT, X, Y, xdot, ydot, psi, psidot

    def s_traj(self, t0,s0,v0,sf,vf,a,d):
        """
        Computes the parameters for an s-parameterized pyramid trajectory which 
        starts at (t0,s0,v0) and ends at (sf,vf) while accelerating at a and 
        decelerating at d. If a speed is unreachable, this trajectory hits the 
        closest possible speed.
        """
        sf = sf - s0 # zero
        tp = ( np.sqrt( (d*v0**2 + a*vf**2 + 2*a*d*sf) / (a+d) ) - v0) / a
        vp = v0 + a*tp
        sp = 0.5 * (v0+vp)*tp
        tf = 2*(sf-sp)/(vp+vf) + tp
        
        sp = sp + s0 # reoffset
        tp = tp + t0 # offset
        tf = tf + t0 # offset
        
        return tp, tf, vp, sp
    
    def s_traj_follow(self, s, a,d, t0,s0,v0, tp,sp,vp):
        """
        For an s-parameterized pyramid trajectory with the given parameters 
        produced by s_traj, this returns the target time and velocity at position 
        s.
        """
        if s <= sp:
            Dt = (np.sqrt(v0**2 + 2*a*(s-s0)) - v0) / a
            t = t0 + Dt
            v = v0 + a*Dt
        else:
            Dt = (vp - np.sqrt(vp**2 - 2*d*(s-sp))) / d
            t = tp + Dt
            v = vp - d*Dt
            
        return v,t
    
    def s_traj_waypoints(self, pos_traj, a, d, vel_waypoints):
        if vel_waypoints[0][0] != 0:
            vel_waypoints = [(0,0)] + vel_waypoints
        
        if vel_waypoints[-1][0] != pos_traj.shape[0]:
            vel_waypoints = vel_waypoints + [(pos_traj.shape[0],0)]
        
        traj_info = []
        t0 = 0
        s0 = 0
        for w in range(1,len(vel_waypoints)):
            i_prev, v_prev = vel_waypoints[w-1]
            i, vi = vel_waypoints[w]
            
            # Path length of section:
            Ds = self.path_length(pos_traj, i_prev, i)
            sf = s0 + Ds
            
            # Compute Trajectory Parameters:
            tp, tf, vp, sp = self.s_traj(t0,s0,v_prev, sf,vi, a,d)
            
            # Append info defining the sub trapezoidal trajectory:
            traj_info.append((t0,s0,v_prev, tp,sp,vp, tf,sf,vi))
            
            # Update s0,t0
            s0 = sf
            t0 = tf
            
        
        #print(vel_waypoints)
        #print(traj_info)
        
        
        dist_traj = np.zeros((pos_traj.shape[0],1))
        time_traj = np.zeros((pos_traj.shape[0],1))
        vel_traj = np.zeros((pos_traj.shape[0],1))
        vel_traj[0] = vel_waypoints[0][1]
        
        t = 0 # trajectory info index
        s = 0 # total distance into trajectory of current waypoint
        (t0,s0,v_prev, tp,sp,vp, tf,sf,vi) = traj_info[t] # parameters of current trapezoid
        for i in range(1,pos_traj.shape[0]):
            s = s + np.sqrt(np.sum((pos_traj[i] - pos_traj[i-1])**2))
            
            # If s is outside of pyramid, advance pyramid index t until it's inside
            while s > sf:
                t = t+1
                (t0,s0,v_prev, tp,sp,vp, tf,sf,vi) = traj_info[t]
            
            assert(s0 <= s <= sf)
            #assert(sm <= se)
            
            vel_traj[i], time_traj[i] = self.s_traj_follow(s, a,d, t0,s0,v_prev, tp,sp,vp)
            dist_traj[i] = s
            
        #print(vel_traj)
        return vel_traj, time_traj, dist_traj, tf, traj_info

    def fastest_trap_vel_traj(self, pos_traj, vavg, amax, dmax, vel_waypoints):
        """
        Compute the fastest *max deceleration* velocity trajectory.
        
        NOTE: @TODO: This function is still very buggy (its computed 
                     trajectories don't meet the required constraints but it 
                     generally does something close to correct so for this 
                     assignment it is sufficient).
        
        The fastest velocity trajectory is an array consisting of a target 
        (longitudinal) velocity for each point in the given position trajectory 
        while ensuring that the vehicle hits each target velocity at each of 
        the specified indices for each of the (waypoint_idx, target_velocity) 
        tuples in the vel_waypoints array.
        
        That is, between each waypoint specified in vel_waypoints, a 
        path-length parameterized trapezoidal velocity profile is computed 
        which starts at the previous target velocity at the previous index 
        and ends at the specified target velocity at the specified index while 
        ensuring that the vehicle *decelerates at dmax*, doesn't exceed 
        amax (or throws a warning if it does), and ensures that the average 
        velocity of the entire trajectory is vavg.
        This is best used when you want to wait to brake until the last 
        possible second (for drifting) or have very limited braking (i.e. you 
        can only coast down from a speed and don't have real brakes).
        
        Returns the fastest *max deceleration* velocity trajectory as a numpy 
        array as well as the ratio vavg to its maximum allowable value (the 
        value above which the vehicle will have to exceed amax).
        
        Note:
        - If the index of the first waypoint isn't 0, (0,0) will get added 
        to the beginning.
        - If the index of the last waypoint isn't pos_traj.shape[0], 
        (pos_traj.shape[0], 0) will be added.
        - If amax is ever exceeded, a warning will be thrown but the trajectory
        will still be computed as normal (requiring the vehicle to exceed amax). 
        This is useful if you want the vehicle to just floor it after a certain 
        point by setting an unreachably high target velocity for the final waypoint.

        Parameters
        ----------
        pos_traj
            Nx2 numpy array containing the [x,y] coordinates of each waypoint.
        vavg
            Target average velocity for the entire trajectory (used for tuning vehicle performance).
        amax
            Maximum allowable vehicle acceleration.
        dmax
            Maximum achievable vehicle deceleration (used for all decelerations).
        vel_waypoints
            List of tuples containing the index of a waypoint and the target vehicle velocity for that waypoint.

        Returns
        -------
        vel_traj
            Computed velocity trajectory.
        time_traj
            The target time for arriving at each point in the trajectory.
        vavg_ratio_max
            Largest ratio of vavg to vavg_max. That is: how close is vavg to requiring one of the trapezoids to exceed amax

        """
        if vel_waypoints[0][0] != 0:
            vel_waypoints = [(0,0)] + vel_waypoints
        
        if vel_waypoints[-1][0] != pos_traj.shape[0]:
            vel_waypoints = vel_waypoints + [(pos_traj.shape[0],0)]
        
        traj_info = []
        vavg_ratio_max = 0 # How close is vavg to requiring one of the trapezoids to exceed amax
        for w in range(1,len(vel_waypoints)):
            i_prev, v_prev = vel_waypoints[w-1]
            i, vi = vel_waypoints[w]
            
            # Path length of section:
            s = self.path_length(pos_traj, i_prev, i)
            sd = s * dmax;
            
            # Peak velocity:
            vp = (vi**2 - v_prev*vi + 2*sd - (sd*v_prev)/vavg)/(vi - v_prev + sd/vavg)
            # Required acceleration:
            a = (vavg*(vavg*v_prev**2 - 2*vavg*v_prev*vi - 2*sd*v_prev + vavg*vi**2 + 2*sd*vavg))/(s*(- 2*vavg**2 + 2*vi*vavg + sd))
            
            if abs(a) > amax:
                warn("Max acceleration of {}m/s/s being exceeded at {}m/s/s between {} and {} in fastest_trap_vel_traj".format(amax, a, vel_waypoints[w-1], vel_waypoints[w]))
            
            # Maximum allowable average velocity between these two waypoints:
            vavg_max = (sd*v_prev + np.sqrt((sd + amax*s)*(sd*v_prev**2 + amax*s*vi**2 + 2*amax*s*sd)) + amax*s*vi)/(v_prev**2 - 2*v_prev*vi + vi**2 + 2*sd + 2*amax*s)
            
            # Ratio of requested vavg to vavg_max:
            vavg_ratio = vavg / vavg_max
            # Store the highest one:
            vavg_ratio_max = max(vavg_ratio, vavg_ratio_max)
            
            # Midpoint Distance:
            sm = (vp**2 - v_prev**2) / 2 / a
            # Endpoint Distance:
            se = sm + (vp**2 - vi**2) / 2 / dmax
            
            # Append info defining the sub trapezoidal trajectory:
            traj_info.append((vp,a,sm,se))
        
        #print(vel_waypoints)
        #print(traj_info)
        
        
        time_traj = np.zeros((pos_traj.shape[0],1))
        vel_traj = np.zeros((pos_traj.shape[0],1))
        vel_traj[0] = vel_waypoints[0][1]
        
        t = 0 # trajectory info index
        s = 0 # total distance into trajectory of current waypoint
        t0 = 0
        s0 = 0 # starting distance of current trapezoid
        vp,a,sm,se = traj_info[t] # parameters of current trapezoid
        v_prev = vel_waypoints[t][1]
        vi = vel_waypoints[t+1][1]
        for i in range(1,pos_traj.shape[0]):
            s = s + np.sqrt(np.sum((pos_traj[i] - pos_traj[i-1])**2))
            
            # Ensure point is still in the trapezoid:
            while s > (se + s0):
                #print((t,s0,s,(se+s0)))
                t = t+1
                s0 = se + s0
                t0 = (np.sqrt(v_prev**2 + 2*a*sm) - v_prev)/a + (vp - np.sqrt(vp**2 - 2*dmax*(se-sm)))/dmax + t0
                vp,a,sm,se = traj_info[t]
                v_prev = vel_waypoints[t][1]
                vi = vel_waypoints[t+1][1]
            
            assert(s0 <= s <= (se+s0))
            #assert(sm <= se)
            
            if s < (sm + s0):
                # If in acceleration portion:
                time_traj[i] = (np.sqrt(v_prev**2 + 2*a*(s-s0)) - v_prev)/a + t0
                vel_traj[i] = np.sqrt(v_prev**2 + 2*a*(s-s0)) #v_prev + np.sqrt(2*a*(s-s0))
            else:
                # If in decelertion portion:
                time_traj[i] = (np.sqrt(v_prev**2 + 2*a*sm) - v_prev)/a + (vp - np.sqrt(vp**2 - 2*dmax*(s-sm-s0)))/dmax + t0
                vel_traj[i] = np.sqrt(vp**2 - 2*dmax*(s-s0-sm)) #vp - np.sqrt(2*dmax*(s-sm-s0))
                
        #print(vel_traj)
        return vel_traj, time_traj, vavg_ratio_max
                
 
    def path_length(self, pos_traj, start, end):
        """
        Compute the path distance between two points on a trajectory.
        
        Computes the path_length distance along the given position trajectory 
        between the points at the given indices. If end < start, the distance 
        will be negative.

        Parameters
        ----------
        pos_traj
            Nx2 numpy array containing the [x,y] coordinates of each waypoint.
        start
            Index of the first point.
        end
            Index of the second point.

        Returns
        -------
        path_distance
            Path length distance between points at given indices.

        """
        idx_1, idx_2 = tuple(map(int, (start, end)))
        path_sign = 1.0
        if idx_1 > idx_2:
            # Ensure proper ordering. Swap order and sign if necessary.
            path_sign = -1.0
            idx_1, idx_2 = idx_2, idx_1
        section = pos_traj[idx_1:(idx_2+1)] # Section of trajectory between idx_1 and idx_2, inclusive
        point_distances = np.sqrt(np.sum(np.diff(section, axis=0)**2, axis=1)) # Distances between each point
        path_distance = path_sign * np.sum(point_distances)
        return path_distance

    def computeCurvature(self):
        # Function to compute and return the curvature of a trajectory.
        sigmaGauss = 5 # We can change this value to increase filter strength
        trajectory = self.trajectory
        xp = gaussian_filter1d(input=trajectory[:,0],sigma=sigmaGauss,order=1)
        xpp = gaussian_filter1d(input=trajectory[:,0],sigma=sigmaGauss,order=2)
        yp = gaussian_filter1d(input=trajectory[:,1],sigma=sigmaGauss,order=1)
        ypp = gaussian_filter1d(input=trajectory[:,1],sigma=sigmaGauss,order=2)
        curve = np.zeros(len(trajectory))
        for i in range(len(xp)):
            curve[i] = (xp[i]*ypp[i] - yp[i]*xpp[i])/(xp[i]**2 + yp[i]**2)**1.5
                    
        return curve

    def dlqr(self, A,B,Q,R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        S = np.matrix(linalg.solve_discrete_are(A, B, Q, R))
        #compute the LQR gain
        K = np.matrix(linalg.inv(B.T@S@B+R)@(B.T@S@A))
        eigVals, eigVecs = linalg.eig(A-B@K)
        return K, S, eigVals

    def _compute_measurements(self, X, Y, psi):
        x = np.zeros(3+2*self.n)
        x[0:3] = np.array([X, Y, psi])
        x[3:] = self.map
        
        p = x[0:2]
        psi = x[2]
        m = x[3:].reshape((-1,2))

        y = np.zeros(2*self.n)

        for i in range(self.n):
            y[i] = np.linalg.norm(m[i, :] - p)
            y[self.n+i] = wrapToPi(np.arctan2(m[i,1]-p[1], m[i,0]-p[0]) - psi)
            
        y = y + np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V)
        # print(np.random.multivariate_normal(np.zeros(2*self.n), self.slam.V))
        return y

    def update(self, timestep, driver):
        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the newly defined getStates method
        delT, X, Y, xdot, ydot, psi, psidot = self.getStates(timestep, use_slam=False)

        self.time += delT # update total time into trajectory execution
    
        # Design your controllers in the spaces below. 
        # Remember, your controllers will need to use the states
        # to calculate control inputs (F, delta).

        # Compute rotation matrix from world frame to local (vehicle) frame:
        rotation_mat = np.array([[np.cos(psi), np.sin(psi)], [-np.sin(psi), np.cos(psi)]])

        ####
        # Compute Pose to Path Error:
        # This is what keeps the vehicle on track.
        ####
        ### Compute Vector from Vehicle COM to Nearest Waypoint in Local (Vehicle) Frame:
        # Get nearest waypoint to vehicle COM:
        _, nearest_waypoint_idx = closestNode(X, Y, trajectory)
        nearest_waypoint = trajectory[nearest_waypoint_idx]
        # Compute pose to path (waypoint) vector in world frame:
        p2p_world = nearest_waypoint - np.asarray([X,Y])
        # Compute pose to path (waypoint) vector in local (vehicle) frame:   
        p2p_local = rotation_mat @ p2p_world
        
        ### Primitive "Path-Aware" Lookahead:
        prev_idx = 0
        if nearest_waypoint_idx > 1700:
            prev_idx, prev_lam = 1700, 17.7
            targ_idx, targ_lam = 2100, 17.7
        if nearest_waypoint_idx > 2100:
            prev_idx, prev_lam = 2100, 17.7
            targ_idx, targ_lam = 2200, 17.7
        if nearest_waypoint_idx > 2200:
            prev_idx, prev_lam = 2200, 17.7
            targ_idx, targ_lam = 3000, 17.7
        if nearest_waypoint_idx > 3000:
            prev_idx, prev_lam = 3000, 17.7
            targ_idx, targ_lam = 5000, (48-17.7)/3000 * 2000 + 17.7
        if nearest_waypoint_idx > 5000:
            prev_idx, prev_lam = 5000, (48-17.7)/3000 * 2000 + 17.7
            targ_idx, targ_lam = 6000, 70
        if nearest_waypoint_idx > 6000:
            prev_idx, prev_lam = 6000, 42
            targ_idx, targ_lam = 7800, 12
        if nearest_waypoint_idx > 7800:
            prev_idx, prev_lam = 7800, 8
            targ_idx, targ_lam = 8203, 0
        if prev_idx > 0:
            self.look_ahead_multiple = (targ_lam-prev_lam) / (targ_idx-prev_idx) * (nearest_waypoint_idx-prev_idx) + prev_lam
            self.look_ahead_indices = int(self.look_ahead_multiple * 24.0) # Corresponding number of indices (note: car length is approx. equiv. to path length over 24 points)
        
        ### Compute Pose to Path Heading Angle Error:
        # Points that comes before and 2.5 car lengths later waypoint:
        nearest_waypoint_prev = trajectory[(nearest_waypoint_idx-1) % trajectory.shape[0]]
        nearest_waypoint_fwd = trajectory[(nearest_waypoint_idx+self.look_ahead_indices) % trajectory.shape[0]]
        # Desired Trajectory Position Delta around Current Waypoint:
        nearest_waypoint_delta = nearest_waypoint_fwd - nearest_waypoint_prev
        # Compute Desired Heading:
        desired_heading = np.arctan2(nearest_waypoint_delta[1], nearest_waypoint_delta[0])
        # Compute current heading of the vehicle (direction it's currently driving, not necessarily pointing direction if there's sideslip):
        Xdot, Ydot = np.linalg.inv(rotation_mat) @ np.asarray([xdot,ydot])
        current_heading = np.arctan2(Ydot,Xdot)
        # Compute Heading Error:
        p2p_heading_error = desired_heading - current_heading
        p2p_heading_error = np.arctan2(np.sin(p2p_heading_error), np.cos(p2p_heading_error)) # Remap to atan2 space
        
        ####
        # Compute Alongtrack Error as longitudinal speed difference:
        ####
        speed_diff = clamp(self.vel_traj[nearest_waypoint_idx],0, self.vmax) - xdot
        
        if False:
            # Print Deceleration Rate:
            decel = (xdot - self.xdot_last)/delT
            if decel < 0:
                print(decel)
            self.xdot_last = xdot
        
        """
        # Second oldest alongtrack error:
        ###
        # Compute path length from the nearest waypoint to the desired waypoint based on desired completion time:
        # This is what propels the vehicle forward.
        ###
        ### Get The Waypoint Vehicle Should be at given the time into execution:
        target_waypoint_idx = clamp(np.round(self.time / self.target_time * trajectory.shape[0]), 0,trajectory.shape[0]-1)
        ### Compute the path distance between the waypoints:
        idx_1, idx_2 = tuple(map(int, (nearest_waypoint_idx, target_waypoint_idx)))
        path_sign = 1.0
        if idx_1 > idx_2:
            # Ensure proper ordering. Swap order and sign if necessary.
            path_sign = -1.0
            idx_1, idx_2 = idx_2, idx_1
        section = trajectory[idx_1:(idx_2+1)] # Section of trajectory between idx_1 and idx_2, inclusive
        point_distances = np.sqrt(np.sum(np.diff(section, axis=0)**2, axis=1)) # Distances between each point
        path_distance = path_sign * np.sum(point_distances)
        """
        """
        ### Old alongtrack error (alongtrack distance from current position to target position):
        target_waypoint = trajectory[int(target_waypoint_idx)]
        # Compute pose to target waypoint vector in world frame:
        p2target_world = target_waypoint - np.asarray([X,Y])
        # Compute pose to target waypoint vector in local (vehicle) frame:   
        p2target_local = rotation_mat @ p2target_world
        path_distance = p2target_local[0]
        """
        
        ####
        # Update Error Signals in an Errors Matrix, E:
        ####
        # Rows are: (x,y,psi) = (alongtrack, crosstrack, heading),
        # Cols are: Position, Integral, Derivative
        E = self.E
        E_prev = E.copy()
        
        ### Update Proportional Signals:
        # Alongtrack error (speed difference from required speed)
        E[0,0] = speed_diff
        # Crosstrack error (from current position to nearest waypoint in trajectory):
        E[1,0] = p2p_local[1]
        # Heading error (from current heading to heading implictly specified by trajectory at nearest waypoint):
        E[2,0] = p2p_heading_error
            
        ### Update Integral Signals:
        E[:,1] = E[:,1] + E[:,0] * delT # mult by timestep in case not uniform (not if Webots Fast changes dt)
            
        ### Update Derivative Signals:
        E[:,2] = (E[:,0] - E_prev[:,0]) / delT # divide by timestep in case not uniform (not if Webots Fast changes dt)
        
        ####
        # Condition Error Signals:
        ####
        ### Prevent Integral Windup:
        # If error has crossed zero, zero it:
        zero_crossing = (np.sign(E[:,0]) * np.sign(E_prev[:,0])) <= 0.0
        E[zero_crossing,1] = 0.0
        # If derivative is high (still driving towards equilibrium), reduce integral contribution:
        high_deriv_error= E[:,2]*delT > np.asarray([
            0.5,
            (lf+lr)/4, # High Y err deriv means > 1/4 car length
            2*np.pi/6/4 # High TH err deriv means > 1/4 steering range
        ])
        E[high_deriv_error,1] -= E[high_deriv_error,0] * delT * 9.0/10.0 # only contribute 1/10th of what you would have to the integral

        ### Find and Print Oscillation Period (for determining Ziegler-Nichols Tu):
        if np.count_nonzero(zero_crossing):
            self.oscillation_period[zero_crossing] = self.time - self.last_zero_crossing[zero_crossing]
            self.last_zero_crossing[zero_crossing] = self.time
            #print(self.oscillation_period.T)

        ####
        # Compute PID Correction Factors (allow the same tuned constants to 
        # drive each controller = fewer constants to tune overall).
        ####
        kt = self.kt
        kx = 1/kt;
        kth = 1/kt;
        V = np.abs(np.linalg.norm([xdot, ydot]));
        if V < 0.05: # Prevent from getting too large
            ky = 2 / 2/0.05 / (kt**2); 
        else:
            ky = 2 / 2/V / (kt**2); # Lessen impact of crosstrack error as velocity increases (to minimize wild swings)
        
        ####
        # Create a PID Constants Matrix, K:
        # Each row contains the (kp, ki, and kd) for the error terms 
        # corresponding to that row (ex, ey, or eth).
        ####
        K_pid = np.asarray([self.k_pid_longitudinal*kx, self.k_pid_lateral*ky, self.k_pid_lateral*kth]).squeeze()
        
        ####
        # Compute Control Signals to Minimize Each Type of Error 
        # (rows: x=alongtrack, y=crosstrack, th=heading)
        ####
        C = np.sum(E*K_pid, axis=1)
        # Extract Independent Control Signals for Minimizing Alongtrack, Crosstrack, and Heading Errors:
        (Cx, Cy, Cth) = C;
        
        #print((np.round(E[0,0]), self.track_length / self.target_time, xdot, np.round(Cx), np.round(Cy), np.round(Cth)))
        #print((np.round(1000*self.time), np.round(speed_diff)))
        
        # ---------------|Lateral Controller|-------------------------
        # Note: Lateral controller consists of two PID subcontrollers working togther to minimize heading error and crosstrack error.
        # Using a heading control component allows for improved handling of turns (since its trying to point the car in the right direction)
        
        # Compute State Space:
        Vx = xdot
        A = np.asarray([
            [0, 1, 0, 0],
            [0, -4*Ca/m/Vx, 4*Ca/m, -2*Ca*(lf-lr)/m/Vx],
            [0, 0, 0, 1],
            [0, -2*Ca*(lf-lr)/Iz/Vx, 2*Ca*(lf-lr)/Iz, -2*Ca*(lf**2+lr**2)/Iz/Vx]
        ])
        B = self.B[:,0].reshape(-1,1) # only consider delta as an input
        
        C = np.eye(A.shape[0])
        D = np.zeros(B.shape)
        
        # Discretize System:
        ss_CT = signal.StateSpace(A,B,C,D)
        ss_DT = ss_CT.to_discrete(delT)
        
        # Perform IH-DT-LQR Control:
        Q = self.Q0 # State Cost Weights
        R = (xdot / self.max_cornering_speed + 1) * self.R0 # Input Cost Weights
        K, _, _ = self.dlqr(ss_DT.A,ss_DT.B, Q,R)
        
        # Calculate Appropriate States:
        e1 = -p2p_local[1]
        
        e1dot = (e1 - self.e1_last) / delT
        self.e1_last = e1
        
        e2 = np.arctan2(np.sin(psi), np.cos(psi)) - desired_heading # TODO: Clean this up
        #e2 = -p2p_heading_error
        e2 = wrapToPi(psi)-wrapToPi(desired_heading)
        e2 = np.arctan2(np.sin(e2), np.cos(e2))
        #print((psi, desired_heading, p2p_heading_error, psi-desired_heading, wrapToPi(psi)-wrapToPi(desired_heading)))
        
        #psidotDesired = xdot * self.curve[(nearest_waypoint_idx+self.look_ahead_indices) % trajectory.shape[0]]
        #e2dot = psidot - psidotDesired
        
        e2dot = (e2 - self.e2_last) / delT
        self.e2_last = e2
        
        X_lat = np.asarray([e1,e1dot,e2,e2dot]).reshape(-1,1) # Lateral States
            
        # Calculate Feedforward Steering Angle:
        kappa = self.curve[int(nearest_waypoint_idx+0*self.look_ahead_indices/10) % trajectory.shape[0]] # Signed path curvature at half lookahead distance
        L = lr + lf # wheel base (axle-to-axle separation)
        mr = m * lf/L # mass carried by rear wheel(s)
        mf = m * lr/L # mass carried by front wheel(s)
        Kv = mf/2/Ca - mr/2/Ca # understeer gradient
        ay = Vx**2 * kappa # lateral acceleration due to curvature
        e2_ss = (lf*m*ay/2/Ca/L - lr * kappa) # steady state heading error under minimized lateral position error
        k3 = K[0,2] # Gain term affecting heading error
        
        print(np.sign(kappa))
        deltaFF = L*kappa + Kv * ay + k3 * e2_ss
        
        # Implement Feedforward + Fullstate Feedback Control:
        delta = -K@X_lat
        # delta = deltaFF - K@X_lat # use feedforward
        delta = delta[0,0] # convert from 1x1 array to scalar
        delta = clamp(delta, -self.delta_max, self.delta_max)

        # ---------------|Longitudinal Controller|-------------------------
        F = np.sign(Cx) * np.abs(np.linalg.norm([clamp(Cx,0,self.F_max), Cy])) # Allow crosstrack and alongtrack errors to drive the throttle
        F = F - np.abs(delta) * 0.05 * self.F_max / self.delta_max # don't accelerate as much if you're actively turning the wheel
        F_w_steering_correction = F - np.abs(delta) * 0.5 * self.decel_max*m / self.delta_max # brake if you're actively turning the wheel a lot
        print((F,F_w_steering_correction,-F_w_steering_correction/m/self.decel_max))
        if F_w_steering_correction < 0:
            driver.setBrakeIntensity(clamp(-12.0*F_w_steering_correction/m/self.decel_max, 0, 1))
            F = 0
        else:
            driver.setBrakeIntensity(0) # stop braking
        F = clamp(F,0,self.F_max) # Ensure appropriate bounds (main will otherwise treat negative F as positive)
        # Note: This approach is effectively equivalent to having one longitudinal pid controller which controls norm[ey/kt, eth/V/kt**2] but is just a cleaner representation
        # Todo: Slow down based on Cth? (slower when steering high to increase control authority)

        #print((int(100*delta*6/np.pi), int(100*F/15737)) )

        #print((int(100*delta/self.delta_max),int(100*F/self.F_max)))

        # Return all states and calculated control inputs (F, delta)
        # Setting brake intensity is enabled by passing
        # the driver object, which is used to provide inputs
        # to the car, to our update function
        # Using this function is purely optional.
        # An input of 0 is no brakes applied, while
        # an input of 1 is max brakes applied
        
        # Determine max decel rate:
        run_braking_test = False
        if run_braking_test and nearest_waypoint_idx > 400 and F < 0.1 and self.applied_brakes_time == 0:
            # If running test and well into straight away and have already peeked speed and haven't applied brakes yet
            self.applied_brakes_time = self.time
            self.applied_brakes_speed = xdot
            driver.setBrakeIntensity(1)
        
        if run_braking_test and self.applied_brakes_time > 0 and xdot < self.max_cornering_speed:
            F = 0
            driver.setBrakeIntensity(1)
            if xdot < 1:
                Dt = self.time - self.applied_brakes_time
                Dv = xdot - self.applied_brakes_speed
                decel = -Dv/Dt
                print("Max Decel Rate: {}".format(decel))
                raise Exception("Max Decel Rate determined. Killing sim.")

        #driver.setBrakeIntensity(clamp(-F/m/self.decel_max, 0, 1))

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta