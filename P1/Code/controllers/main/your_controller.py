# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *
from warnings import warn

# CustomController class (inherits from BaseController)
class CustomController(BaseController):

    def __init__(self, trajectory):

        super().__init__(trajectory)

        # Define constants
        # These can be ignored in P1
        self.lr = 1.39
        self.lf = 1.55
        self.Ca = 20000
        self.Iz = 25854
        self.m = 1888.6
        self.g = 9.81
        
        ####
        # Vehicle Configuration Constants:
        ####
        self.F_max = 15737 # [N] Maximum vehicle thrust
        self.delta_max = np.pi/6 # [rad] Maximum steering angle (symmetric about 0)
        
        self.decel_max = 1.1 # [m/s/s] Maximum capable longitudinal deceleration rate under any condition (empirical from coasting down)
        self.accel_max = self.F_max / self.m # [m/s/s] Maximum vehicle acceleration
        
        ####
        # Track Configuration Constants:
        ####
        self.track_length = 1290.385282704354 # [m] Total path length of the track
        
        ####
        # Settings:
        ####
        self.target_time = 40 # [s] Target time to complete loop
        self.max_cornering_speed = 8.2  # [m/s] Maximum Cornering Speed (empirical)
        self.max_cornering_speed = min(self.max_cornering_speed, self.track_length/self.target_time)
        vcm = self.max_cornering_speed # short hand
        
        # Velocity Waypoints (what speed should the car be going at key points along the track):
        self.vel_waypoints = [
            (0,0),
            (2256,vcm),
            (2453,vcm),
            (3236,2*vcm),
            (5217,2*vcm),
            (5835,vcm),
            #(6574,2*vcm),
            (7799,vcm),
            (8203,vcm/2.0)
        ]
        
        ####
        # Setup:
        ####
        self.time = 0 # [s] Current time into trajectory execution
        
        # Velocity Trajectory (at each position point along the track, what's the target vehicle speed). Used to set xdot depending on track conditions.
        self.vel_traj, time_traj, v_ratio = self.fastest_trap_vel_traj(
            pos_traj = self.trajectory,
            vavg = self.track_length / self.target_time,
            amax = self.accel_max,
            dmax = self.decel_max,
            vel_waypoints = self.vel_waypoints
        )
        print(r"Requested Avg. Velocity is {}% of max speed for given profile waypoints.".format(int(100*v_ratio)))
        
        if False:
            # Plot Velocity Trajectory:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(np.arange(0,self.vel_traj.shape[0]), self.vel_traj)
            plt.title('Velocity Trajectory Points')
            plt.show()
            
            plt.figure()
            plt.plot(np.arange(0,time_traj.shape[0]), time_traj)
            plt.title('Time Trajectory Points')
            plt.show()
            
            plt.figure()
            plt.plot(time_traj, self.vel_traj)
            plt.title('Velocity Trajectory')
            plt.show()
        
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
        
        self.look_ahead_multiple = 7.5 # How many car lengths to look ahead and average over when determining desired heading angle
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
        
        print(vel_waypoints)
        print(traj_info)
        
        
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
                print((t,s0,s,(se+s0)))
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
                
        print(vel_traj)
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

    def update(self, timestep):
        
        trajectory = self.trajectory

        lr = self.lr
        lf = self.lf
        Ca = self.Ca
        Iz = self.Iz
        m = self.m
        g = self.g

        # Fetch the states from the BaseController method
        delT, X, Y, xdot, ydot, psi, psidot = super().getStates(timestep)

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
        speed_diff = self.vel_traj[nearest_waypoint_idx] - xdot
        
        print((self.vel_traj[nearest_waypoint_idx], xdot))
        
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
        # Crosstrack error (speed difference from required speed)
        E[0,0] = speed_diff
        # Alongtrack error (from current position to nearest waypoint in trajectory):
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
        K = np.asarray([self.k_pid_longitudinal*kx, self.k_pid_lateral*ky, self.k_pid_lateral*kth]).squeeze()
        
        ####
        # Compute Control Signals to Minimize Each Type of Error 
        # (rows: x=alongtrack, y=crosstrack, th=heading)
        ####
        C = np.sum(E*K, axis=1)
        # Extract Independent Control Signals for Minimizing Alongtrack, Crosstrack, and Heading Errors:
        (Cx, Cy, Cth) = C;
        
        #print((np.round(E[0,0]), self.track_length / self.target_time, xdot, np.round(Cx), np.round(Cy), np.round(Cth)))
        #print((np.round(1000*self.time), np.round(speed_diff)))
        
        # ---------------|Lateral Controller|-------------------------
        # Note: Lateral controller consists of two PID subcontrollers working togther to minimize heading error and crosstrack error.
        # Using a heading control component allows for improved handling of turns (since its trying to point the car in the right direction)
        delta = clamp(Cy + Cth, -self.delta_max, self.delta_max); # Allow crosstrack and heading errors to drive the steering angle
        # Note: This approach is effectively equivalent to having one lateral pid controller which controls (ey/kt + eth/V/kt**2) but is just a cleaner representation

        # ---------------|Longitudinal Controller|-------------------------
        F = np.abs(np.linalg.norm([clamp(Cx,0,self.F_max), Cy])) # Allow crosstrack and alongtrack errors to drive the throttle
        F = clamp(F,0,self.F_max) - np.abs(delta) * 0.1 * self.F_max / self.delta_max # don't accelerate as much if you're actively turning the wheel
        F = clamp(F,0,self.F_max) # Ensure appropriate bounds (main will otherwise treat negative F as positive)
        # Note: This approach is effectively equivalent to having one longitudinal pid controller which controls norm[ey/kt, eth/V/kt**2] but is just a cleaner representation
        # Todo: Slow down based on Cth? (slower when steering high to increase control authority)

        #print((int(100*delta*6/np.pi), int(100*F/15737)) )

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
