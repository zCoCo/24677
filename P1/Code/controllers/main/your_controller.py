# Fill in the respective functions to implement the controller

# Import libraries
import numpy as np
from base_controller import BaseController
from scipy import signal, linalg
from util import *

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
        
        self.time = 0 # [s] Current time into trajectory execution
        self.target_time = 300 # [s] Target time to complete loop.
        self.track_length = 1290.385282704354; # [m] Total path length of the track
        
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
        Ku_lat = 0.60*0.61
        Tu_lat = 6
        self.k_pid_longitudinal = np.asarray([0.60*Ku_long, 1.2*Ku_long/Tu_long, 3*Ku_long*Tu_long/40]).reshape((1,3)) #np.asarray([163500,0,0]).reshape((1,3))#np.asarray([1.2*T_long/L_long, 1.2*T_long/L_long/(2.0*L_long), 1.2*T_long/L_long*0.5*L_long]).reshape((1,3))#np.asarray([0.60*Ku, 1.2*Ku/Tu, 3*Ku*Tu/40]).reshape((1,3))
        self.k_pid_lateral = np.asarray([0.60*Ku_lat, 1.2*Ku_lat/Tu_lat, 3*Ku_lat*Tu_lat/40]).reshape((1,3))#np.asarray([500,0,0]).reshape((1,3))#np.asarray([1,0,0]).reshape((1,3))#
        
        self.look_ahead_multiple = 2.5 # How many car lengths to look ahead and average over when determining desired heading angle
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
        nearest_waypoint_fwd = trajectory[(nearest_waypoint_idx+self.look_ahead_indices) % trajectory.shape[0]] # look roughly 2.5 car lengths ahead
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
        speed_diff = self.track_length / self.target_time - xdot
        
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
            ky = 2 / 0.05 / (kt**2); 
        else:
            ky = 2 / V / (kt**2); # Lessen impact of crosstrack error as velocity increases (to minimize wild swings)
        
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
        delta = Cy + Cth; # Allow crosstrack and heading errors to drive the steering angle
        # Note: This approach is effectively equivalent to having one lateral pid controller which controls (ey/kt + eth/V/kt**2) but is just a cleaner representation

        # ---------------|Longitudinal Controller|-------------------------
        F = np.linalg.norm([clamp(Cx,0,15737), Cy]); # Allow crosstrack and alongtrack errors to drive the throttle
        # Note: This approach is effectively equivalent to having one longitudinal pid controller which controls norm[ey/kt, eth/V/kt**2] but is just a cleaner representation

        # Return all states and calculated control inputs (F, delta)
        return X, Y, xdot, ydot, psi, psidot, F, delta
