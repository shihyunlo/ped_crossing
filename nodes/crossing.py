#!/usr/bin/env python
import sys
sys.path.append('/home/slo/ws/src/ped_crossing/src')
import poly_traj_generation
from poly_traj_generation import PolyTrajGeneration
from poly_traj_generation_y import PolyTrajGenerationY

import math as m
import numpy as np
import math
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt

import rospy
import std_msgs.msg as std_msg
from nav_msgs.msg import Odometry
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import PoseStamped, Twist
import visualization_msgs.msg as vis_msg
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
#import gazebo_msgs.msg  as gazebo_msg

#from utils import linear_interp_traj
#from holonomic_controller import HolonomicController

#argv: python ped_crossing.py arg1 arg2 arg3 arg4 arg5 arg6
#arg1: helmet id for ped, arg2: chair id for ped_goal
#arg3: timing_sm, arg4: safety_sm, arg5: max_accelx (0.4), arg6: max_accely (0.4)
class PedCrossing :
    def __init__(self,sim_mode=True):
	# general parameters
	self.sim = sim_mode
	self.n_states = 10
	self.dt = 0.1 # traj publisher rate
        self.ped_helmet_id = sys.argv[1]
        self.ped_goal_chair_id = sys.argv[2]

        # ped/robot goal information
        self.ped_goal = np.zeros((2,1))
	self.rob_goal = np.zeros((2,1))
	self.vh = 1.1 #defult, to be estimate
	self.vr = 0.7 #nominal speed
	self.T = 16.0

	self.ped_pos = np.zeros((2,1))
        self.ped_vel = np.zeros((2,1))
        self.rob_pos = np.zeros((2,1))
        self.rob_vel = np.zeros((2,1))
	self.ped_pos_old = np.zeros((2,1))
	self.rob_pos_old = np.zeros((2,1))

        self.rob_intent = 0 # 0: pass first
        self.ped_intent = 1 # 1: yield

        self.rob_reaction_time = 3.0

        # polynomial traj generation
        self.poly_plan = 0
        self.ct = self.T/2
        self.timing_sm = 0.6
        self.safety_sm = 0.9
        # self.timing_sm = sys.argv[3]
        # self.safety_sm = sys.argv[4]

	self.max_accelx = 0.0
        self.max_accely = 0.4
        # self.max_accelx = sys.argv[5]
        # self.max_accely = sys.argv[6]
	self.recovery_gain = 4.0
	self.local_traj_duration = 4.0

        self.Xtraj = []
        self.Ytraj = []
        self.Xvel = []
        self.Yvel = []
        self.x_pos_ref = 0.0
        self.y_pos_ref = 0.0
        self.x_vel_ref = 0.0
        self.y_vel_ref = 0.0

	self.angular_vel_z = 0.0

	self.calibrated = False
	self.start = False
        #helmets and chairs:
	if self.sim:
            rospy.wait_for_service('/gazebo/get_model_state')

	self.num_obs = 3
        self.obs_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_pre_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_vel = np.full((self.num_obs, 2), np.inf)
        self.obs_pose_update_time = [np.inf, np.inf, np.inf, np.inf, np.inf]
        self.corner_pose = np.full((4,2), np.inf)

        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        #self.cmd_state_pub = rospy.Publisher('/cmd_state', Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('ped_pos', vis_msg.Marker, latch=False, queue_size=1)
        self.line_seg_pub = rospy.Publisher('path', vis_msg.Marker, latch=False, queue_size=1)
        self.robot_marker_pub = rospy.Publisher('robot', vis_msg.Marker, latch=False, queue_size=1)
        self.goal_marker_pub = rospy.Publisher('goal', vis_msg.Marker, latch=False, queue_size=1)

        if self.sim:
            self.get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.request = GetModelStateRequest()
            self.request.model_name = 'ballbot'







        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

        self.start_pre_x = 0.0
        self.start_pre_y = 0.0

        self.vx_ = 0.0
        self.vy_ = 0.0

        #self.goal_x = 0.0
        #self.goal_y = 0.0
        #self.goal_yaw = 0.0

        self.dt_mocap = 1.0 / 120.0
        self.robot_time_del = np.Inf
        self.robot_poly_index = np.Inf

        self.vel_msg = Twist()

        #self.holonomic_controller = HolonomicController()

        #if self.sim:
        #    rospy.wait_for_service('/gazebo/get_model_state')


	self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.helmet00_sub = rospy.Subscriber('/vrpn_client_node/HelmetXL/pose', PoseStamped, self.helmet00_callback, queue_size=1)
        self.helmet01_sub = rospy.Subscriber('/vrpn_client_node/HelmetM/pose', PoseStamped, self.helmet01_callback, queue_size=1)
        self.helmet02_sub = rospy.Subscriber('/vrpn_client_node/HelmetS/pose', PoseStamped, self.helmet02_callback, queue_size=1)

        self.corner00_sub = rospy.Subscriber('/vrpn_client_node/corner1/pose', PoseStamped, self.corner00_callback, queue_size=1)
        self.corner01_sub = rospy.Subscriber('/vrpn_client_node/corner2/pose', PoseStamped, self.corner01_callback, queue_size=1)
        self.corner02_sub = rospy.Subscriber('/vrpn_client_node/corner3/pose', PoseStamped, self.corner02_callback, queue_size=1)
        self.corner03_sub = rospy.Subscriber('/vrpn_client_node/corner4/pose', PoseStamped, self.corner03_callback, queue_size=1)

        self.robot_vrpn_pose_sub = rospy.Subscriber('/vrpn_client_node/RockHopper6/pose', PoseStamped, self.robot_vrpn_pose_callback, queue_size=1)
        #self.robot_gazebo_pose_sub = rospy.Subscriber('/gazebo/model_states', PoseStamped, self.robot_vrpn_pose_callback, queue_size=1)

        self.goal_pose_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_pose_callback, queue_size=1)

        self.run()

    def helmet00_callback(self, pose_msg):
        idx = 0
        self.obs_pose[idx, 0] = pose_msg.pose.position.x
        self.obs_pose[idx, 1] = pose_msg.pose.position.y
        current_time = time.time()
        if self.obs_pre_pose[idx, 0] != np.inf and self.obs_pre_pose[idx, 1] != np.inf and self.obs_pose_update_time[idx] != np.inf:
            # delta_time = current_time - self.obs_pose_update_time[0]
            self.obs_vel[idx] = (self.obs_pose[idx] - self.obs_pre_pose[idx]) / self.dt_mocap
        self.obs_pose_update_time[idx] = current_time
        self.obs_pre_pose[idx, 0] = pose_msg.pose.position.x
        self.obs_pre_pose[idx, 1] = pose_msg.pose.position.y

    def helmet01_callback(self, pose_msg):
        idx = 1
        self.obs_pose[idx, 0] = pose_msg.pose.position.x
        self.obs_pose[idx, 1] = pose_msg.pose.position.y
        current_time = time.time()
        if self.obs_pre_pose[idx, 0] != np.inf and self.obs_pre_pose[idx, 1] != np.inf and self.obs_pose_update_time[idx] != np.inf:
            # delta_time = current_time - self.obs_pose_update_time[0]
            self.obs_vel[idx] = (self.obs_pose[idx] - self.obs_pre_pose[idx]) / self.dt_mocap
        self.obs_pose_update_time[idx] = current_time
        self.obs_pre_pose[idx, 0] = pose_msg.pose.position.x
        self.obs_pre_pose[idx, 1] = pose_msg.pose.position.y

    def helmet02_callback(self, pose_msg):
        idx = 2
        self.obs_pose[idx, 0] = pose_msg.pose.position.x
        self.obs_pose[idx, 1] = pose_msg.pose.position.y
        current_time = time.time()
        if self.obs_pre_pose[idx, 0] != np.inf and self.obs_pre_pose[idx, 1] != np.inf and self.obs_pose_update_time[idx] != np.inf:
            # delta_time = current_time - self.obs_pose_update_time[0]
            self.obs_vel[idx] = (self.obs_pose[idx] - self.obs_pre_pose[idx]) / self.dt_mocap
        self.obs_pose_update_time[idx] = current_time
        self.obs_pre_pose[idx, 0] = pose_msg.pose.position.x
        self.obs_pre_pose[idx, 1] = pose_msg.pose.position.y

    def corner00_callback(self, pose_msg):
        idx = 0
        self.corner_pose[idx, 0] = pose_msg.pose.position.x
        self.corner_pose[idx, 1] = pose_msg.pose.position.y

    def corner01_callback(self, pose_msg):
        idx = 1
        self.corner_pose[idx, 0] = pose_msg.pose.position.x
        self.corner_pose[idx, 1] = pose_msg.pose.position.y

    def corner02_callback(self, pose_msg):
        idx = 2
        self.corner_pose[idx, 0] = pose_msg.pose.position.x
        self.corner_pose[idx, 1] = pose_msg.pose.position.y

    def corner03_callback(self, pose_msg):
        idx = 3
        self.corner_pose[idx, 0] = pose_msg.pose.position.x
        self.corner_pose[idx, 1] = pose_msg.pose.position.y
    def odom_callback(self, pose_msg):
        self.angular_vel_z = pose_msg.twist.twist.angular.x

    def robot_vrpn_pose_callback(self, pose_msg):
        if pose_msg.pose.position.z > 0.05 :
            self.start_x = pose_msg.pose.position.x
            self.start_y = pose_msg.pose.position.y
            q = pose_msg.pose.orientation
            self.start_yaw = math.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

            self.vx_ = (self.start_x - self.start_pre_x) / self.dt_mocap
            self.vy_ = (self.start_y - self.start_pre_y) / self.dt_mocap

            self.start_pre_x = self.start_x
            self.start_pre_y = self.start_y
	    
	    if not self.sim:
	        self.rob_pos[0] = pose_msg.pose.position.x
	        self.rob_pos[1] = pose.msg.pose.position.y

    def goal_pose_callback(self, goal_pose_msg):
        self.goal_x = goal_pose_msg.pose.position.x
        self.goal_y = goal_pose_msg.pose.position.y
        q = goal_pose_msg.pose.orientation
        #self.goal_yaw = math.atan2(dd2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

    def run(self):
	cmd_pub_rate = 10
        rate = rospy.Rate(cmd_pub_rate)
        vel_count = 0
        vel_n = 5
        vel_est = np.zeros((2,vel_n))
        if self.sim:
	        ## set-up callback values for planning

	    self.rob_pos[0] = self.start_x
            self.rob_pos[1] = self.start_y
            self.ped_vel[0] = 0.0
            self.ped_vel[1] = self.vh
            self.rob_vel[0] = self.vr
            self.rob_vel[1] = 0.0
	    self.rob_goal[0] = self.rob_pos[0] + self.T*self.rob_vel[0]
	    self.rob_goal[1] = self.rob_pos[1] + self.T*self.rob_vel[1]
	    self.ped_pos[0] = (self.rob_pos[0]+self.rob_goal[0])/2 - (self.T/2-0.1)*self.ped_vel[0]
	    self.ped_pos[1] = (self.rob_pos[1]+self.rob_goal[1])/2 - (self.T/2-0.1)*self.ped_vel[1]
            self.ped_goal[0] = self.ped_pos[0] + self.T*self.ped_vel[0]
            self.ped_goal[1] = self.ped_pos[1] + self.T*self.ped_vel[1]
	 
        vel_msg_ = Twist()
	goal_thr = 0.4
	rate_count = int(cmd_pub_rate/(1/self.dt))
	
        while (not rospy.is_shutdown()) and (np.linalg.norm(self.rob_pos-self.rob_goal)>goal_thr):
#	        else:
#                self.ped_goal[0] = -0.2
#                self.ped_goal[1] = 8.71
#		        self.rob_goal[0] = 4.6
#                self.rob_goal[1] = 2.0

#		        self.ped_pos[0] = -0.2
#                self.ped_pos[1] = -4.49
#		        self.rob_pos[0] = -5.0
#                self.rob_pos[1] = 2.0

#                self.ped_vel[0] = 0.0
#                self.ped_vel[1] = 1.1
#		        self.rob_vel[0] = 0.8
#                self.rob_vel[1] = 0.0
            #    q = model_state_result.pose.orientation

            #    self.start_x = model_state_result.pose.position.x
            #    self.start_y = model_state_result.pose.position.y
            #    self.start_yaw = math.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

            #    self.vx = model_state_result.twist.linear.x
            #    self.vy = model_state_result.twist.linear.y

            #waypoints, lin_x, lin_y, ind, obs_lin_x, obs_lin_y = self.straighline_planner(self.obs_pose)
            #vel_est[0,vel_count] = self.vx_
            #vel_est[1,vel_count] = self.vy_
	    #if not self.calibrated :
#		criteria1 = (self.ped_goal[0]!=0.0)
#		criteria2 = (self.ped_pos[0]!=0.0)
#		criteria3 = (self.rob_goal[0]!=0.0)
#		pub 100hz
#		if criteria1 and criteria2 and criteria3:
#			self.calibrated = 1
#		else :
#			rate.sleep()
#			continue		
	    
	    

	    if not self.start :
	        raw_input('Environment Ready, Press Any Key to Start')
	        self.start = True

	        

	    if self.sim :
		# gazebo
                model_state_result = self.get_model_srv(self.request)
                q = model_state_result.pose.orientation
                self.rob_pos[0] = model_state_result.pose.position.x
                self.rob_pos[1] = model_state_result.pose.position.y
                self.rob_yaw = math.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
                self.rob_vel[0] = model_state_result.twist.linear.x
                self.rob_vel[1] = model_state_result.twist.linear.y

		
#	    if rate_count < int(cmd_pub_rate/(1/self.dt)) :
#		rate_count += 1
#		# TODO: state_estimate
#		rate.sleep()
#		continue
#	    else :
#	        self.velocity_pub.publish(self.vel_msg)
#		rate_count = 0
#		rate_count += 1
		
            self.vh = np.linalg.norm(self.ped_vel)

            pR = self.rob_pos
	    vR = self.rob_vel
	    pH = self.ped_pos
	    vH = self.ped_vel

            vH_ = (self.ped_goal - pH)/np.linalg.norm(self.ped_goal - pH)*self.vh
            vR_ = (self.rob_goal - pR)/np.linalg.norm(self.rob_goal - pR)*self.vr
	        # collision timing check
	    p_rel = -pR+pH
            A = np.zeros((2,2))
	    A[0][0] = vR_[0]
	    A[1][0] = vR_[1]
	    A[0][1] = -vH_[0]
	    A[1][1] = -vH_[1]

	    #if np.linalg.matrix_rank(A)<3 :
		#print 'vH_ = {}'.format([vH_[0], vH_[1]])
		#print 'pH = {}'.format([pH[0],pH[1]])
		#print 'ped_goal = {}'.format([self.ped_goal[0],self.ped_goal[1]])
		#print 'vh = {}'.format(self.vh)
 	    tbc = np.linalg.solve(A,p_rel)
	    tbc_rob = tbc[0]
	    tbc_ped = tbc[1]

            time_ = 0.0

	    if self.poly_plan==0 :
                vR_ = vR_*tbc_rob/(tbc_ped-self.timing_sm/self.vh)
		self.x_pos_ref  = vR_[0]*self.dt + self.rob_pos[0]
		self.y_pos_ref  = vR_[1]*self.dt + self.rob_pos[1]
		self.x_vel_ref = vR_[0]
		self.y_vel_ref = vR_[1]
		#print '{}'.format(tbc_ped-tbc_rob-self.timing_sm/self.vh)
	        start_time = 0.0

                if tbc_ped < self.rob_reaction_time :
            	    self.local_traj_duration = np.copy(tbc_ped)
		    start_time = time.time()
		    # poly_traj_generation
		    time_shift = 0.5
	            action, pplan, xtraj, ytraj, xvel, yvel = PolyTrajGenerationY(time_shift,self.rob_pos, self.rob_vel, self.rob_intent, self.ped_pos, self.ped_vel, self.local_traj_duration, self.dt, time_, self.rob_reaction_time, self.max_accelx, self.max_accely, self.timing_sm, self.safety_sm, self.ped_goal, self.rob_goal, self.vh, self.vr, self.n_states)
		    print 'x_traj = {}'.format(xtraj)
		    print 'x_vel = {}'.format(xvel)
		    print 'y_traj = {}'.format(ytraj)
		    print 'y_vel = {}'.format(yvel)
		    #if self.max_accelx < 0.05 and self.max_accely <0.05 :
		    for l in range(0,1):
		        self.Xtraj = np.copy(xtraj[l])
                        self.Ytraj = np.copy(ytraj[l])
                        self.Xvel = np.copy(xvel[l])
                        self.Yvel = np.copy(yvel[l])
	        	terminate_index = len(self.Xvel)-max(sum(abs(self.Xvel)>5.0),sum(abs(self.Yvel)>5.0))
			print 'max x vel = {}'.format(self.Xvel[0:terminate_index])
			print 'max y vel = {}'.format(self.Yvel[0:terminate_index])
			


                    self.poly_plan = pplan
                    self.robot_time_del = 0#
                    self.robot_poly_index = 0#
		    ratio = 1.1/max(self.Yvel[0:20])
		    ratio =1.1/0.54
		    vR_baseline = vR_ + vH_/ratio
	    time_now = time.time()
	    #if (time_now-start_time<self.local_traj_duration-1) and tbc_ped< self.rob_reaction_time:
		#vR_ = vR_baseline
		
			
			    
	    
	    lookahead = int(0.2/self.dt)
	    lookahead = 0
	    early_terminate = 15
	    #print 'self.Xvel = {}'.format(self.Xvel)
	    #print 'self.Xvel>5.0 = {}'.format(self.Xvel>5.0)
	    if len(self.Xvel)==0 :
	        terminate_index = 0
	    #compute commands: cmd_state or vel_cmd
            #if self.robot_poly_index < (self.local_traj_duration/self.dt-lookahead-early_terminate) :
            if self.robot_poly_index < terminate_index :
                self.x_pos_ref = np.copy(self.Xtraj[self.robot_poly_index]+lookahead)
                self.y_pos_ref = np.copy(self.Ytraj[self.robot_poly_index]+lookahead)
                self.x_vel_ref = np.copy(self.Xvel[self.robot_poly_index])
                self.y_vel_ref = np.copy(self.Yvel[self.robot_poly_index])
		#print 'poly tracking x_pos_ref = {}'.format(self.x_pos_ref)
                x_pos_ref_old = np.copy(self.Xtraj[max(0,self.robot_poly_index-1)])
                y_pos_ref_old = np.copy(self.Ytraj[max(0,self.robot_poly_index-1)])
                
		vel_msg_.linear.x = self.x_vel_ref
       	        vel_msg_.linear.y = self.y_vel_ref
                vel_msg_.linear.x = 1.0*(self.x_pos_ref-self.rob_pos[0])/self.dt
       	        vel_msg_.linear.y = 1.0*(self.y_pos_ref-self.rob_pos[1])/self.dt
                vel_msg_.linear.x = 0.2*(self.x_pos_ref-self.rob_pos[0])/self.dt + 0.8*self.x_vel_ref
       	        vel_msg_.linear.y = 0.2*(self.y_pos_ref-self.rob_pos[1])/self.dt + 0.8*self.y_vel_ref
                #self.vel_msg.linear.x = 0.2*(self.x_pos_ref-y_pos_ref_old)/self.dt + 0.8*self.x_vel_ref
       	        #self.vel_msg.linear.y = 0.2*(self.y_pos_ref-y_pos_ref_old)/self.dt + 0.8*self.y_vel_ref
	    
            else :
                a_rob = self.recovery_gain*(vR_-vR)
                self.x_vel_ref = vR_[0]
                self.y_vel_ref = vR_[1]
                self.x_pos_ref = self.rob_pos[0] + vR_[0]*self.dt
                self.y_pos_ref = self.rob_pos[1] + vR_[1]*self.dt
		#print 'vR_ tracking = {}'.format(vR_)
                vel_msg_.linear.x = 1*self.x_vel_ref
       	        vel_msg_.linear.y = 1*self.y_vel_ref
                #self.vel_msg.linear.x = 1.0*self.x_vel_ref + 2.0*self.dt*(vR_[0]-vR[0])
       	        #self.vel_msg.linear.y = 1.0*self.y_vel_ref + 2.0*self.dt*(vR_[1]-vR[1])
           	
	    self.vel_msg.linear.x = vel_msg_.linear.x*m.cos(self.rob_yaw) - vel_msg_.linear.y*m.sin(self.rob_yaw)
	    self.vel_msg.linear.y = vel_msg_.linear.x*m.sin(self.rob_yaw) + vel_msg_.linear.y*m.cos(self.rob_yaw)
            omega = 1
	    damping = 1
	
	   	
	    self.vel_msg.angular.z = omega*(self.start_yaw-self.rob_yaw) + 2*damping*omega*(0-self.angular_vel_z)
	    self.velocity_pub.publish(self.vel_msg)
            
	    # forward simulation
	    if self.sim :
                #self.rob_pos[0] = self.x_pos_ref
                #self.rob_pos[1] = self.y_pos_ref
                self.ped_pos[0] += vH_[0]*self.dt
                self.ped_pos[1] += vH_[1]*self.dt
		#self.rob_vel[0] = vR_[0]
		#self.rob_vel[1] = vR_[1]
		self.ped_vel[0] = vH_[0]
		self.ped_vel[1] = vH_[1]
	    else :		
            	if vel_count== 0 :
                    self.rob_pos_old[0] = self.rob_pos[0]
                    self.rob_pos_old[1] = self.rob_pos[1]
                    self.ped_pos_old[0] = self.ped_pos[0]
                    self.ped_pos_old[1] = self.ped_pos[1]

                if vel_count>=(vel_n-1) :
                    self.rob_vel = (self.rob_pos-self.rob_pos_old)/self.dt_mocap/vel_n
                    self.ped_vel = (self.ped_pos-self.ped_pos_old)/self.dt_mocap/vel_n
                    vel_count = 0
		    #print 'pos_old = {}'.format(pos_ped_old)
		    #print 'pos_new = {}'.format(self.ped_pos)
		    #print 'vH = {}'.format(self.ped_vel)

            vel_count = vel_count + 1

	    #plt.clf()
            plt.axis('equal')
            x_min = -6
            x_max = 12
            y_min = -8
            y_max = 8
            plt.xlim((x_min, x_max))
            plt.ylim((y_min, y_max))
            plt.grid(True)
            plt.autoscale(False)
            for i in range(0, self.num_obs):
                if self.obs_pose[i, 0] != np.inf and self.obs_pose[i, 1] != np.inf:
                    plt.plot(self.obs_pose[i, 0], self.obs_pose[i, 1], 'or')
                    plt.text(self.obs_pose[i, 0], self.obs_pose[i, 1], str(i), color='r')
                if self.obs_vel[i, 0] != np.inf and self.obs_vel[i, 1] != np.inf:
                    plt.arrow(self.obs_pose[i, 0], self.obs_pose[i, 1], self.obs_vel[i, 0], self.obs_vel[i, 1], head_width=0.05, head_length=0.08, fc='k')
            # plot chairs
            plt.plot(self.corner_pose[:, 0], self.corner_pose[:, 1], 'k')
            plt.plot([self.corner_pose[-1, 0], self.corner_pose[0, 0]], [self.corner_pose[-1, 1], self.corner_pose[0, 1]], 'k')
            # plot robot
            #plt.plot(self.start_x, self.start_y, 'or')
	    plt.plot(self.ped_pos[0],self.ped_pos[1],'ok')
	    plt.plot(self.rob_pos[0],self.rob_pos[1],'om')
	    if (len(self.Xtraj)>0) :
		plt.plot(self.Xtraj[0:terminate_index],self.Ytraj[0:terminate_index],'r')
	    else :
		x_local_ref = np.zeros(2)
		y_local_ref = np.zeros(2)
		x_local_ref[0] = self.x_pos_ref
		x_local_ref[1] = self.rob_pos[0]	
		y_local_ref[0] = self.y_pos_ref
		y_local_ref[1] = self.rob_pos[1]	

            #plt.arrow(self.rob_pos[0], self.rob_pos[1], vR_[0], vR_[1], head_width=0.05, head_length=0.08, fc='k')
            

#plt.plot(self.goal_x, self.goal_y, 'ob')
            #plt.arrow(self.start_x, self.start_y, 0.3 * math.cos(self.start_yaw), 0.3 * math.sin(self.start_yaw), head_width=0.02, head_length=0.08, fc='r', ec='k')
            #plt.arrow(self.start_x, self.start_y, 0.3 * -math.sin(self.start_yaw), 0.3 * math.cos(self.start_yaw), head_width=0.02, head_length=0.08, fc='g', ec='k')
            #plt.arrow(self.start_x, self.start_y, obs_lin_x * math.cos(self.start_yaw), obs_lin_x * math.sin(self.start_yaw), head_width=0.05, head_length=0.08, fc='r', ec='k')
            #plt.arrow(self.start_x, self.start_y, obs_lin_y * -math.sin(self.start_yaw), obs_lin_y * math.cos(self.start_yaw), head_width=0.05, head_length=0.08, fc='g', ec='k')
            ## plot goal
            #plt.arrow(self.goal_x, self.goal_y, 0.3 * math.cos(self.goal_yaw), 0.3 * math.sin(self.goal_yaw), head_width=0.02, head_length=0.08, fc='r', ec='k')
            #plt.arrow(self.goal_x, self.goal_y, 0.3 * -math.sin(self.goal_yaw), 0.3 * math.cos(self.goal_yaw), head_width=0.02, head_length=0.08, fc='g', ec='k')
            #plt.plot(waypoints[:, 0], waypoints[:, 1], 'or')
            #plt.plot(waypoints[ind, 0], waypoints[ind, 1], 'og')

            #plt.draw()

            self.robot_poly_index += 1
            self.robot_time_del += self.dt
            plt.pause(0.05)
	    #plt.hold

            rate.sleep()

    def straighline_planner(self, obs=None):
        waypoints = linear_interp_traj([[self.start_x, self.start_y], [self.goal_x, self.goal_y]], 0.1)
        waypoints = np.array(waypoints)

        lin_x, lin_y, angu_z, ind, obs_lin_x, obs_lin_y = self.holonomic_controller.control(waypoints, self.start_x, self.start_y, self.start_yaw, self.goal_yaw, self.vel_rob[0], self.vel_rob[1], obs)

        self.vel_msg.linear.x = lin_x
        self.vel_msg.linear.y = lin_y
        self.vel_msg.angular.z = 0.0
        # self.vel_msg.angular.z = angu_z
        # self.velocity_pub.publish(self.vel_msg)

        robot_marker = vis_msg.Marker(type=vis_msg.Marker.SPHERE_LIST, action=vis_msg.Marker.ADD)
        robot_marker.header.frame_id = 'world'
        robot_marker.header.stamp = rospy.Time.now()
        robot_marker.scale.x = 0.5
        robot_marker.scale.y = 0.5
        robot_marker.points = [geom_msg.Point(self.start_x, self.start_y, 0.0)]
        robot_marker.colors = [std_msg.ColorRGBA(1.0, 0.0, 0.0, 1.0)]
        self.robot_marker_pub.publish(robot_marker)

        goal_marker = vis_msg.Marker(type=vis_msg.Marker.SPHERE_LIST, action=vis_msg.Marker.ADD)
        goal_marker.header.frame_id = 'world'
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.scale.x = 0.5
        goal_marker.scale.y = 0.5
        goal_marker.points = [geom_msg.Point(self.goal_x, self.goal_y, 0.0)]
        goal_marker.colors = [std_msg.ColorRGBA(0.0, 0.0, 1.0, 1.0)]
        self.goal_marker_pub.publish(goal_marker)

        return waypoints, lin_x, lin_y, ind, obs_lin_x, obs_lin_y
    
if __name__ == '__main__':
    rospy.init_node('ped_crossing_node')
    #arg1: helmet id for ped, arg2: chair id for ped_goal
    #arg3: timing_sm, arg4: safety_sm, arg5: max_accelx (0.4), arg6: max_accely (0.4)
    ped_crossing = PedCrossing()
    rospy.spin()
