#!/usr/bin/env python
import sys
sys.path.append('/home/researcher_ri/ws_adelie/src/ped_crossing/src')
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
#arg1: helmet id for ped, arg2: chair id for ped_goal arg3: robot goal chair id
#arg4: timing_sm, arg5: safety_sm, arg6: max_accelx (0.4), arg7: max_accely (0.4)
class PedCrossing :
    def __init__(self,sim_mode=True):
	# general parameters
	self.sim = sim_mode
	self.n_states = 10
	self.dt = 0.1 # traj publisher rate
        self.ped_helmet_id = int(sys.argv[1])
        self.ped_goal_chair_id = int(sys.argv[2])
        self.rob_goal_chair_id = int(sys.argv[3])

        # ped/robot goal information
        self.ped_goal = np.zeros((2,1))
	self.rob_goal = np.zeros((2,1))
	self.vh = 1.1 #defult, to be estimate
	self.vr = 0.7 #nominal speed
	self.T = 16.0

	self.ped_pos = np.zeros((2,1))
	self.ped_pos = np.full((2,1),np.inf)
        self.ped_vel = np.zeros((2,1))
        self.rob_pos = np.zeros((2,1))
        self.rob_vel = np.zeros((2,1))
	self.ped_pos_old = np.zeros((2,1))
	self.rob_pos_old = np.zeros((2,1))

        self.rob_intent = 0 # 0: pass first
        self.ped_intent = 1 # 1: yield

        self.rob_reaction_time = 4.5

        # polynomial traj generation
        self.poly_plan = 0
        self.ct = self.T/2
        self.timing_sm = 1.5
        self.safety_sm = 2.4
        # self.timing_sm = sys.argv[3]
        # self.safety_sm = sys.argv[4]

	self.max_accelx = 0.0
        self.max_accely = 0.1
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
	#obs_time
	self.obs_time = 0.0
	self.obs_time_sec = 0
	self.obs_time_nsec = 0
        self.obs_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_pre_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_vel = np.full((self.num_obs, 2), np.inf)
        self.obs_pose_update_time = [np.inf, np.inf, np.inf, np.inf, np.inf]
        self.corner_pose = np.full((4,2), np.inf)

        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        #self.cmd_state_pub = rospy.Publisher('/cmd_state', Twist, queue_size=1)
        #self.marker_pub = rospy.Publisher('ped_pos', vis_msg.Marker, latch=False, queue_size=1)
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

        self.dt_mocap = 1.0 / 120.0
        self.robot_time_del = np.Inf
        self.robot_poly_index = np.Inf

        self.vel_msg = Twist()


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
	#print 'helmet location = {}'.format(pose_msg.pose.position.y)
	time_del = -(self.obs_time_sec - pose_msg.header.stamp.secs) - (10**(-9))*(self.obs_time_nsec - pose_msg.header.stamp.nsecs)
	#print 'time_del = {}'.format(time_del)
	self.obs_time_sec = np.copy(pose_msg.header.stamp.secs)
	self.obs_time_nsec = np.copy(pose_msg.header.stamp.nsecs)
        if not m.isnan(self.obs_pose[idx, 0]) and not m.isnan(self.obs_pose[idx, 1]) and m.isnan(pose_msg.pose.position.x)!=True and m.isnan(pose_msg.pose.position.y)!=True:
	    if self.obs_time_sec!=0 and time_del>0.005:
                self.obs_vel[idx,0] = (pose_msg.pose.position.x - self.obs_pose[idx,0]) / time_del
                self.obs_vel[idx,1] = (pose_msg.pose.position.y - self.obs_pose[idx,1]) / time_del
		#print 'self.obs_vel[idx,1],{}'.format(self.obs_vel[idx,1])
            self.obs_pose[idx, 0] = np.copy(pose_msg.pose.position.x)
            self.obs_pose[idx, 1] = np.copy(pose_msg.pose.position.y)

    def helmet01_callback(self, pose_msg):
        idx = 1
	time_del = -(self.obs_time_sec - pose_msg.header.stamp.secs) - (10**(-9))*(self.obs_time_nsec - pose_msg.header.stamp.nsecs)
	#print 'time_del = {}'.format(time_del)
	self.obs_time_sec = np.copy(pose_msg.header.stamp.secs)
	self.obs_time_nsec = np.copy(pose_msg.header.stamp.nsecs)
        if not m.isnan(self.obs_pose[idx, 0]) and not m.isnan(self.obs_pose[idx, 1]) and m.isnan(pose_msg.pose.position.x)!=True and m.isnan(pose_msg.pose.position.y)!=True:
	    if self.obs_time_sec!=0 and time_del>0.005:
                self.obs_vel[idx,0] = (pose_msg.pose.position.x - self.obs_pose[idx,0]) / time_del
                self.obs_vel[idx,1] = (pose_msg.pose.position.y - self.obs_pose[idx,1]) / time_del
		#print 'self.obs_vel[idx,1],{}'.format(self.obs_vel[idx,1])
            self.obs_pose[idx, 0] = np.copy(pose_msg.pose.position.x)
            self.obs_pose[idx, 1] = np.copy(pose_msg.pose.position.y)


    def helmet02_callback(self, pose_msg):
        idx = 2
	time_del = -(self.obs_time_sec - pose_msg.header.stamp.secs) - (10**(-9))*(self.obs_time_nsec - pose_msg.header.stamp.nsecs)
	#print 'time_del = {}'.format(time_del)
	self.obs_time_sec = np.copy(pose_msg.header.stamp.secs)
	self.obs_time_nsec = np.copy(pose_msg.header.stamp.nsecs)
        if not m.isnan(self.obs_pose[idx, 0]) and not m.isnan(self.obs_pose[idx, 1]) and m.isnan(pose_msg.pose.position.x)!=True and m.isnan(pose_msg.pose.position.y)!=True:
	    if self.obs_time_sec!=0 and time_del>0.005:
                self.obs_vel[idx,0] = (pose_msg.pose.position.x - self.obs_pose[idx,0]) / time_del
                self.obs_vel[idx,1] = (pose_msg.pose.position.y - self.obs_pose[idx,1]) / time_del
		#print 'self.obs_vel[idx,1],{}'.format(self.obs_vel[idx,1])
            self.obs_pose[idx, 0] = np.copy(pose_msg.pose.position.x)
            self.obs_pose[idx, 1] = np.copy(pose_msg.pose.position.y)

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
	    self.rob_pos[0] = pose_msg.pose.position.x
	    self.rob_pos[1] = pose_msg.pose.position.y
	    self.rob_yaw = self.start_yaw

    def goal_pose_callback(self, goal_pose_msg):
        self.goal_x = goal_pose_msg.pose.position.x
        self.goal_y = goal_pose_msg.pose.position.y
        q = goal_pose_msg.pose.orientation
        #self.goal_yaw = math.atan2(dd2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

    def run(self):
	#initialization: poses, goals
        if self.sim:
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
	else :
	    self.rob_vel[0] = 0
	    self.rob_vel[1] = 0
	    pos_ = np.copy(self.obs_pose[self.ped_helmet_id])
	    if not math.isnan(pos_[0]):
	        self.ped_pos[0] = np.copy(pos_[0])
	    if not math.isnan(pos_[1]):
	        self.ped_pos[1] = np.copy(pos_[1])
	    #self.ped_pos[1] = np.copy(self.obs_pose[self.ped_helmet_id,1])
	    #self.ped_pos[0] = np.copy(self.obs_pose[self.ped_helmet_id,0])
	
	start_yaw = 0.0
	#initialization: system level
	cmd_pub_rate = 100
        rate = rospy.Rate(cmd_pub_rate)
	rate_count = int(cmd_pub_rate/(1/self.dt))
        vel_msg_ = Twist()
        
	plot_count = 0
	vel_count = 0
        vel_n = 5
        vel_est = np.zeros((2,vel_n))
	goal_thr = 2.5
	initial_time = time.time()
	program_start_time = time.time()
	robot_time_del = 1/self.dt

	#initialization: pedestrian velcity estimate, robot fixed command
	vR_ = np.zeros(2)
	vR_baseline = np.zeros(2)
	vR_accel = np.zeros(2)
	vh_filtered = 0.0
	vh_filtered_old = 0.0
	vh_fixed = 0
	vh_fixed_count = 0
	vh_fixed_value = 1.0
	tbc_ped = np.inf
	vR_fixed = np.zeros(2)
	convergence_count = 0
        while (not rospy.is_shutdown()) :
	   #initialization: mocap communication checks
	    if not self.start :
		
		current_time = time.time()
		robot_time_del = current_time - initial_time
		criteria0 = self.rob_pos[1]!=0.0
		pos_ = np.copy(self.obs_pose[self.ped_helmet_id])
		if not math.isnan(pos_[0]):
	            self.ped_pos[0] = np.copy(pos_[0])
		if not math.isnan(pos_[1]):
	            self.ped_pos[1] = np.copy(pos_[1])
	        
		criteria1 = (self.ped_pos[1]!=np.inf and self.ped_pos[0]!=np.inf)
		
		self.ped_goal[1] = self.corner_pose[self.ped_goal_chair_id,1]
		self.ped_goal[0] = self.corner_pose[self.ped_goal_chair_id,0]
		criteria2 = (self.ped_goal[1]!=np.inf and self.ped_goal[0]!=np.inf)

		self.rob_goal[1] = self.corner_pose[self.rob_goal_chair_id,1]
		self.rob_goal[0] = self.corner_pose[self.rob_goal_chair_id,0]
		criteria3 = (self.rob_goal[1]!=np.inf and self.rob_goal[0]!=np.inf)
		print'rob_pos = {}'.format(self.rob_pos)
		if criteria1 :	
		    print 'criteria1 fixed'
		    print'obs_pose = {}'.format(self.obs_pose)
		    print'obs_pos_target = {}'.format(self.obs_pose[self.ped_helmet_id,1])
		    print'ped_pos = {}'.format(self.ped_pos)
		if criteria2 :	
		    print 'criteria2 fixed'
		if criteria3 :	
		    print 'criteria3 fixed'
		if criteria0 :	
		    print 'criteria0 fixed'
		    
		if criteria0 and criteria1 and criteria2 and criteria3 :	
		    initial_rob_pos = np.copy(self.rob_pos)
		    initial_ped_pos = np.copy(self.ped_pos)
		    start_yaw = np.copy(self.start_yaw)
		    print 'start yaw = {}'.format(start_yaw)
		    
                    goal_delta = -1*(self.rob_goal - self.rob_pos)/np.linalg.norm(self.rob_goal - self.rob_pos)*1.0
		    self.rob_goal = self.rob_goal-goal_delta
	            raw_input('Environment Ready, Press Any Key to Start')
		    print'rob_goal, ped_goal, rob_pos, ped_pos = {}{}{}{}'.format(self.rob_goal,self.ped_goal,self.rob_pos,self.ped_pos)
		    print'ped_pos = {}'.format(self.ped_pos[1])
		    print'ped_pos = {}'.format(self.obs_pose[self.ped_helmet_id,1])
	            self.start = True
		else :
		    continue


	        
	    #initialization: sanity check on pedestrian readings
	    if self.sim :
		# gazebo
                model_state_result = self.get_model_srv(self.request)
                q = model_state_result.pose.orientation
                self.rob_pos[0] = model_state_result.pose.position.x
                self.rob_pos[1] = model_state_result.pose.position.y
                self.rob_yaw = math.atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)
                self.rob_vel[0] = model_state_result.twist.linear.x
                self.rob_vel[1] = model_state_result.twist.linear.y
	    else: 
	
		pos_ = np.copy(self.obs_pose[self.ped_helmet_id])
		if not math.isnan(pos_[0]):
	            self.ped_pos[0] = np.copy(pos_[0])
		else:
		    continue
		if not math.isnan(pos_[1]):
	            self.ped_pos[1] = np.copy(pos_[1])
		else :
		    continue
		    
		current_time = time.time()
		current_ped_pos = np.copy(self.ped_pos)
		current_rob_pos = np.copy(self.rob_pos)
		robot_time_del = current_time - initial_time

		

		## Velocity Filtering: checks on measurement values
	        vh_thr = 0.6
	        if not current_ped_pos[0]==initial_ped_pos[0] and not np.linalg.norm(self.obs_vel[self.ped_helmet_id])>1.8:
		    ped_vel_est_ = self.obs_vel[self.ped_helmet_id]
		    self.ped_vel =np.copy(ped_vel_est_)
		    #print 'ped_vel={}'.format(self.ped_vel)
	
		## Velocity Filtering: low-pass, multiple counts for convergence check
		    ccount_thr = 4
                    vh_ = np.linalg.norm(self.ped_vel)
		    filter_alpha = 0.15
		    filter_thr = 0.05

	            vh_filtered = filter_alpha*vh_filtered + (1-filter_alpha)*vh_
	            if abs(vh_filtered-vh_filtered_old) < filter_thr*vh_filtered and not vh_fixed and vh_filtered>vh_thr:
		        convergence_count+= 1
		        if (convergence_count>ccount_thr and tbc_ped<7.5) or tbc_ped<5.5:
		            vh_fixed = 1
		            vh_fixed_value = np.copy(vh_filtered)
			    print '-----------------------vh estimate fixed'
			    print'-----------------------------arrival timing of pedestrian = {}'.format(tbc_ped)
			    print'-----------------------------arrival timing of pedestrian = {}'.format(tbc_ped)
			    print'-----------------------------arrival timing of pedestrian = {}'.format(tbc_ped)
			    print'-----------------------------arrival timing of pedestrian = {}'.format(tbc_ped)
			#else:
	                    #print 'self.ped_vel, before filtered value = {}'.format(np.linalg.norm(self.ped_vel))
	                    #print 'filtered_old_value = {}'.format(vh_filtered_old)
	                    #print 'vh_filtered = {}'.format(vh_filtered)
	                    #print 'filter value converged?          {}'.format(abs(vh_filtered-vh_filtered_old) < 0.1*vh_filtered)
	                    #print 'vh=           {}'.format(self.vh)
	                    #print 'vh_fixed=           {}'.format(vh_fixed)

		
		    else :
		        convergence_count = 0

			
		#        print '--------------------------------------vh_fixed( see next data printed)!!!!!!!{}'.format(vh_fixed_value)

	            if vh_fixed :
		        self.vh = np.copy(vh_fixed_value)
	            else:
		        self.vh = np.copy(vh_filtered)
	            vh_filtered_old = np.copy(vh_filtered)
		    
		## cmd_vel pub at {cmd_pub_rate} hz, and traj update at {1/dt} hz
		if  robot_time_del > (self.dt-1/cmd_pub_rate):
			
			
			self.rob_vel = (current_rob_pos-initial_rob_pos)/robot_time_del
			initial_time = np.copy(current_time)
			initial_ped_pos = np.copy(current_ped_pos)
			initial_rob_pos = np.copy(current_rob_pos)
			rate.sleep()
		else:
		        self.velocity_pub.publish(self.vel_msg)
			rate.sleep()
			continue
		

	    ##crossing checks begin
            pR = np.copy(self.rob_pos)
	    vR = np.copy(self.rob_vel)
	    pH = np.copy(self.ped_pos)
	    vH = np.copy(self.ped_vel)
            vH_ = (self.ped_goal - pH)/np.linalg.norm(self.ped_goal - pH)*self.vh
	    vR_ = (self.rob_goal - pR)/np.linalg.norm(self.rob_goal - pR)*self.vr
	    
	    ## collision timing check
	    p_rel = -pR+pH
            A = np.zeros((2,2))
	    A[0][0] = vR_[0]
	    A[1][0] = vR_[1]
	    A[0][1] = -vH_[0]
	    A[1][1] = -vH_[1]

	    if np.linalg.matrix_rank(A)<2 :
		print 'Matrix Singular, vH_ = {}'.format([vH_[0], vH_[1]])
		self.velocity_pub.publish(self.vel_msg)
		continue
 	    tbc = np.linalg.solve(A,p_rel)
	    tbc_rob = tbc[0]
	    tbc_ped = tbc[1]

	    ##arrival timing checks (only for prints)
	    if vh_fixed:
	        vR_2 = (self.rob_goal - initial_rob_pos)/np.linalg.norm(self.rob_goal - initial_rob_pos)*self.vr
		A_ = np.zeros((2,2))
	        A_[0][1] = -vH_[0]
	        A_[1][1] = -vH_[1]
	        A_[0][0] = -vR_2[0]
	        A_[1][0] = -vR_2[1]
		
 	        tbc_init = np.linalg.solve(A_,-initial_rob_pos+pH)
	        #print ('Estimated arrival ped timing = {}'.format(tbc_init[1]))
		if vh_fixed_count<2:
		    print '----------------vR_fixed to {} before interaction starts'.format(np.linalg.norm(vR_fixed))

            time_ = 0.0
	    #print 'vR_ = {}'.format(np.linalg.norm(vR_))
	
	    


	    ## Command computation
	    ## Phases defined based on pedestrian arrival timing estimates (tbc_ped)
	    start_time = 0.0
	    if self.poly_plan==0 :
		##Before interaction starts, update vR_fixed, to catch up pedestrian arrival
                vR_accel = vR_*tbc_rob/(tbc_ped-self.timing_sm/self.vh)

		if not vh_fixed or vh_fixed_count < 1:
		    vR_fixed = np.copy(vR_accel)
		    vh_fixed_count+=1
		    
		self.x_pos_ref  = vR_[0]*self.dt + self.rob_pos[0]
		self.y_pos_ref  = vR_[1]*self.dt + self.rob_pos[1]
		self.x_vel_ref = vR_[0]
		self.y_vel_ref = vR_[1]
		#if vel_count%10==0 :
		    #print '--------------------------timing difference = {}'.format(tbc_ped-tbc_rob-self.timing_sm/self.vh)
		
                if tbc_ped < self.rob_reaction_time :
		    #Interaction begins:
            	    self.local_traj_duration = np.copy(tbc_ped)
		    start_time = time.time()
		    #poly_traj_generation:
		    time_shift = 1.2
		    #PolyTrajGenerationY is the constant-x-velocity implementation for waypoint generation
		    #PolyTrajGeneration is old implementation for waypoint generation in both x-y
	            rx, ry, x_sampled_accel, y_sampled_accel, action, pplan, xtraj, ytraj, xvel, yvel, xtraj_,ytraj_,xvel_,yvel_ = PolyTrajGenerationY(time_shift,self.rob_pos, vR_fixed, self.rob_intent, self.ped_pos, self.ped_vel, self.local_traj_duration, self.dt, time_, self.rob_reaction_time, self.max_accelx, self.max_accely, self.timing_sm, self.safety_sm, self.ped_goal, self.rob_goal, self.vh, self.vr, self.n_states)
	            #action, pplan, xtraj, ytraj, xvel, yvel = PolyTrajGeneration(time_shift,self.rob_pos, self.rob_vel, self.rob_intent, self.ped_pos, self.ped_vel, self.local_traj_duration, self.dt, time_, self.rob_reaction_time, self.max_accelx, self.max_accely, self.timing_sm, self.safety_sm, self.ped_goal, self.rob_goal, self.vh, self.vr, self.n_states)
		    print 'x_traj = {}'.format(xtraj)
		    print 'x_vel = {}'.format(xvel)
		    print 'y_traj = {}'.format(ytraj)
		    print 'y_vel = {}'.format(yvel)
		    for l in range(0,1):
		        self.Xtraj = np.copy(xtraj[l])
                        self.Ytraj = np.copy(ytraj[l])
                        self.Xvel = np.copy(xvel[l])
                        self.Yvel = np.copy(yvel[l])
			terminate_index = len(self.Xvel)-max(sum(abs(self.Xvel)>2.0),sum(abs(self.Yvel)>2.0))
			#terminate_index = 0 # TODO:uncomment for non-poly performance
			#print 'max x vel = {}'.format(self.Xvel[0:terminate_index])
			#print 'max y vel = {}'.format(self.Yvel[0:terminate_index])
			


                    self.poly_plan = pplan
		    #self.poly_plan = 1#TODO: uncomment for non-poly performance
		    
		    if pplan :
			print 'vR_fixed = {}'.format(vR_fixed)
		        plot_count = 0
		        plt.plot(self.Xvel[0:terminate_index],self.Yvel[0:terminate_index],'r')
		        plt.plot(xvel_[0:terminate_index],yvel_[0:terminate_index],'r')
                        plt.pause(0.05)
			print 'xtraj_ = {}'.format(xtraj_[0:terminate_index])
			print 'ytraj_ = {}'.format(ytraj_[0:terminate_index])
                    self.robot_time_del = 0#
                    self.robot_poly_index = 0#
		    ratio = 1.1/max(self.Yvel[0:20])
		    ratio =1.1/0.54
		    vR_baseline = vR_ + vH_/ratio
	    #else:
		#print 'vR_[1] = {}'.format(vR_[1])

	    time_now = time.time()
	    vR_ = np.copy(vR_fixed)
	    ##BASELINE implementation(non-reactive behavior): replace whatever poly commands, after interaction starts
	    if (time_now-start_time<self.local_traj_duration-0) and tbc_ped< self.rob_reaction_time:
		#vR_baseline = np.copy(vR_fixed)# non-reactive baseline, TODO 
		#vR_ = vR_baseline #send baseline commands, TODO
		print 'REACHED INTERACTION PHASE'
	    
			    
	    
	    lookahead = int(0.2/self.dt)
	    lookahead = 0
	    if len(self.Xvel)==0 :
	        terminate_index = 0
	    ##Compute commands: vel_cmd
            if self.robot_poly_index < terminate_index :
		#polynomial traj followining
                self.x_pos_ref = np.copy(self.Xtraj[self.robot_poly_index]+lookahead)
                self.y_pos_ref = np.copy(self.Ytraj[self.robot_poly_index]+lookahead)
                self.x_vel_ref = np.copy(self.Xvel[self.robot_poly_index])
                self.y_vel_ref = np.copy(self.Yvel[self.robot_poly_index])
		#print 'poly tracking x_pos_ref = {}'.format(self.x_pos_ref)
                x_pos_ref_old = np.copy(self.Xtraj[max(0,self.robot_poly_index-1)])
                y_pos_ref_old = np.copy(self.Ytraj[max(0,self.robot_poly_index-1)])
                
		##Tracking Options
		##1. open-loop vel commands
		vel_msg_.linear.x = self.x_vel_ref
       	        vel_msg_.linear.y = self.y_vel_ref
		##2. close-loop position tracking commands
                #vel_msg_.linear.x = 1.0*(self.x_pos_ref-self.rob_pos[0])/self.dt
       	        #vel_msg_.linear.y = 1.0*(self.y_pos_ref-self.rob_pos[1])/self.dt
                ##3. close-loop combined tracking commands
		#vel_msg_.linear.x = 0.2*(self.x_pos_ref-self.rob_pos[0])/self.dt + 0.8*self.x_vel_ref
       	        #vel_msg_.linear.y = 0.2*(self.y_pos_ref-self.rob_pos[1])/self.dt + 0.8*self.y_vel_ref
                #self.vel_msg.linear.x = 0.2*(self.x_pos_ref-y_pos_ref_old)/self.dt + 0.8*self.x_vel_ref
       	        #self.vel_msg.linear.y = 0.2*(self.y_pos_ref-y_pos_ref_old)/self.dt + 0.8*self.y_vel_ref
	        #print 'self.vel_msg.linear.y = {}'.format(self.vel_msg.linear.y) 
	        #print 'self.vel_msg.linear = {}'.format(np.linalg.norm([self.vel_msg.linear.y,self.vel_msg.linear.x])) 
            else :
		##command velocity (non-poly-tracking)
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
	    #print 'cmd_vel = {}{}'.format(vel_msg_.linear.x,vel_msg_.linear.y) 
           	
	    ss_error = 1.2
	    self.vel_msg.linear.x = ss_error*(vel_msg_.linear.x*m.cos(self.rob_yaw) + vel_msg_.linear.y*m.sin(self.rob_yaw))
	    self.vel_msg.linear.y = ss_error*(-vel_msg_.linear.x*m.sin(self.rob_yaw) + vel_msg_.linear.y*m.cos(self.rob_yaw))
	    
            omega = 1.0
	    damping = 1.0
	    self.vel_msg.angular.z = omega*(start_yaw-self.rob_yaw) + 2*damping*omega*(0-self.angular_vel_z)

	    # protections, send out commands with small constant magnitude
	   #self.vel_msg.linear.x = (vel_msg_.linear.x*m.cos(self.rob_yaw) - vel_msg_.linear.y*m.sin(self.rob_yaw))/np.linalg.norm([vel_msg_.linear.x,vel_msg_.linear.y])*0.4
	    #self.vel_msg.linear.y = (vel_msg_.linear.x*m.sin(self.rob_yaw) + vel_msg_.linear.y*m.cos(self.rob_yaw))/np.linalg.norm([vel_msg_.linear.x,vel_msg_.linear.y])*0.4
	    #print 'cmd_vel = {}{}{}'.format(self.vel_msg.linear.x,self.vel_msg.linear.y,self.vel_msg.angular.z) 
	    ##Protections: command 0.0 when robot goes outside coverage 
	    if self.rob_pos[0]<-8 or self.rob_pos[0] > 6 or self.rob_pos[1] <-4 or self.rob_pos[1]>3 or (np.linalg.norm(self.rob_pos-self.rob_goal)<goal_thr):
		self.vel_msg.linear.x = 0.0
		self.vel_msg.linear.y = 0.0


	    self.velocity_pub.publish(self.vel_msg)
            
	    # Forward simulation in simulation mode
	    if self.sim :
                #self.rob_pos[0] = self.x_pos_ref
                #self.rob_pos[1] = self.y_pos_ref
                self.ped_pos[0] += vH_[0]*self.dt
                self.ped_pos[1] += vH_[1]*self.dt
		#self.rob_vel[0] = vR_[0]
		#self.rob_vel[1] = vR_[1]
		self.ped_vel[0] = vH_[0]
		self.ped_vel[1] = vH_[1]

            vel_count = vel_count + 1
	
	    # Plotting starts here (actual code ends here)
	    if plot_count%10>0 :
                self.robot_poly_index += 1
                self.robot_time_del += self.dt
	        plot_count+=1
	        continue
	    plot_count+=1
	    plt.cla()
            plt.axis('equal')
            x_min = -10
            x_max = 8
            y_min = -6
            y_max = 4
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
            plt.plot(self.corner_pose[:, 0], self.corner_pose[:, 1], 'k')
            plt.plot([self.corner_pose[-1, 0], self.corner_pose[0, 0]], [self.corner_pose[-1, 1], self.corner_pose[0, 1]], 'k')
	    plt.plot(self.ped_pos[0],self.ped_pos[1],'ok')
	    plt.plot(self.rob_pos[0],self.rob_pos[1],'om')
	    plt.plot(self.ped_goal[0],self.ped_goal[1],'ok')
	    plt.plot(self.rob_goal[0],self.rob_goal[1],'om')
	    if (len(self.Xtraj)>0) :
		plt.plot(self.Xtraj[0:terminate_index],self.Ytraj[0:terminate_index],'r')
	    else :
		x_local_ref = np.zeros(2)
		y_local_ref = np.zeros(2)
		x_local_ref[0] = self.x_pos_ref
		x_local_ref[1] = self.rob_pos[0]	
		y_local_ref[0] = self.y_pos_ref
		y_local_ref[1] = self.rob_pos[1]	


            self.robot_poly_index += 1
            self.robot_time_del += self.dt
            plt.pause(0.05)

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
    ped_crossing = PedCrossing(sim_mode=False)
    rospy.spin()
