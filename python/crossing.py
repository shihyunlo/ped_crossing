import sys
sys.path.append('/home/ped_crossing')

import numpy as np
import math
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt

import rospy
import std_msgs.msg as std_msg
import geometry_msgs.msg as geom_msg
from geometry_msgs.msg import PoseStamped, Twist
import visualization_msgs.msg as vis_msg
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

#from utils import linear_interp_traj
#from holonomic_controller import HolonomicController

#argv: python ped_crossing.py arg1 arg2 arg3 arg4 arg5 arg6
#arg1: helmet id for ped, arg2: chair id for ped_goal
#arg3: timing_sm, arg4: safety_sm, arg5: max_accelx (0.4), arg6: max_accely (0.4)
class PedCrossing:
    def __init__(self):
	    # general parameters
	    self.n_states = 10
	    self.dt = 0.1 # traj publisher rate
        self.ped_helmet_id = sys.argv[1]
        self.ped_goal_chair_id = sys.arg[2]

        # ped/robot goal information
        self.ped_goal = np.zeros((2,1))
	    self.rob_goal = np.zeros((2,1))
	    self.vh = 1.0 #defult, to be estimate
	    self.vr = 0.7 #nominal speed
	    self.T = 12.0

	    self.ped_pos = np.zeros((2,1))
        self.ped_vel = np.zeros((2,1))
        self.rob_pos = np.zeros((2,1))
        self.ped_vel = np.zeros((2,1))

        self.rob_intent = 0 # 0: pass first
        self.ped_intent = 1 # 1: yield

        self.rob_reaction_time = 3.0

        # polynomial traj generation
        self.poly_plan = 0
        self.ct = self.T/2
        self.timing_sm = 1.0
        self.safety_sm = 1.8
        # self.timing_sm = sys.argv[3]
        # self.safety_sm = sys.argv[4]

	    self.max_accelx = 0.4
        self.max_accely = 0.4
        # self.max_accelx = sys.argv[5]
        # self.max_accely = sys.argv[6]
	    self.recovery_gain = 4.0

        self.Xtraj = []
        self.Ytraj = []
        self.Xvel = []
        self.Yvel = []
        self.x_pos_ref = 0.0
        self.y_pos_ref = 0.0
        self.x_vel_ref = 0.0
        self.y_vel_ref = 0.0

        #helmets and chairs
        self.obs_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_pre_pose = np.full((self.num_obs, 2), np.inf)
        self.obs_vel = np.full((self.num_obs, 2), np.inf)
        self.obs_pose_update_time = [np.inf, np.inf, np.inf, np.inf, np.inf]
        self.corner_pose = np.full((4,2), np.inf)

        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('ped_pos', vis_msg.Marker, latch=False, queue_size=1)
        self.line_seg_pub = rospy.Publisher('path', vis_msg.Marker, latch=False, queue_size=1)
        self.robot_marker_pub = rospy.Publisher('robot', vis_msg.Marker, latch=False, queue_size=1)
        self.goal_marker_pub = rospy.Publisher('goal', vis_msg.Marker, latch=False, queue_size=1)

        #if self.sim:
        #    self.get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        #    self.request = GetModelStateRequest()
        #    self.request.model_name = 'ballbot'

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


        self.helmet00_sub = rospy.Subscriber('/vrpn_client_node/HelmetXL/pose', PoseStamped, self.helmet00_callback, queue_size=1)
        self.helmet01_sub = rospy.Subscriber('/vrpn_client_node/HelmetM/pose', PoseStamped, self.helmet01_callback, queue_size=1)
        self.helmet02_sub = rospy.Subscriber('/vrpn_client_node/HelmetS/pose', PoseStamped, self.helmet02_callback, queue_size=1)

        self.corner00_sub = rospy.Subscriber('/vrpn_client_node/corner1/pose', PoseStamped, self.corner00_callback, queue_size=1)
        self.corner01_sub = rospy.Subscriber('/vrpn_client_node/corner2/pose', PoseStamped, self.corner01_callback, queue_size=1)
        self.corner02_sub = rospy.Subscriber('/vrpn_client_node/corner3/pose', PoseStamped, self.corner02_callback, queue_size=1)
        self.corner03_sub = rospy.Subscriber('/vrpn_client_node/corner4/pose', PoseStamped, self.corner03_callback, queue_size=1)

        self.robot_pose_sub = rospy.Subscriber('/vrpn_client_node/RockHopper6/pose', PoseStamped, self.robot_pose_callback, queue_size=1)

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

    def robot_pose_callback(self, pose_msg):
        if pose_msg.pose.position.z > 0.05 :
            self.start_x = pose_msg.pose.position.x
            self.start_y = pose_msg.pose.position.y
            # TODO: get the real yaw
            self.start_yaw = 0.0

            self.vx_ = (self.start_x - self.start_pre_x) / self.dt_mocap
            self.vy_ = (self.start_y - self.start_pre_y) / self.dt_mocap

            self.start_pre_x = self.start_x
            self.start_pre_y = self.start_y

	    self.rob_pos[0] = pose_msg.pose.position.x
	    self.rob_pos[1] = pose.msg.pose.position.y

    def goal_pose_callback(self, goal_pose_msg):
        self.goal_x = goal_pose_msg.pose.position.x
        self.goal_y = goal_pose_msg.pose.position.y
        q = goal_pose_msg.pose.orientation
        #self.goal_yaw = math.atan2(dd2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z)

    def run(self):
        rate = rospy.Rate(1.0/self.dt)
        vel_count = 0
        vel_n = 5
        vel_est = np.zeros((2,vel_n))
        pos_rob_old = np.zeros((2,1))
        pos_ped_old = np.zeros((2,1))
        if self.sim:
	        ## set-up callback values for planning
            self.ped_goal[0] = -0.2
            self.ped_goal[1] = 8.71
	        self.rob_goal[0] = 4.6
            self.rob_goal[1] = 2.0

		    self.ped_pos[0] = -0.2
            self.ped_pos[1] = -4.49
	        self.rob_pos[0] = -5.0
            self.rob_pos[1] = 2.0

            self.ped_vel[0] = 0.0
            self.ped_vel[1] = 1.1
            self.rob_vel[0] = 0.8
            self.rob_vel[1] = 0.0

        while not rospy.is_shutdown():
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


                if vel_count== 0 :
                    pos_rob_old = self.rob_pos
                    pos_ped_old = self.ped_pos

                if vel_count>=(vel_n-1) :
                    self.rob_vel = (self.rob_pos-pos_rob_old)/dt_mocap/vel_n
                    self.ped_vel = (self.ped_pos-pos_ped_old)/dt_mocap/vel_n
                    vel_count = 0

                vel_count = vel_count + 1

            pR = self.rob_pos
	        vR = self.rob_vel
	        pH = self.ped_pos
	        vH = self.ped_vel
	        self.vh = np.linalg.norm(vH)

	        vH_ = (self.ped_goal - pH)/np.linalg.norm(self.ped_goal - pH)*self.vh
	        vR_ = (self.rob_goal - pR)/np.linalg.norm(self.rob_goal - pR)*self.vr
	        # collision timing check
	        p_rel = -pR+pH
	        A = np.zeros((2,2))
	        A[0][0] = vR_[0]
	        A[1][0] = vR_[1]
	        A[0][1] = -vH_[0]
	        A[1][1] = -vH_[1]

 	        tbc = np.linalg.solve(A,p_rel)
	        tbc_rob = tbc[0]
	        tbc_ped = tbc[1]

            local_traj_duration = tbc_ped
            time = 0.0

	        if !self.poly_plan :
                vR_ = vR_*tbc_rob/(tbc_ped-self.timing_sm/self.vh)

                if tbc_ped < self.rob_reation_time :
	                action, pplan, xtraj, ytraj, xvel, yvel = self.poly_traj_generation(self.rob_pos, self.rob_vel, self.rob_intent, self.ped_pos, self.ped_vel, local_traj_duration, self.dt, time, self.rob_reaction_time, self.max_accelx, self.max_accely, self.timing_sm, self.safety_sm)
                    self.Xtraj = xtraj
                    self.Ytraj = ytraj
                    self.Xvel = xvel
                    self.Yvel = yvel

                    self.poly_plan = pplan
                    self.robot_time_del = 0
                    self.robot_poly_index = 0

            if self.robot_poly_index < local_traj_duration/self.dt :
                self.x_pos_ref = self.Xtraj[0,self.robot_poly_index]
                self.y_pos_ref = self.Ytraj[0,self.robot_poly_index]
                self.x_vel_ref = self.Xvel[0,self.robot_poly_index]
                self.y_vel_ref = self.Yvel[0,self.robot_poly_index]
            else :
                a_rob = self.recovery_gain*(vR_-vR)
                self.x_vel_ref = vR_[0]
                self.y_vel_ref = vR_[1]
                self.x_pos_ref = pR[0] + vR_[0]*self.dt
                self.y_pos_ref = pR[1] + vR_[1]*self.dt

            if self.sim :
                self.rob_pos[0] = self.x_pos_ref
                self.rob_pos[1] = self.y_pos_ref
                self.ped_pos[0] += vH_[0]*self.dt
                self.ped_pos[1] += vH_[1]*self.dt

            plt.cla()
            plt.axis('equal')
            x_min = -4
            x_max = 4
            y_min = -3
            y_max = 3
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
            plt.plot(self.start_x, self.start_y, 'or')
            plt.plot(self.goal_x, self.goal_y, 'ob')
            plt.arrow(self.start_x, self.start_y, 0.3 * math.cos(self.start_yaw), 0.3 * math.sin(self.start_yaw), head_width=0.02, head_length=0.08, fc='r', ec='k')
            plt.arrow(self.start_x, self.start_y, 0.3 * -math.sin(self.start_yaw), 0.3 * math.cos(self.start_yaw), head_width=0.02, head_length=0.08, fc='g', ec='k')
            plt.arrow(self.start_x, self.start_y, obs_lin_x * math.cos(self.start_yaw), obs_lin_x * math.sin(self.start_yaw), head_width=0.05, head_length=0.08, fc='r', ec='k')
            plt.arrow(self.start_x, self.start_y, obs_lin_y * -math.sin(self.start_yaw), obs_lin_y * math.cos(self.start_yaw), head_width=0.05, head_length=0.08, fc='g', ec='k')
            # plot goal
            plt.arrow(self.goal_x, self.goal_y, 0.3 * math.cos(self.goal_yaw), 0.3 * math.sin(self.goal_yaw), head_width=0.02, head_length=0.08, fc='r', ec='k')
            plt.arrow(self.goal_x, self.goal_y, 0.3 * -math.sin(self.goal_yaw), 0.3 * math.cos(self.goal_yaw), head_width=0.02, head_length=0.08, fc='g', ec='k')
            plt.plot(waypoints[:, 0], waypoints[:, 1], 'or')
            plt.plot(waypoints[ind, 0], waypoints[ind, 1], 'og')

            plt.draw()

            self.robot_poly_index += 1
            self.robot_time_del += self.dt
            plt.pause(0.00000001)

            rate.sleep()
"
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
"
if __name__ == '__main__':
    rospy.init_node('ped_crossing_node')
    #arg1: helmet id for ped, arg2: chair id for ped_goal
    #arg3: timing_sm, arg4: safety_sm, arg5: max_accelx (0.4), arg6: max_accely (0.4)
    ped_crossing = PedCrossing()
    rospy.spin()
