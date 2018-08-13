import numpy as np
import numpy.linalg as linalg
import sys
import math as m

def PolyTrajGenerationY(time_shift,rob_pos, rob_vel, rob_intent, ped_pos, ped_vel,local_traj_duration, dt, time, reaction_time, max_accelx, max_accely, timing_sm, safety_sm, ped_goal, rob_goal, vh, vr, n_states):
    if rob_intent>1 :
        rob_intent = 0

    p1_goal_hat = ped_goal
    p2_goal_hat = rob_goal
    vh_hat = vh
    vr_hat = vr

    pR = rob_pos
    vR = rob_vel
    vR_ = (p2_goal_hat-pR)/linalg.norm(p2_goal_hat-pR)*vr_hat
    pH = ped_pos
    vH = ped_vel
    vH_ = (p1_goal_hat-pH)/linalg.norm(p1_goal_hat-pH)*vh_hat

    p_rel = -pR+pH
    A = np.zeros((2,2))
    A[0][0] = vR_[0]
    A[1][0] = vR_[1]
    A[0][1] = -vH_[0]
    A[1][1] = -vH_[1]

    tbc = np.linalg.solve(A,p_rel)
    tbc_rob = tbc[0]
    tbc_ped = tbc[1]
    arrival_timing_ped = tbc[1]-time_shift

    if arrival_timing_ped > reaction_time :
        poly_plan = 0
    else :
        poly_plan = 1

    tau = 1
    x_sampled_accel_lower_bound = vr_hat*(tbc_rob-arrival_timing_ped)*2/(tau**2)
    x_sampled_accel_lower_bound = max(x_sampled_accel_lower_bound,0)

    delta_accel = 0.05
    rx = (max_accelx-x_sampled_accel_lower_bound)/delta_accel
    rx = int(m.floor(rx))

    x_sampled_accel = x_sampled_accel_lower_bound+np.dot(range(0,rx),delta_accel)
    if len(x_sampled_accel)==0 :
	x_sampled_accel = max_accelx

    ry = max_accely/delta_accel
    ry = int(m.floor(ry))
    y_sampled_accel = np.dot(range(0,ry),delta_accel)

    x_acc_mesh, y_acc_mesh = np.meshgrid(x_sampled_accel,y_sampled_accel)
    
    #xy_sampled_accel = [x_acc_mesh.reshape(1,rx*ry),y_acc_mesh.reshape(1,rx*ry)]
    x_sampled_accel = x_acc_mesh.reshape(1,rx*ry)
    y_sampled_accel = y_acc_mesh.reshape(1,rx*ry)


    # Forward Rollouts --x:vR_ direction
    n_coeff = 5
    x_coeff = np.zeros(n_coeff)
    y_coeff = np.zeros(n_coeff)
    x_coeff[n_coeff-1] = sum(np.multiply(vR,vR_))/linalg.norm(vR_)
    y_coeff[n_coeff-1] = m.sqrt(max(0,linalg.norm(vR)**2-linalg.norm(x_coeff[n_coeff-1])**2))


    recover_t = 4
    rt = int(m.ceil(local_traj_duration/dt))
    t = np.dot(range(0,rt),dt)
    n_accel = len(x_acc_mesh)

    #print 'n_accel = {}'.format(n_accel)
    Xtraj = np.zeros((n_accel,rt))
    Ytraj = np.zeros((n_accel,rt))
    Xvel = np.zeros((n_accel,rt))
    Yvel = np.zeros((n_accel,rt))

    action = [0,0]
    
    for i in range(0,n_accel):
	if len(x_sampled_accel)>0 :
            action[0] = x_sampled_accel[0][-1-i]
            action[1] = y_sampled_accel[0][-1-i]

        x_coeff[3] = action[0]/2
        y_coeff[3] = action[1]/2

        new_tbc_rob = arrival_timing_ped - timing_sm/linalg.norm(vH_)
        ntr = new_tbc_rob
        Ax = np.zeros((3,3))
        Ax[0][0] = ntr**n_coeff
        Ax[0][1] = ntr**(n_coeff-1)
        Ax[1][0] = (n_coeff)*ntr**(n_coeff-1)
        Ax[1][1] = (n_coeff-1)*ntr**(n_coeff-2)
        if n_coeff>3 :
            Ax[0][2] = ntr**(n_coeff-2)
            Ax[1][2] = 3*ntr**2
            Ax[2][0] = 5*4*ntr**3
            Ax[2][1] = 4*3*ntr**2
            Ax[2][2] = 3*2*ntr**1

        #bx[0] = linalg.norm(vR_)*tbc_rob-[ntr**2,ntr]*x_ceoff[3:5]
        #bx[1] = linalg.norm(vR_) - [2*ntr,1]*x_coeff[3:5]
        #bx[2] = -2*x_coeff[3]


        #Ax = np.array([ntr**5, ntr**4, ntr**3], \
        #        [5*ntr**4, 4*ntr**3, 3*ntr**2], \
        #        [5*4*ntr**3, 4*3*ntr**2, 3*2*ntr**1])

        ## for n_coeff == 5:
	bx = np.zeros(3)
	bx[0] = linalg.norm(vR_)*tbc_rob-np.dot(x_coeff[3:5],[ntr**2,ntr])
        bx[1] = linalg.norm(vR_) - np.dot(x_coeff[3:5],[2*ntr,1])
        bx[2] = -2*x_coeff[3]       
        x_coeff[0:3] = linalg.solve(Ax,bx)

        late_tbc_rob = ntr+recover_t
        ltr = late_tbc_rob
        Ay = np.zeros((3,3))
        Ay[0][0] = ntr**5
        Ay[0][1] = ntr**4
        Ay[0][2] = ntr**3
        Ay[1][0] = 5*ntr**4
        Ay[1][1] = 4*ntr**3
        Ay[1][2] = 3*ntr**2
        Ay[2][0] = ltr**5
        Ay[2][1] = ltr**4
        Ay[2][2] = ltr**3

        sm_rob = safety_sm - timing_sm
	by = np.zeros(3)
        by[0] = sm_rob-np.dot(y_coeff[3:5],[ntr**2,ntr])
	by[1] = -np.dot(y_coeff[3:5],[2*ntr,1])
	by[2] = -np.dot(y_coeff[3:5],[ltr**2,ltr])
                 
        y_coeff[0:3] = linalg.solve(Ay,by)

        x_traj = np.zeros(rt)
        y_traj = np.zeros(rt)
        x_vel = np.zeros(rt)
        y_vel = np.zeros(rt)

        for j in range(0,n_coeff) :
            y_traj += np.multiply(np.power(t,j+1),y_coeff[n_coeff-1-j])
            y_vel += np.multiply(np.power(t,j),(j+1)*y_coeff[n_coeff-1-j])

	x_vel = np.power(t,0)*vr_hat
	x_traj = np.power(t,1)*vr_hat

        x_dir = vR_/np.linalg.norm(vR_)
        y_dir = np.dot([[0,1],[-1,0]],x_dir)
        y_dir = np.sign(sum(np.multiply(p1_goal_hat-pR,y_dir)))*y_dir

        x_traj_ref = x_dir[0]*x_traj + y_dir[0]*y_traj
        y_traj_ref = x_dir[1]*x_traj + y_dir[1]*y_traj
        x_vel_ref = x_dir[0]*x_vel + y_dir[0]*y_vel
        y_vel_ref = x_dir[1]*x_vel + y_dir[1]*y_vel

        Xtraj[i] = pR[0]+np.copy(x_traj_ref)
        Ytraj[i] = pR[1]+np.copy(y_traj_ref)
        Xvel[i] = np.copy(x_vel_ref)
        Yvel[i] = np.copy(y_vel_ref)

    return action, poly_plan, Xtraj, Ytraj, Xvel, Yvel

