import numpy as np
import numpy.linalg as linalg
import sys
import math as m

def poly_traj_generation(self, rob_pos, rob_vel, rob_intent, ped_pos, ped_vel,local_traj_duration, dt, time, reaction_time, max_accelx, max_accely, timing_sm, safety_sm, ped_goal, rob_goal, vh, vr, n_states):
    if len(robot_intent)==0 :
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
    vR_ = (p1_goal_hat-pH)/linalg.norm(p1_goal_hat-pH)*vh_hat

    p_rel = -pR+pH
    A = np.zeros((2,2))
    A[0][0] = vR_[0]
    A[1][0] = vR_[1]
    A[0][1] = -vH_[0]
    A[1][1] = -vH_[1]

    tbc = np.linalg.solve(A,p_rel)
    tbc_rob = tbc[0]
    tbc_ped = tbc[1]
    arrival_timing_ped = tbc[1]

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

    x_sampled_accel = x_sampled_accel_lower_bound+range(0,rx)*delta_accel
    ry = max_accely/delta_accel
    ry = int(m.floor(ry))
    y_sampled_accel = range(0,ry)*delta_accel

    x_acc_mesh, y_acc_mesh = meshgrid(x_sampled_accel,y_sampled_accel)
    xy_sampled_accel = [x_acc_mesh.reshape(1,rx*ry),y_acc_mesh.reshape(1,rx*ry)]


    # Forward Rollouts --x:vR_ direction
    n_coeff = 5
    x_coeff = np.zeros(n_coeff)
    y_coeff = np.zeros(n_coeff)
    x_coeff[n_coeff-1] = sum(np.multiply(vR,vR_))/linalg.norm(vR_)
    y_coeff[n_coeff-1] = m.sqrt(linalg.norm(vR)**2-linalg.norm(x_coeff[n_coeff-1])**2)


    recover_t = 4
    rt = int(m.ceil(local_traj_duration/dt))
    t = range(0,rt)*dt
    n_accel = len(x_acc_mesh)
    Xtraj = np.array((n_accel),rt)
    Ytraj = np.array((n_accel),rt)
    Xvel = np.array((n_accel),rt)
    Yvel = np.array((n_accel),rt)

    for i in range(0,n_accel):
        action[0] = xy_sampled_accel[-1-i][0]
        action[1] = xy_sampled_accel[-1-i][1]

        x_coeff[3] = action[0]/2
        y_coeff[3] = action[1]/2

        new_tbc_rob = arrival_time_ped - timing_sm/linalg.norm(vH_)
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
        bx = np.array([linalg.norm(vR_)*tbc_rob-sum(np.multiply([ntr**2,ntr],x_ceoff[3:5])),\
                linalg.norm(vR_) - sum(np.multiply([2*ntr,1]*x_coeff[3:5],-2*x_coeff[3]])))
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
        by = np.array([sm_rob-sum(np.multiply([ntr**2,ntr],y_coeff[3:5])),\
                -sum(np.multiply([2*ntr,1],y_coeff[3:5])), -sum(np.multiply([ltr**2,ltr],y_coeff[3:5]])))
        y_coeff[0:3] = linalg.solve(Ay,by)

        x_traj = zeros(rt)
        y_traj = zeros(rt)
        x_vel = zeros(rt)
        y_vel = zeros(rt)

        for j in range(0,n_coeff) :
            x_traj += np.multiply(np.power(t,j+1),x_coeff[n_coeff-1-j])
            y_traj += np.multiply(np.power(t,j+1),y_coeff[n_coeff-1-j])
            x_vel += np.multiply(np.power(t,j),(j+1)*x_coeff[n_coeff-1-j])
            y_vel += np.multiply(np.power(t,j),(j+1)*y_coeff[n_coeff-1-j])

        x_dir = vR_/np.linalg.norm(vR_)
        y_dir = np.dot([[0,1],[-1,0]],x_dir)
        y_dir = np.sign(sum(np.multiply(p1_goal-pR,y_dir)))*y_dir

        x_traj_ref = x_dir[0]*x_traj + y_dir[0]*y_traj
        y_traj_ref = x_dir[1]*x_traj + y_dir[1]*y_traj
        x_vel_ref = x_dir[0]*x_vel + y_dir[0]*y_vel
        y_vel_ref = x_dir[1]*x_vel + y_dir[1]*y_vel

        Xtraj[i,:] = x_traj_ref
        Ytraj[i,:] = y_traj_ref
        Xvel[i,:] = x_vel_ref
        Yvel[i,:] = y_vel_ref

    return action poly_plan Xtraj Ytraj Xvel Yvel

