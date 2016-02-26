## Usage ###################

Make sure posAnd_dir.csv is located within the same directory as the images. The posAnd_dir.csv file should have the following format:
```
filename,latitude,longitude,altitude,yaw,pitch,roll
dji_0644.jpg,­123.114661,38.426805,90.689292,9.367337,1.260910,0.385252
...
```


##  From labview ###
Output file from labview has this information:

raw_data.log
time, signals out (X_tp, Y_tp, Z_pg, 0, 0,psi_c, u_dvl, v_dvl, z_dvl, p, q, r), MT DATA, dvl alts (1-4), Ship pos (X,Y, Psi [deg])

eta_data.log
time, eta_mes, eta_est, eta_des (6-DOF), alt_mes, alt_est, alt_des, gradF_x, gradF_y, gradF_z 

tau_data.log
time, tau [N/Nm] (X,Y,Z,K,M,N), rpm(side,vert,starboard,port) 

nu_data.log
time, nu_mes, nu_est, nu_des (6-DOF) 

## Estimated dvl altitude ##
alt_est

## Estimated latitude,longitude ##
- Maybe its possible to use some other relative factor


## Estimated yaw,pitch,roll ##





