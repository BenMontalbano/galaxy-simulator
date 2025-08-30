import numpy as np
from numba import cuda, float32
import math
import random
import matplotlib.pyplot as plt
from vispy import scene, app
from vispy.scene import visuals
from PyQt5 import QtCore
from scipy.special import erf
import time
import imageio
import imageio.v3 as iio
import os
from PIL import Image
from datetime import datetime



# Author: Ben Montalbano


# Purpose: Simulate a galaxy, and demonstrate the different galaxy rotation curves
# with and without a dark matter halo. Thes script produces an animation of the galaxy,
# and a galaxy rotation curve plot. This script is designed to make use of a desktop 
# GPU to speed up higher pop simulations. 


# Units: kpc, solar mass, year





# Physical Constants
G = np.float32(4.5E-24) # kpc^3/(M_sun*yr^2)
c = np.float32(1/3260) # kpc/yr


# Simulation Parameters
duration = 100000000 # years
dt = np.float32(500)
num_steps = int(duration / dt)
epsilon = np.float32(0.1)  # Softening parameter to avoid division by zero
frame_speed = 100 # number of steps per frame

# Galaxy Structure
star_density = np.float32(1E6) # stars in a starblock
starblock_mass = np.float32(1 * star_density) # solar masses * density
dm_block_mass = np.float32(4 * star_density) 

sn_center = np.array([-20.0, 0.0, 0.0], dtype=np.float32)
sn_velocity = np.array([1.5E-7, 1.0E-7, 0.0], dtype=np.float32)
sn_tot_blocks = 12000
sn_disk_frac = 0.8
sn_bulge_frac = 1 - sn_disk_frac
sn_dm_frac = 0.5
sn_disk_blocks = int(sn_disk_frac * sn_tot_blocks)
sn_bulge_blocks = int(sn_bulge_frac * sn_tot_blocks)
sn_dm_blocks = int(sn_dm_frac * sn_tot_blocks)
sn_core_mass = np.float32(1E8)
sn_disk_mass = np.float32(sn_disk_blocks * starblock_mass)
sn_bulge_mass = np.float32(sn_bulge_blocks * starblock_mass)
sn_dm_mass = np.float32(sn_dm_blocks * dm_block_mass)
sn_tot_mass = np.float32(sn_disk_mass + sn_bulge_mass + sn_dm_mass)
sn_max_rad = np.float32(100)
sn_scale_r = np.float32(4)
sn_scale_h = np.float32(0.3)
sn_bulge_scale_r = np.float32(0.1)
sn_dm_scale_r = np.float32(20)
sn_num_arms = np.float32(2)
sn_pitch_angle = np.float32(np.pi / 6)
sn_arm_width = np.float32(np.pi / 6)



alm_center = np.array([20.0, 0.0, 0.0], dtype=np.float32)
alm_velocity = np.array([-1.5E-7, -1.0E-7, 0.0], dtype=np.float32)
alm_tot_blocks = 8000
alm_disk_frac = 0.8
alm_bulge_frac = 1 - alm_disk_frac
alm_dm_frac = 1
alm_disk_blocks = int(sn_disk_frac * sn_tot_blocks)
alm_bulge_blocks = int(sn_bulge_frac * sn_tot_blocks)
alm_dm_blocks = int(sn_dm_frac * sn_tot_blocks)
alm_core_mass = np.float32(1E9)
alm_disk_mass = np.float32(sn_disk_blocks * starblock_mass)
alm_bulge_mass = np.float32(sn_bulge_blocks * starblock_mass)
alm_dm_mass = np.float32(sn_dm_blocks * dm_block_mass)
alm_tot_mass = np.float32(sn_disk_mass + sn_bulge_mass + sn_dm_mass)
alm_max_rad = np.float32(50)
alm_scale_r = np.float32(2)
alm_scale_h = np.float32(0.2)
alm_bulge_scale_r = np.float32(0.05)
alm_dm_scale_r = np.float32(10)
alm_num_arms = np.float32(2)
alm_pitch_angle = np.float32(np.pi / 8)
alm_arm_width = np.float32(np.pi / 6)






class Galaxy:
    def __init__(self, center, vel, tot_blocks, disk_frac, bulge_frac, dm_frac,\
        core_mass, max_rad, scale_r, scale_h, bulge_scale_r, dm_scale_r,\
        num_arms, pitch_angle, arm_width):
        
        self.center = center
        self.vel = vel
        self.tot_blocks = tot_blocks
        self.disk_blocks = tot_blocks * disk_frac
        self.bulge_blocks = tot_blocks * bulge_frac
        self.dm_blocks = tot_blocks * dm_frac
        self.disk_mass = self.disk_blocks * starblock_mass
        self.bulge_mass = self.bulge_blocks * starblock_mass
        self.dm_mass = self.dm_blocks * dm_block_mass
        self.core_mass = core_mass
        self.max_rad = max_rad
        self.scale_r = scale_r
        self.scale_h = scale_h
        self.bulge_scale_r = bulge_scale_r
        self.dm_scale_r = dm_scale_r
        self.num_arms = num_arms
        self.pitch_angle = pitch_angle
        self.arm_width = arm_width

        self.positions = []
        self.velocities = []
        self.masses = []

    


    # Uses Hernquist Profile to calculate spherical mass distributions
    def hernquist_enclosed_mass(self, r, total_mass, scale_radius):
        return total_mass * (r ** 2) / (r + scale_radius) ** 2

    # Calculate disk mass distribution
    def disk_enclosed_mass(self, r, disk_mass):
        x = r / self.scale_r
        return disk_mass * (1 - np.exp(-x) * (1 + x + 0.5 * x ** 2))

    def enclosed_mass(self, r, disk_mass, bulge_mass, dm_mass):
        dm_mass_enclosed = self.hernquist_enclosed_mass(r, dm_mass, self.dm_scale_r)
        bulge_mass_enclosed = self.hernquist_enclosed_mass(r, bulge_mass, self.bulge_scale_r)
        disk_mass_enclosed = self.disk_enclosed_mass(r, disk_mass)
        return self.core_mass + bulge_mass_enclosed + disk_mass_enclosed + dm_mass_enclosed

    def generate_galaxy(self):
        
        self.positions.append(self.center)
        self.velocities.append(self.vel)
        self.masses.append(self.core_mass)

  
        # Create Bulge starblocks
        for bulge_block in range(int(self.bulge_blocks)):
            r_draw = random.uniform(0, 1)
            r_proj = self.bulge_scale_r * r_draw / (1 - r_draw)
            if r_proj > 2:
                continue
            
            theta, phi = random.uniform(0, np.pi), random.uniform(0, 2 * np.pi)
            x = r_proj * np.sin(theta) * np.cos(phi)
            y = r_proj * np.sin(theta) * np.sin(phi)
            z = r_proj * np.cos(theta)

            r = np.sqrt(r_proj**2 + z**2)

            Mass_enc = self.hernquist_enclosed_mass(r, self.bulge_mass, self.bulge_scale_r)
            v_max = np.sqrt(G * Mass_enc / (r_proj + epsilon/4))

            while True:
                v = random.uniform(0, v_max)
                f_Eddington = v**2 * (1 - (v / v_max)**2)**3.5
                f_peak = 0.1 * v_max**2
                if random.uniform(0, f_peak) < f_Eddington:
                    break

            
            v = min(v, 0.95 * v_max)
            theta_v, phi_v = random.uniform(0, np.pi), random.uniform(0, 2 * np.pi)
            vx = v * np.sin(theta_v) * np.cos(phi_v)
            vy = v * np.sin(theta_v) * np.sin(phi_v)
            vz = v * np.cos(theta_v)

            pos = np.array([x, y, z], dtype=np.float32)
            vel = np.array([vx, vy, vz], dtype=np.float32)

            self.positions.append(pos + self.center)
            self.velocities.append(vel)
            self.masses.append(starblock_mass)
      
    

        # Create Disk starblocks
        for disk_block in range(int(self.disk_blocks)):
            draw = random.uniform(0, 1)
            h_draw = np.random.uniform(0, 1)
            r_proj = min(-self.scale_r * np.log(1 - draw), self.max_rad)

            # Create spiral arms
            while True:
                phi = random.uniform(0, 2*np.pi)

                spiral_phase = (np.log(r_proj / self.scale_r) / np.tan(self.pitch_angle))
                arm_number = random.randint(0, self.num_arms - 1) 
                desired_phi = np.mod(spiral_phase * self.num_arms + arm_number * 2*np.pi/self.num_arms, 2*np.pi)

                delta_phi = np.mod(phi - desired_phi, 2*np.pi)
                if delta_phi > np.pi:
                    delta_phi = 2*np.pi - delta_phi

                prob = np.exp(-(delta_phi / self.arm_width)**2)
                if random.random() < prob:
                    break

            x = r_proj * np.cos(phi)
            y = r_proj * np.sin(phi)
            z = self.scale_h * np.arcsinh(np.tan(np.pi * (h_draw - 0.5)))

            r = np.sqrt(r_proj**2 + z**2)
            theta = np.arcsin(z / r)

            v_hat = np.array([np.sin(phi) * np.cos(theta), -np.cos(phi) * np.cos(theta),\
                        -np.sin(theta)])

            Mass_enc = self.enclosed_mass(r, self.disk_mass, self.bulge_mass, self.dm_mass)
            v_mag = np.sqrt(G * Mass_enc / (r + epsilon))
            v_boost = 1.01
            base_velocity = v_hat * v_mag * v_boost

            # Add in random kicks to break ringing
            velocity_dispersion = 0.01
            random_kick = velocity_dispersion * v_mag * np.random.randn(3)

            pos = np.array([x, y, z], dtype=np.float32)
            vel = base_velocity + random_kick

            self.positions.append(pos + self.center)
            self.velocities.append(vel + self.vel)
            self.masses.append(starblock_mass)
        
    
   
        # Create dark matter blocks
        for wimp in range(int(self.dm_blocks)):
            dm_Draw = np.random.uniform(0,1)
            r_proj = self.dm_scale_r* (np.sqrt(dm_Draw) / (1 - np.sqrt(dm_Draw)))
            theta, phi = random.uniform(0, np.pi), random.uniform(0, 2 * np.pi)

            x = r_proj * np.sin(phi)*np.sin(theta)
            y = r_proj * np.cos(phi)*np.sin(theta)
            z = r_proj * np.cos(theta)
            r = np.sqrt(r_proj**2 + z**2)

            v_hat = np.array([-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta),\
                        -np.sin(theta)])
            
            Mass_enc = self.enclosed_mass(r, self.disk_mass, self.bulge_mass, self.dm_mass)
            v_mag = np.sqrt(G * Mass_enc / (r + epsilon))
            v_boost = 1.01
            base_velocity = v_hat * v_mag * v_boost

            velocity_dispersion = 0.01
            random_kick = velocity_dispersion * v_mag * np.random.randn(3)

            pos = np.array([x, y, z], dtype=np.float32)
            vel = base_velocity + random_kick

            self.positions.append(pos + self.center)
            self.velocities.append(vel + self.vel)
            self.masses.append(dm_block_mass)

def rotate_vectors(vectors, angle_rad, axis='z'):
        """Rotate a batch of 3D vectors by a given angle around specified axis."""
        if axis == 'z':
            R = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad),  np.cos(angle_rad), 0],
                [0,                 0,                1]
            ], dtype=np.float32)
        elif axis == 'y':
            R = np.array([
                [ np.cos(angle_rad), 0, np.sin(angle_rad)],
                [ 0,                1, 0               ],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ], dtype=np.float32)
        elif axis == 'x':
            R = np.array([
                [1, 0,                 0                ],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad),  np.cos(angle_rad)]
            ], dtype=np.float32)
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

        return vectors @ R.T

# Generate both galaxies

snickers = Galaxy(sn_center, sn_velocity, sn_tot_blocks, sn_disk_frac,\
                sn_bulge_frac, sn_dm_frac, sn_core_mass, sn_max_rad, sn_scale_r,\
                sn_scale_h, sn_bulge_scale_r, sn_dm_scale_r, sn_num_arms, sn_pitch_angle,\
                sn_arm_width)

almondjoy = Galaxy(alm_center, alm_velocity, alm_tot_blocks, alm_disk_frac,\
                alm_bulge_frac, alm_dm_frac, alm_core_mass, alm_max_rad, alm_scale_r,\
                alm_scale_h, alm_bulge_scale_r, alm_dm_scale_r, alm_num_arms, alm_pitch_angle,\
                alm_arm_width)

snickers.generate_galaxy()
almondjoy.generate_galaxy()

# Rotate the initial conditions before merging
rotation_angle_snickers = np.radians(35)  # 30 degrees rotation for Snickers
rotation_angle_almondjoy = np.radians(-35)  # -45 degrees for Almondjoy (if you want)

# Snickers galaxy
snickers.positions = [
    rotate_vectors(np.expand_dims(pos - snickers.center, 0), rotation_angle_snickers, axis='y')[0] + snickers.center
    for pos in snickers.positions
]
snickers.velocities = [
    rotate_vectors(np.expand_dims(vel, 0), rotation_angle_snickers, axis='y')[0]
    for vel in snickers.velocities
]

# Almondjoy galaxy
almondjoy.positions = [
    rotate_vectors(np.expand_dims(pos - almondjoy.center, 0), rotation_angle_almondjoy, axis='y')[0] + almondjoy.center
    for pos in almondjoy.positions
]
almondjoy.velocities = [
    rotate_vectors(np.expand_dims(vel, 0), rotation_angle_almondjoy, axis='y')[0]
    for vel in almondjoy.velocities
]
sn_core_pos = [snickers.positions[0]]
sn_star_pos = snickers.positions[1 : 1 + int(snickers.disk_blocks + snickers.bulge_blocks)]
sn_dm_pos = snickers.positions[1 + int(snickers.disk_blocks + snickers.bulge_blocks) : ]

alm_core_pos = [almondjoy.positions[0]]
alm_star_pos = almondjoy.positions[1 : 1 + int(almondjoy.disk_blocks + almondjoy.bulge_blocks)]
alm_dm_pos = almondjoy.positions[1 + int(almondjoy.disk_blocks + almondjoy.bulge_blocks) : ]

# 2. Merge cores first, then stars, then DM
positions = np.array(sn_core_pos + alm_core_pos + sn_star_pos + alm_star_pos + sn_dm_pos + alm_dm_pos, dtype=np.float32)
velocities = np.array(snickers.velocities[0:1] + almondjoy.velocities[0:1] +
                      snickers.velocities[1 : 1 + int(snickers.disk_blocks + snickers.bulge_blocks)] +
                      almondjoy.velocities[1 : 1 + int(almondjoy.disk_blocks + almondjoy.bulge_blocks)] +
                      snickers.velocities[1 + int(snickers.disk_blocks + snickers.bulge_blocks) : ] +
                      almondjoy.velocities[1 + int(almondjoy.disk_blocks + almondjoy.bulge_blocks) : ], dtype=np.float32)
masses = np.array(snickers.masses[0:1] + almondjoy.masses[0:1] +
                  snickers.masses[1 : 1 + int(snickers.disk_blocks + snickers.bulge_blocks)] +
                  almondjoy.masses[1 : 1 + int(almondjoy.disk_blocks + almondjoy.bulge_blocks)] +
                  snickers.masses[1 + int(snickers.disk_blocks + snickers.bulge_blocks) : ] +
                  almondjoy.masses[1 + int(almondjoy.disk_blocks + almondjoy.bulge_blocks) : ], dtype=np.float32)

center_of_mass = np.average(positions, axis=0, weights=masses)


# Force of gravity applied to all non dm bodies
@cuda.jit
def gravity_step(positions, masses, accelerations):
    i = cuda.grid(1)
    N = positions.shape[0]

    if i >= N:
        return

    tx = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    # Shared memory
    shared_pos = cuda.shared.array((128, 3), dtype=float32)
    shared_mass = cuda.shared.array(128, dtype=float32)

    xi, yi, zi = positions[i][0], positions[i][1], positions[i][2]
    ax = ay = az = 0.0

    for tile_start in range(0, N, block_size):
        j = tile_start + tx

        if j < N:
            shared_pos[tx][0] = positions[j][0]
            shared_pos[tx][1] = positions[j][1]
            shared_pos[tx][2] = positions[j][2]
            shared_mass[tx] = masses[j]
        else:
            # Padding out-of-range values
            shared_pos[tx][0] = 0.0
            shared_pos[tx][1] = 0.0
            shared_pos[tx][2] = 0.0
            shared_mass[tx] = 0.0

        cuda.syncthreads()

        for k in range(block_size):
            jx = shared_pos[k][0]
            jy = shared_pos[k][1]
            jz = shared_pos[k][2]
            mj = shared_mass[k]

            # Convert tile index to global particle index
            global_j = tile_start + k

            if global_j != i:
                dx = jx - xi
                dy = jy - yi
                dz = jz - zi
                r2 = dx * dx + dy * dy + dz * dz + epsilon
                r3 = r2 * math.sqrt(r2)
                force = G * mj / r3
                ax += force * dx
                ay += force * dy
                az += force * dz

        cuda.syncthreads()

    accelerations[i][0] = ax
    accelerations[i][1] = ay
    accelerations[i][2] = az


# Position and velocity updates leapfrog each other
def leapfrog_step(positions, velocities, masses):
    
    N = positions.shape[0]
    threads_per_block = 128
    blocks_per_grid = math.ceil(N / threads_per_block)
    d_positions = cuda.to_device(positions)
    d_masses = cuda.to_device(masses)
    accelerations = cuda.device_array((N, 3), dtype=np.float32)

    gravity_step[blocks_per_grid, threads_per_block](d_positions, d_masses, accelerations)

    accelerations = accelerations.copy_to_host()
    velocities_half = velocities + 0.5 * accelerations * dt

    # Cap intermediate velocity
    for i in range(N):
        v = velocities_half[i]
        speed = np.linalg.norm(v)
        if speed > c:
            velocities_half[i] = v * (c / speed)

    new_positions = positions.copy()
    new_positions += velocities_half * dt

    d_new_positions = cuda.to_device(new_positions)
    d_new_accelerations = cuda.device_array((N, 3), dtype=np.float32)
    gravity_step[blocks_per_grid, threads_per_block](d_new_positions, d_masses, d_new_accelerations)
    new_accelerations = d_new_accelerations.copy_to_host()  
    new_velocities = velocities_half + 0.5 * new_accelerations * dt

    # Cap final velocity
    for i in range(N):
        v = new_velocities[i]
        speed = np.linalg.norm(v)
        if speed > c:
            new_velocities[i] = v * (c / speed)


    return new_positions, new_velocities



    return total_mass * (r**2) / (r + scale_radius)**2

center_of_mass = np.array([0,0,0])
N = positions.shape[0]

# Create block path storage
X_paths = np.zeros((N, num_steps), dtype=np.float32)
Y_paths = np.zeros((N, num_steps), dtype=np.float32)
Z_paths = np.zeros((N, num_steps), dtype=np.float32)

#Vel_Rad_Array = np.zeros((N, 2, num_steps))


# Start the timer
start_time = time.time()  


# Simulate galaxy over dt
for t in range(num_steps):
    X_paths[:, t] = positions[:, 0]
    Y_paths[:, t] = positions[:, 1]
    Z_paths[:, t] = positions[:, 2]
    
    positions, velocities = leapfrog_step(positions, velocities, masses)
    
    # Print upon completion of 1000 steps
    if t % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Step {t}/{num_steps} completed in {elapsed:.2f} seconds")


# End the timer
end_time = time.time()  


# Define indices properly
core_index_sn = 0
core_index_alm = 1

sn_num_stars = int(snickers.disk_blocks + snickers.bulge_blocks)
alm_num_stars = int(almondjoy.disk_blocks + almondjoy.bulge_blocks)
sn_num_dm = int(snickers.dm_blocks)
alm_num_dm = int(almondjoy.dm_blocks)

star_start_sn = 1
star_end_sn = star_start_sn + sn_num_stars

star_start_alm = star_end_sn
star_end_alm = star_start_alm + alm_num_stars

dm_start = star_end_alm
dm_end = dm_start + sn_num_dm + alm_num_dm

# Setup canvas and view
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=(0.01, 0.01, 0.03, 1), size=(2560, 1440))
TARGET_SIZE = (2560, 1440)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(elevation=60, azimuth=0, distance=6000, fov=1)
view.camera.center = tuple(center_of_mass)

# Setup particles initial positions
initial_star_pos = np.vstack((
    np.stack((X_paths[star_start_sn:star_end_sn, 0],
              Y_paths[star_start_sn:star_end_sn, 0],
              Z_paths[star_start_sn:star_end_sn, 0]), axis=-1),
    np.stack((X_paths[star_start_alm:star_end_alm, 0],
              Y_paths[star_start_alm:star_end_alm, 0],
              Z_paths[star_start_alm:star_end_alm, 0]), axis=-1)
))

initial_dm_pos = np.stack((X_paths[dm_start:dm_end, 0],
                           Y_paths[dm_start:dm_end, 0],
                           Z_paths[dm_start:dm_end, 0]), axis=-1)

# Create star colors
num_stars = initial_star_pos.shape[0]
color_randoms = np.random.uniform(0, 1, num_stars)
star_colors = np.zeros((num_stars, 4))
for i in range(num_stars):
    t = color_randoms[i]
    r = 1 - 0.2 * t
    g = 0.7 + 0.2 * t
    b = 0.9 + 0.1 * t
    star_colors[i] = (r, g, b, 1.0)

# Starblock sizes based on distance
radii = np.linalg.norm(initial_star_pos, axis=1)
max_r = np.max(radii)
sizes = 2 + 1 * (1 - radii / max_r)

# Create visual objects
scatter_stars = visuals.Markers()
scatter_stars.set_data(initial_star_pos, face_color=star_colors, size=sizes)
view.add(scatter_stars)

scatter_dm = visuals.Markers()
dm_color = np.array([[0.0, 0.0, 0.9, 1]] * initial_dm_pos.shape[0])
scatter_dm.set_data(initial_dm_pos, face_color=dm_color, size=3)
view.add(scatter_dm)

core_pos = np.stack((X_paths[[core_index_sn, core_index_alm], 0],
                     Y_paths[[core_index_sn, core_index_alm], 0],
                     Z_paths[[core_index_sn, core_index_alm], 0]), axis=-1)
core_glow = visuals.Markers()
core_glow.set_data(core_pos, face_color=(.1, .1, .1, 0.2), size=20)
view.add(core_glow)

core_scatter = visuals.Markers()
core_scatter.set_data(core_pos, face_color='black', size=20, edge_color='white', edge_width=2)
view.add(core_scatter)

# Setup writer for animation
frame_index = 0
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M")
filename = f'galactic_sim_{timestamp}.mp4'
writer = imageio.get_writer(filename, fps=30, macro_block_size=None)

# Animation update function
def update(event):
    global frame_index

    if frame_index >= X_paths.shape[1]:
        writer.close()
        timer.stop()
        print('Animation complete')
        return

    # Clip frame_index if it accidentally overflows
    frame_index = min(frame_index, X_paths.shape[1] - 1)

    try:
        # Update star positions
        pos_star_t = np.vstack((
            np.stack((X_paths[star_start_sn:star_end_sn, frame_index],
                      Y_paths[star_start_sn:star_end_sn, frame_index],
                      Z_paths[star_start_sn:star_end_sn, frame_index]), axis=-1),
            np.stack((X_paths[star_start_alm:star_end_alm, frame_index],
                      Y_paths[star_start_alm:star_end_alm, frame_index],
                      Z_paths[star_start_alm:star_end_alm, frame_index]), axis=-1)
        ))
        scatter_stars.set_data(pos_star_t, face_color=star_colors, size=sizes)

        # Update DM positions
        pos_dm_t = np.stack((X_paths[dm_start:dm_end, frame_index],
                             Y_paths[dm_start:dm_end, frame_index],
                             Z_paths[dm_start:dm_end, frame_index]), axis=-1)
        scatter_dm.set_data(pos_dm_t, face_color=dm_color, size=3)

        # Update SMBH core positions
        core_pos_t = np.stack((X_paths[[core_index_sn, core_index_alm], frame_index],
                               Y_paths[[core_index_sn, core_index_alm], frame_index],
                               Z_paths[[core_index_sn, core_index_alm], frame_index]), axis=-1)
        core_glow.set_data(core_pos_t, face_color=(1, 1, 1, 0.15), size=20)
        core_scatter.set_data(core_pos_t, face_color='black', size=20, edge_color='white', edge_width=2)

        # Save frame
        img = canvas.render(size = TARGET_SIZE)
        img = np.asarray(img)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        writer.append_data(img)

        view.camera.azimuth += 0.1

        frame_index += frame_speed

    except Exception as e:
        print(f"Update failed at frame {frame_index}: {e}")
        writer.close()
        timer.stop()

# Start animation
timer = app.Timer(interval=1/60, connect=update, start=True)
app.run()


# Print simulation runtime and video save location
total_seconds = end_time - start_time
print(f"Simulation completed in {total_seconds:.2f} seconds\
       ({total_seconds/60:.2f} minutes).")

print("Video saved at:", os.path.abspath(filename))
