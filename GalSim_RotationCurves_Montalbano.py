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
duration = 1E6 # years
dt = np.float32(200) 
num_steps = int(duration / dt)


epsilon = np.float32(0.05)  # Softening parameter to avoid division by zero
frame_speed = 40 # number of steps per frame


# Galaxy Structure
num_total_starblocks = 5000 # does not include dm particles

disk_fraction = 0.8 # disk_fraction and bulge_fraction should sum to one
bulge_fraction = 1 - disk_fraction
dm_fraction = 1

num_disk_starblocks = int(num_total_starblocks * disk_fraction)
num_bulge_starblocks = int(num_total_starblocks * bulge_fraction)
num_dm_blocks = int(num_total_starblocks * dm_fraction)

star_density = np.float32(1E7) # stars in a starblock
starblock_mass = np.float32(1 * star_density) # solar masses * density
dm_block_mass = np.float32(3 * star_density) 

max_disk_radius = np.float32(100) # max starblock radius in kpc
scale_radius = np.float32(4) 
scale_height = np.float32(0.3)
bulge_scale_radius = np.float32(0.1) 
dm_scale_radius = np.float32(20) 

disk_mass = np.float32(num_disk_starblocks * starblock_mass)
bulge_mass = np.float32(num_bulge_starblocks * starblock_mass)
core_mass = np.float32(1E7)  # SMBH

num_arms = 2
pitch_angle = np.pi / 6 # How tighly wound
arm_width = np.pi / 3.5 # How thick


# Create bins for rotation curve plot
bin_num = 200
binsize = np.float32(max_disk_radius / bin_num)
r_bin = np.zeros(1 + num_total_starblocks, dtype=np.float32)





# Force of gravity applied to all non dm bodies
@cuda.jit
def gravity_step(positions, masses, accelerations):
    i = cuda.grid(1)
    N = positions.shape[0]

    if i >= N - num_dm_blocks:
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
def leapfrog_step(positions, velocities, masses, binsize):
    
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
    for i in range(1 + num_total_starblocks + num_dm_blocks):
        v = velocities_half[i]
        speed = np.linalg.norm(v)
        if speed > c:
            velocities_half[i] = v * (c / speed)

    new_positions = positions.copy()
    new_positions[:1 + num_total_starblocks] += velocities_half[:1 + num_total_starblocks] * dt

    r_bin = np.empty(positions.shape[0])
    for i in range(1 + num_total_starblocks):
        r_bin[i] = np.linalg.norm(new_positions[i])/binsize

    d_new_positions = cuda.to_device(new_positions)
    d_new_accelerations = cuda.device_array((N, 3), dtype=np.float32)
    gravity_step[blocks_per_grid, threads_per_block](d_new_positions, d_masses, d_new_accelerations)
    new_accelerations = d_new_accelerations.copy_to_host()  
    new_velocities = velocities_half + 0.5 * new_accelerations * dt

    # Cap final velocity
    for i in range(1 + num_total_starblocks):
        v = new_velocities[i]
        speed = np.linalg.norm(v)
        if speed > c:
            new_velocities[i] = v * (c / speed)


    return new_positions, new_velocities, r_bin


# Calculate mass enclosed in goussian surface,
# used in initial velocity calcuations, and theoretical rotation curve plot
def enclosed_mass(r, disk_mass, dark_mass):
    dm_mass = hernquist_enclosed_mass(r, dark_mass, dm_scale_radius)
    x_disk = r / scale_radius
    bulge_mass_enclosed = hernquist_enclosed_mass(r, bulge_mass, bulge_scale_radius)
    return disk_mass * (1 - np.exp(-x_disk) * (1 + x_disk + 0.5 * x_disk**2)) \
        + core_mass + bulge_mass_enclosed + dm_mass


# Uses Hernquist Profile to calculate bulge mass distribution,
# used in max initial velocity calculation for bulge stars
def hernquist_enclosed_mass(r, total_mass, scale_radius):
    return total_mass * (r**2) / (r + scale_radius)**2





# Create arrays for block data
positions = cuda.pinned_array((1 + num_total_starblocks + num_dm_blocks, 3),\
                               dtype=np.float32)
masses = cuda.pinned_array((1 + num_total_starblocks + num_dm_blocks),\
                            dtype=np.float32)
velocities = cuda.pinned_array((1 + num_total_starblocks + num_dm_blocks, 3),\
                                dtype=np.float32)
orbital_radii = np.zeros(1 + num_total_starblocks + num_dm_blocks,\
                          dtype=np.float32)

idx = 0


# Place SMBH
core_index = 0
positions[idx] = [0, 0, 0]
velocities[idx] = [0, 0, 0]
masses[idx] = core_mass

idx += 1





# Create Bulge starblocks
for bulge_block in range(num_bulge_starblocks // 2):
    r_draw = random.uniform(0, 1)
    orbdist = bulge_scale_radius * r_draw / (1 - r_draw)
    if orbdist > 2:
        continue

    Mass_enc = hernquist_enclosed_mass(orbdist, bulge_mass, bulge_scale_radius)
    v_max = np.sqrt(G * Mass_enc / (orbdist + epsilon/4))

    while True:
        v = random.uniform(0, v_max)
        f_Eddington = v**2 * (1 - (v / v_max)**2)**3.5
        f_peak = 0.1 * v_max**2
        if random.uniform(0, f_peak) < f_Eddington:
            break

    theta, phi = random.uniform(0, np.pi), random.uniform(0, 2 * np.pi)
    x = orbdist * np.sin(theta) * np.cos(phi)
    y = orbdist * np.sin(theta) * np.sin(phi)
    z = orbdist * np.cos(theta)

    v = min(v, 0.95 * v_max)
    theta_v, phi_v = random.uniform(0, np.pi), random.uniform(0, 2 * np.pi)
    vx = v * np.sin(theta_v) * np.cos(phi_v)
    vy = v * np.sin(theta_v) * np.sin(phi_v)
    vz = v * np.cos(theta_v)

    for sign in [1, -1]:
        positions[idx] = sign * np.array([x, y, z], dtype=np.float32)
        velocities[idx] = sign * np.array([vx, vy, vz], dtype=np.float32)
        masses[idx] = np.float32(starblock_mass)
        idx += 1


# Create Disk starblocks
for block in range(num_disk_starblocks):
    draw = random.uniform(0, 1)
    h_draw = np.random.uniform(0, 1)
    orbdist = min(-scale_radius * np.log(1 - draw), max_disk_radius)

    # Create spiral arms
    while True:
        phi = random.uniform(0, 2*np.pi)

        spiral_phase = (np.log(orbdist / scale_radius) / np.tan(pitch_angle))
        arm_number = random.randint(0, num_arms - 1) 
        desired_phi = np.mod(spiral_phase * num_arms + arm_number * 2*np.pi/num_arms, 2*np.pi)

        delta_phi = np.mod(phi - desired_phi, 2*np.pi)
        if delta_phi > np.pi:
            delta_phi = 2*np.pi - delta_phi

        prob = np.exp(-(delta_phi / arm_width)**2)
        if random.random() < prob:
            break

    x = orbdist * np.cos(phi)
    y = orbdist * np.sin(phi)
    z = scale_height * np.arcsinh(np.tan(np.pi * (h_draw - 0.5)))

    r = np.sqrt(orbdist**2 + z**2)
    theta = np.arcsin(z / r)

    v_hat = np.array([-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta),\
                -np.sin(theta)])

    Mass_enc = enclosed_mass(r, disk_mass, num_dm_blocks * dm_block_mass)
    v_mag = np.sqrt(G * Mass_enc / (r + epsilon))
    base_velocity = v_hat * v_mag

    # Add in random kicks to break ringing
    velocity_dispersion = 0.05
    random_kick = velocity_dispersion * v_mag * np.random.randn(3)

    positions[idx] = [x, y, z]
    velocities[idx] = base_velocity + random_kick
    masses[idx] = starblock_mass
    orbital_radii[idx] = orbdist

    idx += 1


# Create dark matter blocks
for wimp in range(num_dm_blocks):
    dm_Draw = np.random.uniform(0,1)
    orbdist = dm_scale_radius * (np.sqrt(dm_Draw) / (1 - np.sqrt(dm_Draw)))
    theta, phi = random.uniform(0, np.pi), random.uniform(0, 2 * np.pi)

    x = orbdist * np.sin(phi)*np.sin(theta)
    y = orbdist * np.cos(phi)*np.sin(theta)
    z = orbdist * np.cos(theta)
    r = np.sqrt(orbdist**2 + z**2)

    v_hat = np.array([-np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta),\
                -np.sin(theta)])
    
    Mass_enc = enclosed_mass(r, disk_mass, num_dm_blocks * dm_block_mass)
    v_mag = np.sqrt(G * Mass_enc / (r + epsilon))
    base_velocity = v_hat * v_mag

    velocity_dispersion = 0.05
    random_kick = velocity_dispersion * v_mag * np.random.randn(3)

    positions[idx] = np.array([x,y,z])
    velocities[idx] = base_velocity + random_kick
    masses[idx] = dm_block_mass
    idx+=1

center_of_mass = np.mean(positions[1:num_total_starblocks], axis=0)



# Create block path storage
X_paths = cuda.pinned_array((1 + num_total_starblocks + num_dm_blocks, num_steps),\
                             dtype=np.float32)
Y_paths = cuda.pinned_array((1 + num_total_starblocks + num_dm_blocks, num_steps),\
                             dtype=np.float32)
Z_paths = cuda.pinned_array((1 + num_total_starblocks + num_dm_blocks, num_steps),\
                             dtype=np.float32)

Vel_Rad_Array = np.zeros((1 + num_total_starblocks + num_dm_blocks, 2, num_steps))


# Start the timer
start_time = time.time()  


# Simulate galaxy over dt
for t in range(num_steps):
    X_paths[:, t] = positions[:, 0]
    Y_paths[:, t] = positions[:, 1]
    Z_paths[:, t] = positions[:, 2]
    
    
    for i in range(1 + num_total_starblocks):
        r= np.linalg.norm(positions[i])
        v = np.linalg.norm(velocities[i])
        Vel_Rad_Array[i, 0, t] = r
        Vel_Rad_Array[i, 1, t] = v

    positions, velocities, r_bin = leapfrog_step(positions, velocities, masses, binsize)
    
    # Print upon completion of 1000 steps
    if t % 1000 == 0:
        elapsed = time.time() - start_time
        print(f"Step {t}/{num_steps} completed in {elapsed:.2f} seconds")


# End the timer
end_time = time.time()  





# Bin the data
star_indices = np.arange(1, num_total_starblocks) 
radii_final = Vel_Rad_Array[star_indices, 0, 0]
velocities_final = Vel_Rad_Array[star_indices, 1, 0]
v_min, v_max = 0, 3e5  
v_real = (velocities_final > v_min) & (velocities_final < v_max)

bins = np.linspace(1, max_disk_radius, bin_num + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
mean_velocities = np.zeros(bin_num)
counts = np.zeros(bin_num)

for r, v in zip(radii_final[v_real], velocities_final[v_real]):
    bin_index = np.searchsorted(bins, r) - 1
    if 0 <= bin_index < bin_num:
        mean_velocities[bin_index] += v
        counts[bin_index] += 1

min_radius = 1 
nonzero = (counts > 5) & (bin_centers > min_radius)
mean_velocities[nonzero] /= counts[nonzero]
mean_velocities *= 9.79E8





# Create Animation
canvas = scene.SceneCanvas(keys='interactive', show=True,\
        bgcolor=(0.01, 0.01, 0.03, 1), size=(2560, 1440))
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(elevation=20, azimuth=0,\
                                             distance=20000, fov=1)
view.camera.distance = 15000
view.camera.center = tuple(center_of_mass)

dm_start = num_total_starblocks
dm_end = num_total_starblocks + num_dm_blocks


# Start positions
initial_star_pos = np.stack((X_paths[1:num_total_starblocks, 0],\
            Y_paths[1:num_total_starblocks, 0], Z_paths[1:num_total_starblocks, 0]), axis=-1)
initial_dm_pos = np.stack((X_paths[dm_start:dm_end, 0], \
            Y_paths[dm_start:dm_end, 0], Z_paths[dm_start:dm_end, 0]), axis=-1)

# Create starblock colors (bright reddish-white to bright blue-white)
num_stars = initial_star_pos.shape[0]

# Random numbers to decide star colors
color_randoms = np.random.uniform(0, 1, num_stars)
star_colors = np.zeros((num_stars, 4))  # RGBA

for i in range(num_stars):
    t = color_randoms[i]
    # interpolate from red-white to blue-white
    r = 1 - 0.2 * t    # starts bright pinkish (0.95) -> shifts slightly down to 0.85
    g = 0.7 + 0.2 * t    
    b = 0.9 + 0.1 * t     
    star_colors[i] = (r, g, b, 1.0)  # Full opacity


# Create starblock sizes
radii = np.linalg.norm(initial_star_pos, axis=1)
max_r = np.max(radii)
sizes = 2 +  1 * (1 - radii / max_r)


# Plot Start positions with colors
scatter_stars = visuals.Markers()
scatter_stars.set_data(initial_star_pos, face_color=star_colors, size=sizes)
view.add(scatter_stars)



# Plot dark matter start positions
dm_color = np.array([[0.0, 0.0, 0.9, 1]] * num_dm_blocks)  # dark blue + transparent
scatter_dm = visuals.Markers()
scatter_dm.set_data(initial_dm_pos, face_color=dm_color, size=3)
view.add(scatter_dm)


# Give SMBH unique look
core_pos = np.stack((X_paths[[core_index], 0], Y_paths[[core_index], 0],\
                      Z_paths[[core_index], 0]), axis=-1)
core_glow = visuals.Markers()
core_glow.set_data(core_pos, face_color=(.1, .1, .1, 0.2), size=20)
view.add(core_glow)


# Plot initial SMBH position
core_scatter = visuals.Markers()
core_scatter.set_data(core_pos, face_color='black', size=20,\
                       edge_color='white', edge_width=2)
view.add(core_scatter)


# Update animation
frame_index = 0
loop_animation = False
frames = []
writer = imageio.get_writer('galaxy_simulation.mp4', fps=30, macro_block_size=None)
zooming = True

def update(event):
    global frame_index, zooming
    if frame_index >= X_paths.shape[1]:
        if loop_animation:
            frame_index = 0
        else:
            timer.stop()
            writer.close() 
            return

    # Stars
    pos_star_t = np.stack((
        X_paths[1:num_total_starblocks, frame_index],
        Y_paths[1:num_total_starblocks, frame_index],
        Z_paths[1:num_total_starblocks, frame_index]), axis=-1)
    scatter_stars.set_data(pos_star_t, face_color=star_colors, size=sizes)

    # DM
    pos_dm_t = np.stack((
        X_paths[dm_start:dm_end, frame_index],
        Y_paths[dm_start:dm_end, frame_index],
        Z_paths[dm_start:dm_end, frame_index]), axis=-1)
    scatter_dm.set_data(pos_dm_t, face_color=dm_color, size=3)

    # Core
    core_pos_t = np.stack((
        X_paths[[core_index], frame_index],
        Y_paths[[core_index], frame_index],
        Z_paths[[core_index], frame_index]), axis=-1)
    core_glow.set_data(core_pos_t, face_color=(1, 1, 1, 0.15), size=20)
    core_scatter.set_data(core_pos_t, face_color='black', size=20,\
                           edge_color='white', edge_width=2)

    if zooming:
        if view.camera.distance > 1500:
            view.camera.distance -=150
        else:
            view.camera.distance = 1500
            zooming = False  
    if not zooming: 
        if view.camera.elevation < 85:
                view.camera.elevation += 1
        

    img = canvas.render()
    img = Image.fromarray(img)
    img = img.resize((2560, 1440), Image.LANCZOS)  # high quality downscale
    img = np.asarray(img)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    writer.append_data(img)

    frame_index += frame_speed  # Skip 10 frames


# Run animation
timer = app.Timer(interval=1/60, connect=update, start=True)
app.run()





# Create theoretical rotation curve with no DM
def v_circ_theory(r):
    Visible_mass = np.sum(masses[1 + num_bulge_starblocks: 1 + num_total_starblocks])
    M = enclosed_mass(r, Visible_mass, 0)
    return np.sqrt(G * M / (r + epsilon))

r_theory = bin_centers[nonzero]
v_theory = np.array([v_circ_theory(r) for r in r_theory])
v_theory *= 9.79E8 # kpc/yr to km/s





# Plot theoretical rotation curve without DM and simulated rotation curve
plt.figure(figsize=(8, 5))
plt.plot(r_theory, v_theory, linestyle='--', label='Theoretical(No DM)')
plt.plot(bin_centers[nonzero], mean_velocities[nonzero],\
          marker='o', label='Simulated Galaxy')
plt.xlabel("Radius (kpc)")
plt.ylabel("Average Circular Velocity (km/s)")
plt.title("Galaxy Rotation Curve")
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


# Print simulation runtime and video save location
total_seconds = end_time - start_time
print(f"Simulation completed in {total_seconds:.2f} seconds\
       ({total_seconds/60:.2f} minutes).")
print("Video saved at:", os.path.abspath('galaxy_simulation.mp4'))