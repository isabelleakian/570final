#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
# import simple_tsp



# In[2]:


from torch.utils.data import DataLoader

# import simple_tsp
from generate_data import generate_vrp_data
from utils import load_model
from problems import CVRP, TSP

# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def coord_to_loc(dataset, coord):
    locs = dataset['loc'].cpu().numpy()

    return np.where(locs == coord)[0][0]+1

def get_routes_and_coords(datasets, route_for_coord):
    route_list = [r[r != 0] for r in np.split(route_for_coord.cpu().numpy(), np.where(route_for_coord == 0)[0]) if (r != 0).any()]
    # print(route_list)
    new_locs = datasets['loc'].cpu().numpy()
    # print(new_locs)
    # print(new_locs)
    # print(len(routes))
    new_coords = [[]]*len(route_list)

    for veh_number, r in enumerate(route_list):
        # print(veh_number, r)
        # for i in enumerate(r[1]):
        new_coords[veh_number] = new_locs[r - 1, :]
    # print(new_coords)
    return new_coords, route_list


def plot_vehicle_routes(data, route, ax1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    
    # route is one sequence, separating different routes with 0 (depot)
    routes = [r[r!=0] for r in np.split(route.cpu().numpy(), np.where(route==0)[0]) if (r != 0).any()]
    # print(routes)
    depot = data['depot'].cpu().numpy()
    locs = data['loc'].cpu().numpy()
    demands = data['demand'].cpu().numpy() * demand_scale
    capacity = demand_scale # Capacity is always 1

    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        # print(r)
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order

        route_demands = demands[r - 1]
        coords = locs[r - 1, :]
        # print("coord", coords)
        xs, ys = coords.transpose()

        total_route_demand = sum(route_demands)
        # assert total_route_demand <= capacity
        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d
            
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                int(total_route_demand) if round_demand else total_route_demand, 
                int(capacity) if round_demand else capacity,
                dist
            )
        )
        
        qvs.append(qv)
        
    ax1.set_title('{} routes, total distance {:.2f}'.format(len(routes), total_dist))
    ax1.legend(handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)


# In[4]:


model1, _ = load_model('pretrained/cvrp_100/')
model2, _ = load_model('pretrained/tsp_100/')

torch.manual_seed(1234)
dataset = CVRP.make_dataset(size=100, num_samples=10)


# In[5]:


# Need a dataloader to batch instances
dataloader = DataLoader(dataset, batch_size=1000)
# print(dataset)
# Make var works for dicts
batch = next(iter(dataloader))
# print(batch['loc'].numpy().shape)
# print(batch)

# data_batch = list(batch.items())
# oracle_array = np.array(data_batch, 2)
# Run the model
model1.eval()
model1.set_decode_type('greedy')
model2.eval()
model2.set_decode_type('greedy')

# oracle = simple_tsp.make_oracle(model2, oracle_array)

with torch.no_grad():
    length1, log_p1, pi1 = model1(batch, return_pi=True)
    # print(batch)
    # length2, log_p2, pi2 = model2(batch['loc'], return_pi=True)
    # oracle = simple_tsp.make_oracle(model2, np.array(batch))
    #
    # tour_tsp = []
    # tour_p = []
    # while len(tour_tsp) < length1:
    #     p = oracle(tour_tsp)
    #     i = np.argmax(p)
    #     tour_tsp.append(i)
    #     tour_p.append(p)
    #
    # print(tour_tsp)

# if length1 < length2:
tours = pi1
# print(tours.numpy().shape)
# print(tours)
# print(pi2.numpy().shape)
# print(pi2)

for i, (data, tour) in enumerate(zip(dataset, tours)):
    # print(tour.numpy().shape)
    # print(tour)
    # print("hello")
    fig, ax = plt.subplots(figsize=(10, 10))
    # print("vrp\n")
    plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)
    fig.savefig(os.path.join('images', 'cvrp_v2{}.png'.format(i)))

for i, (data, tour) in enumerate(zip(dataset, tours)):
    newcoords, new_routes = get_routes_and_coords(data, tour)
    # print(len(newcoords))
    newroutes = []
    # j=0
    for k in newcoords:
        with torch.no_grad():
            # print(k)
            # print(torch.FloatTensor([k]).numpy().shape)
            # print(batch['loc'].numpy().shape)
            length2, log_p2, pi2 = model2(torch.FloatTensor([k]), return_pi=True)
        # print("pi2", length2, log_p2, pi2)
        # j=j+1
        tour2 = pi2.tolist()[0]

        for j in range(len(tour2)):
            tour2[j] = coord_to_loc(data, k[j])
        # print(tour2)
        newroutes = newroutes + tour2
        newroutes.append(0)
    fig, ax = plt.subplots(figsize=(10, 10))
    # print(tour.numpy())
    # print("\n\n")
    # print(newroutes)
    # print("tsp\n")
    plot_vehicle_routes(data, torch.IntTensor(newroutes), ax, visualize_demands=False, demand_scale=50, round_demand=True)
    fig.savefig(os.path.join('images', 'cvrp_and_tsp_{}.png'.format(i)))
    # tspdata = DataLoader(TSP.make_dataset(size=100, num_samples=10), batch_size=1000)
# tspdata = DataLoader(TSP.make_dataset(inputdata=tours.numpy(), num_samples=10), batch_size=1000)
# batch = next(iter(tspdata))
# print(batch)
# with torch.no_grad():
#     length2, log_p2, pi2 = model2(batch['loc'], return_pi=True)

# length2, log_p2, pi2 = model2(tours, return_pi=True)
# tours2 = pi2
# print(tours2)
# else:

# Plot the results
# for i, (data, tour) in enumerate(zip(dataset, tours2)):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     plot_vehicle_routes(data, tour, ax, visualize_demands=False, demand_scale=50, round_demand=True)
#     fig.savefig(os.path.join('images', 'cvrp_and_tsp_{}.png'.format(i)))
# animation = simple_tsp.plot_tsp_ani(data, tour, tour_p).animate(interval=500)
# writer = PillowWriter(fps=2)
# ani.save("demo_sine.gif", writer=writer)
# animation.save('images/tsp_adjusted.gif', writer='writer', fps=2)




# In[ ]:




