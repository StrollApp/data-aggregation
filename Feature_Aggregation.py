import numpy as np
import pandas as pd
import math
from heapq import heapify, heappush, heappop
import re

street_data = pd.read_csv('berkeleyWays.csv')
node_data = pd.read_csv('berkeleyNodes.csv') # TODO: multiply by 10^7 and store as int
edge_data = None
node_set = set()
node_id_to_index = {}

# determines the degrees of nodes to see what we can delete
node_degrees = {}

node_data['id'] = node_data['id'].apply(int)
node_data['latitude'] = node_data['latitude'].apply(float)
node_data['longitude'] = node_data['longitude'].apply(float)

for cell in node_data['id']:
    node_degrees[cell] = 0
    
for index, row in street_data.iterrows():
    string_of_nodes = row['node_ids']
    list_of_nodes = string_of_nodes.split('-')
    for i in range(len(list_of_nodes)):
        if len(list_of_nodes) == 0:
            break
        node = int(list_of_nodes[i])
        if i == 0 or i == len(list_of_nodes) - 1:
            if node in node_degrees:
                node_degrees[node] += 1
        else:
            if node in node_degrees:
                node_degrees[node] += 2

# scans through all nodes and keeps only relevant ones, adds every relevant segment to edge_data
node_hashmap = node_data.set_index('id').T.to_dict('list')

THRESHOLD = math.pi / 6
CENTER = math.pi

def get_angle(curr_coords, prev_coords, next_coords):
    b = np.array(curr_coords)
    a = np.array(prev_coords)
    c = np.array(next_coords)
    ba = a - b
    bc = c - b
    prev_angle = math.atan2(ba[1], ba[0])
    next_angle = math.atan2(bc[1], bc[0])
    return (next_angle - prev_angle) % (2 * math.pi)

edge_list = {}
edge_list['name'] = []
edge_list['start_id'] = []
edge_list['end_id'] = []
edge_list['highway'] = []

for index, row in street_data.iterrows():
    string_of_nodes = row['node_ids']
    list_of_nodes = string_of_nodes.split('-')
    nodes_to_keep = []
    last = 0
    nodes_to_keep.append(int(list_of_nodes[0]))
    for i in range(1, len(list_of_nodes) - 1):
        curr_node = int(list_of_nodes[i])
        prev_node = int(list_of_nodes[last])
        next_node = int(list_of_nodes[i + 1])
        if node_degrees[curr_node] > 2: # this is an intersection
            nodes_to_keep.append(curr_node)
            last = i
            continue
        curr_coords = node_hashmap[curr_node]
        prev_coords = node_hashmap[prev_node]
        next_coords = node_hashmap[next_node]
        angle = get_angle(curr_coords, prev_coords, next_coords)
        if abs(angle - CENTER) < THRESHOLD: # turn in the road
            continue
        nodes_to_keep.append(curr_node)
        last = i
    if len(nodes_to_keep) > 0 and nodes_to_keep[0] != int(list_of_nodes[len(list_of_nodes) - 1]):
        nodes_to_keep.append(int(list_of_nodes[len(list_of_nodes) - 1]))
    for i in range(len(nodes_to_keep) - 1):
        edge_list['name'].append(row['name'])
        edge_list['highway'].append(row['highway'])
        edge_list['start_id'].append(nodes_to_keep[i])
        edge_list['end_id'].append(nodes_to_keep[i + 1])
    for node in nodes_to_keep:
        node_set.add(node)
        
edge_data = pd.DataFrame(edge_list)
node_data = node_data[node_data['id'].isin(node_set)]
for index, row in node_data.iterrows():
    node_id_to_index[int(row['id'])] = index

# display(node_data)
# display(edge_data)

# adds columns for all other features a street segment could have

nan = [None for b in range(len(edge_data))]

features = [
    'crime_count', 
    'tree_count', 
    'light_count', 
    'business_count', 
    'signal_count', 
    'pavement_width', 
    'street_type', 
    'crime_ratio', 
    'tree_ratio', 
    'light_ratio', 
    'business_ratio', 
    'signal_ratio', 
    'region']

for feature in features:
    edge_data[feature] = nan

# creates adjacency list for nodes
node_adj = {}

for node in node_set:
    node_adj[node] = {}
    adjacent_edges = edge_data.loc[(edge_data['start_id'] == node) | (edge_data['end_id'] == node)].index.tolist()
    node_adj[node] = set(adjacent_edges)

# writes map data to csv files
edge_data.to_csv('Map_Edges.csv')
node_data.to_csv('Map_Nodes.csv')
budu = [0]


# utility functions to get distances

def get_distance_btwn_points(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# finds distance between c = (x3, y3) to line defined by a = (x1, y1) and b = (x2, y2)
def get_distance_btwn_point_and_line(x1, y1, x2, y2, x3, y3):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3])
    ab = p2 - p1
    ba = p1 - p2
    ac = p3 - p1
    bc = p3 - p2
    bac = np.dot(ab, ac)
    cba = np.dot(ba, bc)
    if bac < 0 and cba < 0:
        print('degens')
    elif bac < 0:
        return np.linalg.norm(ac)
    elif cba < 0:
        return np.linalg.norm(bc)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


# gets street segment indices for a latitude and longitude

index_list = edge_data.index.tolist()
start_ids = edge_data['start_id'].tolist()
end_ids = edge_data['end_id'].tolist()
start_indices = [node_id_to_index[b] for b in start_ids]
end_indices = [node_id_to_index[b] for b in end_ids]
start_latitudes = [node_data.at[start_index, 'latitude'] for start_index in start_indices]
start_longitudes = [node_data.at[start_index, 'longitude'] for start_index in start_indices]
end_latitudes = [node_data.at[end_index, 'latitude'] for end_index in end_indices]
end_longitudes = [node_data.at[end_index, 'longitude'] for end_index in end_indices]

# gets the street segment closest to the latitude and longitude of a given point
# current implementation will assume streets are straight lines and the earth is flat
# also current implementation goes through all edges which is slow, implement regions in the future
# REQUIRES intersections to have coordinates
def get_block(latitude, longitude):
    min_distance = float('inf')
    min_street_index = -1
    for index, start_latitude, start_longitude, end_latitude, end_longitude in zip(index_list, start_latitudes, start_longitudes, end_latitudes, end_longitudes):
        current_distance = get_distance_btwn_point_and_line(
            start_latitude, start_longitude, end_latitude, end_longitude, latitude, longitude)
        if current_distance < min_distance:
            min_distance = current_distance
            min_street_index = index
    return min_street_index

# gets the k closest segments to the given point
# REQUIRES intersections to have coordinates
def get_closest_blocks(latitude, longitude, k):
    print(budu[0])
    budu[0] += 1
    pq = []
    for index, start_latitude, start_longitude, end_latitude, end_longitude in zip(index_list, start_latitudes, start_longitudes, end_latitudes, end_longitudes):
        current_distance = get_distance_btwn_point_and_line(
            start_latitude, start_longitude, end_latitude, end_longitude, latitude, longitude)
        pq.append((current_distance, index))
    pq.sort()
    closest = [pq[i][1] for i in range(k)]
    return closest

# increments the value of parameter at the k street segments closest to location
def update_street_data(latitude, longitude, parameter, k = 1):
    if k == 1:
        index = get_block(latitude, longitude)
        if edge_data.at[index, parameter] is None:
            edge_data.at[index, parameter] = 0
        edge_data.at[index, parameter] += 1
    else:
        index = get_closest_blocks(latitude, longitude, k)
        if index:
            for block in index:
                if edge_data.at[block, parameter] is None:
                    edge_data.at[block, parameter] = 0
                edge_data.at[block, parameter] += 1
                
def update_street_data_coords(coords, parameter, k = 1):
    update_street_data(coords[0], coords[1], parameter, k)

# adds crime data

crime = pd.read_csv('crimes.csv')
crime = crime[['Block_Location']]
pattern = '\((.*)\)'

def extract_coords(given_string, split, lat_first = True):
    s = re.search(pattern, given_string).group(1)
    coords = s.split(split)
    if lat_first:
        return float(coords[0]), float(coords[1])
    return float(coords[1]), float(coords[0])

crime['Block_Location'] = crime['Block_Location'].apply(extract_coords, args = (', ', True))

crime['Block_Location'].apply(update_street_data_coords, args=('crime_count', 3))

display(edge_data)

edge_data.to_csv('Edge_Data_Crime.csv')

# adds tree data
trees = pd.read_csv('City_Trees.csv')
trees = trees[['Latitude', 'Longitude']]

trees['coordinates'] = list(zip(trees.Latitude, trees.Longitude))

trees['coordinates'].apply(update_street_data_coords, args=('tree_count', 1))

display(edge_data)

edge_data.to_csv('Edge_Data_Tree.csv')

# adds light data
streetLights = pd.read_csv('streetLights.csv')
streetLights = streetLights[['the_geom']]

streetLights['the_geom'] = streetLights['the_geom'].apply(extract_coords, args = (' ', False))

streetLights['the_geom'].apply(update_street_data_coords, args=('light_count', 1))

display(edge_data)

# converts DataFrame into csv files

# creates dash separated list of edge indices adjacent to a node
def create_string_from_set(node):
    adj = [str(b) for b in node_adj[node]]
    return '-'.join(adj)

node_data['adjacencies'] = [None for b in range(len(node_data))]
node_data['adjacencies'] = node_data['id'].apply(create_string_from_set)

# writes to files
node_data.to_csv('Node_Data.csv')
edge_data.to_csv('Edge_Data.csv')