import overpy

api = overpy.Overpass()

#test one: (37.87139, -122.27330, 37.87257, -122.27175) 

result = api.query("""
    way(37.84369, -122.33657, 37.91079, -122.22803) ["highway"] ["service" != "alley"] ["service" != "emergency_access"] ["service" != "drive-through"] ["service" != "slipway"] ["service" != "parking_aisle"] ["service" != "driveway"] ["highway" != "motorway"] ["highway" != "motorway_link"] ["highway" != "corridor"];
    (._;>;);
    out body;
    """)

wayFile = open("berkeleyWays.csv", "w")
wayFile.write("name,highway,node_ids\n")
nodeFile = open("berkeleyNodes.csv", "w")
nodeFile.write("id,latitude,longitude\n")

for way in result.ways:
    way_name = way.tags.get("name", "n/a")
    highway_type = way.tags.get("highway", "n/a")
    wayFile.write(way_name + ',' + highway_type + ',')

    node_ids = ''
    for node in way.nodes:
        node_ids = node_ids + str(node.id) + '-'
    node_ids = node_ids[:-1]
    wayFile.write(node_ids + '\n')

for node in result.nodes:
    nodeFile.write(str(node.id) + ',')
    print(f'{node.lat:.7f},{node.lon:.7f}', file=nodeFile)

wayFile.close()
nodeFile.close()
