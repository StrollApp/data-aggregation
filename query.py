import overpy

api = overpy.Overpass()

# test one: (37.87139, -122.27330, 37.87257, -122.27175) 
# city of berkeley: (37.84369, -122.33657, 37.91079, -122.22803)

ways_to_remove = [42213220, 42213222, 42213221, 42213224, 233174480, 43843096, 233174481, 37890122, 429259488, 468644955, 755808071, 843904961, 109192573, 410615368, 410615363, 431330572, 760603475, 760603476, 26131333, 774106512, 37061369, 825518424, 28997628, 315995498, 315995499, 520151688, 520151686, 327897036, 431330572, 271976774, 760603475, 763736266, 669491504, 669491501, 428833982, 428833982, 454023413, 88800450, 429259483, 34361074, 34361074, 874580606, 685038710, 685038711, 702596278, 843765026, 262514539, 262514522, 262514531, 906507793, 836651243, 220677320, 35973828, 6347978, 34865267, 836254397, 836254398, 34865268, 314580358, 314580358, 648979893, 648979895, 26379233, 614080393, 677381359, 25947958, 6405082, 26379231, 25947963, 26379230, 24785079, 26379230, 394991558, 658636582, 857408775, 430174931, 440149767, 757212144, 28997620, 184898397, 190385734, 190385739, 42638944, 190380686, 30734877, 190380687, 309025800, 190385736, 190385737, 28997620, 400536078, 400536079, 293158397, 293158396, 512366677, 293158402]
ways_to_add = [223940141]
region = "(37.84369, -122.33657, 37.91079, -122.22803)"

query_string = "((way" + region + """ ["highway"] ["service" != "alley"] ["service" != "emergency_access"] ["service" != "drive-through"] ["service" != "slipway"] ["service" != "parking_aisle"] ["service" != "driveway"] ["highway" != "motorway"] ["highway" != "motorway_link"] ["highway" != "corridor"] ["access" != "no"] ["access" != "private"];\n"""

for way in ways_to_add:
  query_string += "way(" + str(way) + ");\n"

query_string += "); - ("

for way in ways_to_remove:
  query_string += "way(" + str(way) + ");\n"

query_string += """
    );
    );
      (._;>;);
      out body;
"""

# print(query_string)

result = api.query(query_string)

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