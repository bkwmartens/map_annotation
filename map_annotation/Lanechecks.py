import os
import geopandas as gpd
# plotting lanes
df = gpd.read_file(f'{os.environ["MA_DATA_DIR"]}/lanes.gpkg')
lanes = gpd.GeoDataFrame.explode(df, index_parts=False)

# lanes = lanes.to_string()

# print(lanes)
# print(lanes['lane_id'])

print(lanes)
exit()
    
for col, item in lanes.iterrows():
    if (item['boundary_right'] == True and item['boundary_left'] == True) or (item['boundary_right'] == False and item['boundary_left'] == False):
        assert(f'Error! Lane has incorrect boundary booleans.')
        print(item['element_id'])

    lane_id = item['lane_id']

    for col, item2 in lanes.iterrows():
        if item['lane_id'] == item2['lane_id']:
            if item['element_id'] != item2['element_id']:
                if item['boundary_left'] == item2['boundary_left'] or item['boundary_right'] == item2['boundary_right']:
                    print(item['lane_id'], 'has inconsistent boundaries')
                if item['one-way'] != item2['one-way']:
                    print(item['lane_id'], 'has inconsistent direction')

print('Done, all good to go!')

    #df = lanes.query('lane_id == lane_id')
    #print(df)
