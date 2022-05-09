

##### lm or lmo
lmo_obj_name_obj_id = {
    'ape': 1,
    'benchvise': 2,
    'bowl': 3, 
    'cam': 4,
    'can': 5 ,
    'cat': 6,
    'cup': 7,
    'driller': 8 ,
    'duck': 9 ,
    'eggbox': 10,
    'glue': 11,
    'holepuncher': 12,
    'iron': 13,
    'lamp': 14,
    'phone': 15
}

lmo_symmetry_obj = {
    'eggbox',
    'glue',
    'cup',
    'bowl'
}

##### ycbv
ycbv_obj_name_obj_id = {    
    'master_chef_can': 1,
    'cracker_box': 2,
    'sugar_box': 3,
    'tomato_soup_can': 4,
    'mustard_bottle': 5,
    'tuna_fish_can': 6,
    'pudding_box': 7,
    'gelatin_box': 8,
    'potted_meat_can': 9,
    'banana': 10,
    'pitcher_base': 11,
    'bleach_cleanser': 12,
    'bowl': 13,
    'mug': 14,
    'power_drill': 15,
    'wood_block': 16,
    'scissors': 17,
    'large_marker': 18,
    'large_clamp': 19,
    'extra_large_clamp': 20,
    'foam_brick': 21,
}

ycbv_symmetry_obj = {
    'bowl',
    'wood_block',
    'large_clamp',
    'extra_large_clamp',
    'foam_brick'
}


##### tudl
tless_obj_name_obj_id = {
    'obj01': 1,
    'obj02': 2,
    'obj03': 3,
    'obj04': 4,
    'obj05': 5,
    'obj06': 6,
    'obj07': 7,
    'obj08': 8,
    'obj09': 9,
    'obj10': 10,
    'obj11': 11,
    'obj12': 12,
    'obj13': 13,
    'obj14': 14,
    'obj15': 15,
    'obj16': 16,
    'obj17': 17,
    'obj18': 18,
    'obj19': 19,
    'obj20': 20,
    'obj21': 21,
    'obj22': 22,
    'obj23': 23,
    'obj24': 24,
    'obj25': 25,
    'obj26': 26,
    'obj27': 27,
    'obj28': 28,
    'obj29': 29,
    'obj30': 30
}

tless_symmetry_obj = ['obj{:02d}'.format(obj_id+1) for obj_id in range(0,30)] 



##### tudl
tudl_obj_name_obj_id = {
    'obj01': 1,
    'obj02': 2,
    'obj03': 3
}


tudl_symmetry_obj = [] 


def get_obj_info(dataset_name):
    if dataset_name not in ['lmo', 'ycbv', 'tless', 'tudl']:
        raise AssertionError("dataset name unknow")
    return eval("{}_obj_name_obj_id".format(dataset_name)), eval("{}_symmetry_obj".format(dataset_name))