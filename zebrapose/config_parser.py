def parse_cfg(cfgfile):
    fp = open(cfgfile, 'r')
    line = fp.readline()
    block = dict()
    training_data_list = []
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()

            if value.isnumeric():
                value = int(value)

            if key == 'learning_rate' or key == 'padding_ratio' or key =='train_obj_visible_theshold' or key =='second_dataset_ratio':
                value = float(value)

            if value == 'False':
                value = False
            elif value == 'True':
                value = True
                
            block[key] = value
        line = fp.readline()

    fp.close()
    return block