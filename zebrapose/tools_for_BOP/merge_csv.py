import os
import glob
import pandas as pd
import argparse

def main(input_dir, output_fn):
    os.chdir(input_dir)

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv(output_fn, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--input_dir', type=str) 
    parser.add_argument('--output_fn', type=str) 

    args = parser.parse_args()
    input_dir = args.input_dir
    output_fn = args.output_fn
    main(input_dir, output_fn)