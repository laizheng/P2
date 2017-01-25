cp ../train.p ./
cp ../test.p ./
python dump_pickle_to_files.py
python generate_jittered_image.py
