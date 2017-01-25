wget https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip
unzip traffic-signs-data.zip
python dump_pickle_to_files.py
python generate_jittered_image.py
python main.py
