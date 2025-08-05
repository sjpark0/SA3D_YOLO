workdir=$1
colmap feature_extractor --database_path $workdir/database.db --image_path $workdir/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
colmap exhaustive_matcher --database_path $workdir/database.db
mkdir -p $workdir/sparse/0
colmap mapper --database_path $workdir/database.db --image_path $workdir/images --output_path $workdir/sparse
python imgs2poses.py $workdir

