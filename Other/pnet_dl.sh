url=https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip
zip_path=~/data/shapenetcore_partanno_segmentation_benchmark_v0.zip
wget ${url} --no-check-certificate -O ${zip_path}
unzip -q ${zip_path} -d ~/data/
rm -rf ${zip_path}
