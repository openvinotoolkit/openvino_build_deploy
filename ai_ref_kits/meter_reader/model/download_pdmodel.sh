set -e

wget https://bj.bcebos.com/paddlex/examples2/meter_reader/meter_det_model.tar.gz
wget https://bj.bcebos.com/paddlex/examples2/meter_reader/meter_seg_model.tar.gz

echo "dede311e88ab272e5ad1b9710f4e40dee51279b388a387dabfeab2a159da6d6a meter_det_model.tar.gz" | sha256sum --check
echo "060e0f3616c4359dfc6f6d1867a32de2d7761694652311dc1c28262e9afc51cb meter_seg_model.tar.gz" | sha256sum --check

mkdir -p analog

tar -xvf meter_det_model.tar.gz -C ./analog
tar -xvf meter_seg_model.tar.gz -C ./analog

rm meter_det_model.tar.gz meter_seg_model.tar.gz