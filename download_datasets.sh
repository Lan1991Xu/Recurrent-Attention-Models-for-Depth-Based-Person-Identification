#!/usr/bin/env bash
cd datasets/
wget https://www.albert.cm/projects/ram_person_id/data/DPI-T_train_depth_map.tgz
wget https://www.albert.cm/projects/ram_person_id/data/DPI-T_test_depth_map.tgz
echo 'Expanding tar archive (1 of 2): DPI-T_train_depth_map.tgz'
tar -zxf DPI-T_train_depth_map.tgz
echo 'Expanding tar archive (2 of 2): DPI-T_test_depth_map.tgz'
tar -zxf DPI-T_test_depth_map.tgz
cd ..
echo 'Done'
