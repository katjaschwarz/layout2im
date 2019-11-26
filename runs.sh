# training

echo python train.py --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car1_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car2_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car3_bg1
python train.py --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car1_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car2_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car3_bg1

echo python train.py --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor1_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor2_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor3_bg2
python train.py --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor1_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor2_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor3_bg2

# evaluation

echo python test.py --saved_model checkpoints/pretrained/shapenet_car_iter-300000_netG.pkl --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car1_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car2_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car3_bg1
python test.py --saved_model checkpoints/pretrained/shapenet_car_iter-300000_netG.pkl --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car1_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car2_bg1;/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car3_bg1

echo python test.py --saved_model checkpoints/pretrained/shapenet_indoor_iter-300000_netG.pkl --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor1_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor2_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor3_bg2
python test.py --saved_model checkpoints/pretrained/shapenet_indoor_iter-300000_netG.pkl --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor1_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor2_bg2;/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor3_bg2