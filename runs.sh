# training

echo python train.py --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher
python train.py --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher

echo python train.py --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg
python train.py --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg

# evaluation

echo python test.py --saved_model checkpoints/pretrained/shapenet_car_iter-300000_netG.pkl --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher
python test.py --saved_model checkpoints/pretrained/shapenet_car_iter-300000_netG.pkl --dataset shapenet_car --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher

echo python test.py --saved_model checkpoints/pretrained/shapenet_indoor_iter-300000_netG.pkl --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg
python test.py --saved_model checkpoints/pretrained/shapenet_indoor_iter-300000_netG.pkl --dataset shapenet_indoor --shapenet_dir /is/rg/avg/yliao/neural_rendering/data_blender_newbg