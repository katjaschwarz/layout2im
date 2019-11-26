import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import pickle
import PIL

from utils.data import Resize

from torch.utils.data import DataLoader


class ShapenetSceneGraphDataset(Dataset):
    def __init__(self, root_dir, dset_id, split, image_size=(64, 64), normalize_images=True):
        """
        A PyTorch Dataset for loading Indoor ShapeNet scenes with instance labels, bounding boxes, object masks.

        Inputs:
        - root_dir: Path to a directory where scenes are held
        - dset_id: "car" or "indoor" for Car Dataset and Flying Chairs Indoor Dataset, respectively
        - split: train, test or val
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - normalize_image: If True then normalize images by subtracting training dataset
          mean pixel and dividing by training dataset std pixel.
        """
        super(Dataset, self).__init__()
        self.dset_id = dset_id
        self.split = split
        self.image_size = image_size

        dirnames = root_dir.split(';')
        self.data_dirs = [os.path.join(root_dir, dirname, self.split) for dirname in dirnames]
        self.data_dirs = list(filter(lambda x: os.path.isdir(x), self.data_dirs))
    
        self.num_objects = max([self._get_num_objects(dirname) for dirname in dirnames])
    
        self.normalize_images = normalize_images
        if normalize_images:
            mean, std = self._get_mean_var()
            self.normalize = T.Normalize(mean=mean/255., std=std/255.)
    
        self.set_image_size()
        self._get_filenames()
        
    def set_image_size(self):
        print('called set_image_size', self.image_size)
        transform = [Resize(self.image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(self.normalize)
        self.transform = T.Compose(transform)

    def _get_filenames(self):
        self.filenames = []
        for data_dir in self.data_dirs:
            scenes = filter(lambda x: os.path.isdir(os.path.join(data_dir, x)), os.listdir(data_dir))
            scenes = sorted(scenes)
            
            for s in tqdm(scenes, 'Load filenames...'):
                filenames = filter(lambda x: x.endswith('.png'), os.listdir(os.path.join(data_dir, s)))
                filenames = map(lambda x: os.path.join(data_dir, s, x), filenames)
                self.filenames.extend(sorted(filenames))
    
    def _get_num_objects(self, dirname):
        return int(dirname.split(f'{self.dset_id}')[1].split('_bg')[0])
    
    @staticmethod
    def _get_idx(filename):
        basename = os.path.basename(filename)
        return int(os.path.splitext(basename)[0].split('_')[-1])
    
    def _bbox_to_mask(self, x0, y0, x1, y1):
        H, W = self.image_size
        mask = torch.zeros(1, H, W)
        mask[:, round((y0 * H).item()):max(round((y0 * H).item()) + 1, round((y1 * H).item())),
        round((x0 * W).item()):max(round((x0 * W).item()) + 1, round((x1 * W).item()))] = 1
        return mask
    
    def _get_mean_var(self):
        for data_dir in self.data_dirs:
            root_dir, dirname = os.path.split(os.path.split(data_dir)[0])
            meanfile = os.path.join(root_dir, f'{self.dset_id}_dataset_mean_var_{dirname}.pkl')
           
            if not os.path.isfile(meanfile):
                dset = ShapenetSceneGraphDataset(root_dir=os.path.join(root_dir, dirname),
                                                 dset_id=self.dset_id, split='train',
                                                 image_size=self.image_size, normalize_images=False)
                mean = []
                stddev = []
                
                for imfile in tqdm(dset.filenames, f'Compute mean and variance for {dirname}'):
                    with open(imfile, 'rb') as f:
                        with PIL.Image.open(f) as image:
                            image = image.convert('RGB')
                            stat = PIL.ImageStat.Stat(image)
                            mean.append(stat.mean)
                            stddev.append(stat.stddev)
                mean = np.stack(mean).mean(axis=0)
                stddev = np.stack(stddev).mean(axis=0)
                with open(meanfile, 'wb') as f:
                    pickle.dump({'mean': mean, 'stddev':stddev, 'n_img': len(dset)}, f)
                    
        weights, means, stddevs = [], [], []
        for data_dir in self.data_dirs:
            root_dir, dirname = os.path.split(os.path.split(data_dir)[0])
            meanfile = os.path.join(root_dir, f'{self.dset_id}_dataset_mean_var_{dirname}.pkl')

            with open(meanfile, 'rb') as f:
                data = pickle.load(f)
                weights.append(data['n_img'])
                means.append(data['mean'])
                stddevs.append(data['stddev'])

        weights = [w/sum(weights) for w in weights]
        mean = np.sum(np.stack([w * np.array(m) for w, m in zip(weights, means)]), axis=0)
        stddev = np.sum(np.stack([w * np.array(sd) for w, sd in zip(weights, stddevs)]), axis=0)

        return mean, stddev

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        view_idx = self._get_idx(filename)
        
        with open(filename, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))
        
        info_path = os.path.join(os.path.dirname(filename), 'bboxes.npz')
        info = np.load(info_path)
       
        class_id = torch.LongTensor(info['class_id'][view_idx])
        n_obj = len(class_id)
        
        bbox = info['bbox'][view_idx]
        x, y, w, h = bbox.transpose()
        x0 = x / WW
        y0 = y / HH
        x1 = (x + w) / WW
        y1 = (y + h) / HH
        bbox = torch.FloatTensor([x0, y0, x1, y1]).transpose(0,1)
        bbox = bbox.clamp(0, 1)         # do not allow boxes out of range
        
        mask = torch.stack(list(map(lambda x: self._bbox_to_mask(*x), bbox)))

        # shuffle objs
        obj_idcs = torch.randperm(n_obj)
        class_id = class_id[obj_idcs]
        bbox = bbox[obj_idcs]
        mask = mask[obj_idcs]
        
        return image, class_id, bbox, mask


def shapenet_collate_fn(batch):
    """
    Collate function to be used when wrapping IndoorSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    """
    all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img = [], [], [], [], []

    for i, (img, objs, boxes, masks) in enumerate(batch):
        all_imgs.append(img[None])
        O = objs.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_obj_to_img = torch.cat(all_obj_to_img)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img)

    return out


def get_dataloader(batch_size=10, dset_id='indoor',
                   DATA_DIR='/is/rg/avg/yliao/neural_rendering/data_blender_newbg/',
                   testset=False):
    image_size = (64, 64)
    batch_size = batch_size
    shuffle_val = True

    # build datasets
    dset_kwargs = {
        'root_dir': DATA_DIR,
        'dset_id': dset_id,
        'split': 'train',
        'image_size': image_size,
        'normalize_images': True,
    }
    train_dset = ShapenetSceneGraphDataset(**dset_kwargs)

    dset_kwargs['split'] = 'val'
    val_dset = ShapenetSceneGraphDataset(**dset_kwargs)

    dset_kwargs['split'] = 'test'
    test_dset = ShapenetSceneGraphDataset(**dset_kwargs)
   
    # build dataloader
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': shapenet_collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = shuffle_val
    loader_kwargs['num_workers'] = 1
    val_loader = DataLoader(val_dset, **loader_kwargs)
    test_loader = DataLoader(test_dset, **loader_kwargs)

    if testset:
        return train_loader, test_loader
    else:
        return train_loader, val_loader
    

if __name__ == '__main__':
    kwargs = {'DATA_DIR': '/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car1_bg1;'
                          '/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car2_bg1;'
                          '/is/rg/avg/yliao/neural_rendering/data_blender_newbg_higher/car3_bg1',
              'dset_id': 'car'
              }

    kwargs = {'DATA_DIR': '/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor1_bg2;'
                          '/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor2_bg2;'
                          '/is/rg/avg/yliao/neural_rendering/data_blender_newbg/indoor3_bg2',
              'dset_id': 'indoor'
              }
    
    train_loader, val_loader = get_dataloader(**kwargs)

    # test reading data
    for i, batch in enumerate(val_loader):
        imgs, objs, boxes, masks, obj_to_img = batch
    
        print(imgs.shape, objs.shape, boxes.shape, masks.shape, obj_to_img.shape)
    
        if i == 20: break