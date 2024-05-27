from itertools import product
import pandas as pd
import torch
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision
import glob
from pathlib import Path
import linecache


def read_metric_txt(file, idx):
    ret = linecache.getline(file, idx+2).strip()
    return [float(x) for x in ret.split(',')]
        


class dataset_validate(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self, dataset_type, novel_type, path, time_type): 
        super().__init__()
        if time_type == '10':
            dir = 'iter_1000'
        elif time_type == '120':
            dir = 'better'
        
        if dataset_type == 'h36m':
            if novel_type == 'validate_novel_view':
                subjects = {
                    '1': {
                        'frames': [0, 150, 300, 450, 600],
                        'views': [3] * 5,
                    },
                    '5': {
                        'frames': [0, 150, 300, 450, 600, 750, 900, 1050, 1200],
                        'views': [3] * 9,
                    },
                    '6': {
                        'frames': [0, 150, 300, 450, 600],
                        'views': [3] * 5,
                    },
                    '7': {
                        'frames': [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350],
                        'views': [3] * 10,
                    },
                    '8': {
                        'frames': [0, 150, 300, 450, 600, 750, 900, 1050, 1200],
                        'views': [3] * 9,
                    },
                    '9': {
                        'frames': [0, 150, 300, 450, 600, 750, 900, 1050, 1200],
                        'views': [3] * 9,
                    },
                    '11': {
                        'frames': [0, 150, 300, 450, 600, 750, 900],
                        'views': [3] * 7,
                    },
                }
            elif novel_type == 'validate_novel_pose':
                subjects = {
                    '1': {
                        'frames': [750, 900],
                        'views': [3] * 2,
                    },
                    '5': {
                        'frames': [1250, 1400, 1550, 1700, 1850],
                        'views': [3] * 5,
                    },
                    '6': {
                        'frames': [750, 900, 1050],
                        'views': [3] * 3,
                    },
                    '7': {
                        'frames': [1500, 1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850],
                        'views': [3] * 10,
                    },
                    '8': {
                        'frames': [1250, 1400, 1550],
                        'views': [3] * 3,
                    },
                    '8': {
                        'frames': [1300, 1450, 1600, 1750, 1900],
                        'views': [3] * 5,
                    },
                    '11': {
                        'frames': [1000, 1150, 1300],
                        'views': [3] * 3,
                    },
                }
        elif dataset_type == 'zju_mocap':
            if novel_type == 'validate_novel_view':
                subjects = {
                    '313': {
                        'frames':[0,0, 0, 0,30,30, 30,30],
                        'views': [2,8,14,20,2, 8, 14, 20],
                    },
                    '315': {
                        'frames': sum([[x] * 4 for x in list(range(0, 391,30))], []),
                        'views': [2,8,14,20] * len(range(0, 391,30)),
                    },
                    '377': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                    '386': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                    '387': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                    '390': {
                        'frames': sum([[x] * 4 for x in list(range(700, 971,30))], []),
                        'views': [2,8,14,20] * len(range(700, 971,30)),
                    },
                    '392': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                    '393': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                    '394': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                }
            elif novel_type == 'validate_novel_pose':            
                subjects = {
                    '313': {
                        'frames': sum([[x] * 4 for x in list(range(60, 331,30))], []),
                        'views': [2,8,14,20] * len(range(60, 331,30)),
                    },
                    '315': {
                        'frames': sum([[x] * 4 for x in list(range(400, 671,30))], []),
                        'views': [2,8,14,20] * len(range(400, 671,30)),
                    },
                    '377': {
                        'frames': sum([[x] * 4 for x in list(range(300, 571,30))], []),
                        'views': [2,8,14,20] * len(range(300, 571,30)),
                    },
                    '386': {
                        'frames': sum([[x] * 4 for x in list(range(300, 571,30))], []),
                        'views': [2,8,14,20] * len(range(300, 571,30)),
                    },
                    '387': {
                        'frames': sum([[x] * 4 for x in list(range(300, 571,30))], []),
                        'views': [2,8,14,20] * len(range(300, 571,30)),
                    },
                    '390': {
                        'frames': sum([[x] * 4 for x in list(range(0, 271,30))], []),
                        'views': [2,8,14,20] * len(range(0, 271,30)),
                    },
                    '392': {
                        'frames': sum([[x] * 4 for x in list(range(300, 541,30))], []),
                        'views': [2,8,14,20] * len(range(300, 541,30)),
                    },
                    '393': {
                        'frames': sum([[x] * 4 for x in list(range(300, 571,30))], []),
                        'views': [2,8,14,20] * len(range(300, 571,30)),
                    },
                    '394': {
                        'frames': sum([[x] * 4 for x in list(range(300, 571,30))], []),
                        'views': [2,8,14,20] * len(range(300, 571,30)),
                    },
                }
                
        
        self.filenames = []
        self.psnrs = []
        self.ssims = []
        self.fpss = []
        path = os.path.join(path)
        for subject, item in subjects.items():
            subject_path = os.path.join(path, subject, '*', 'validate_otf', dir, novel_type)
            for frame, view in zip(item['frames'], item['views']):
                opt_file = os.path.join(subject_path, 'opt', f"val-*-{frame}-{view}.png")
                opt_files = glob.glob(str(opt_file))
                
                if len(opt_files) == 0:
                    print("checked", opt_file)
                else:
                    try:
                        if len(opt_files) == 1:
                            opt_files = opt_files[:1]
                        metrics_path = os.path.join(*opt_files[0].split('/')[:-2], "metrics.txt")
                        self.filenames.append(opt_files[0])
                        # print(opt_files)
                        idx = opt_files[0].split('/')[-1].split('-')[1]
                        idx, mse, psnr, ssim, fps = read_metric_txt(metrics_path, int(idx))
                        self.psnrs.append(psnr)
                        self.ssims.append(ssim)
                        self.fpss.append(fps)
                    except:
                        import pdb; pdb.set_trace()
                    

                
        

    def __len__(self):
        return len(self.filenames)
    
    def print(self):
        print(self.filenames)
    
    def __getitem__(self, idx):
        
        # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
        
        # opt = Image.open(self.opt_files[idx])
        # opt_tensor = torchvision.transforms.functional.pil_to_tensor(opt)
        # ref = Image.open(self.ref_files[idx])
        # ref_tensor = torchvision.transforms.functional.pil_to_tensor(ref)
        filename = self.filenames[idx]
        psnr = self.psnrs[idx]
        ssim = self.ssims[idx]
        fps = self.fpss[idx]
        
        return {
            'filename': filename,
            'psnr': psnr,
            'ssim': ssim,
            'fps': fps,
        }


def collect(dataset_validate, out_dir, ):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []
    ssim_values = []
    fps_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1)

    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    df_list = []
    with open(os.path.join(out_dir, 'metrics.csv'), 'w') as fout:
        fout.write('FILE, PSNR, SSIM, FPS\n')

        print(f"Running validation @ {out_dir}")
        for it, result_dict in enumerate(tqdm(dataloader_validate)):
            
            filename = result_dict['filename']
            psnr = result_dict['psnr']
            ssim = result_dict['ssim']
            fps = result_dict['fps']

            # # Compute metrics
            # opt = torch.clamp(result_dict['opt'][0].permute(1,2,0) / 255., 0.0, 1.0)
            # ref = torch.clamp(result_dict['ref'][0].permute(1,2,0) / 255., 0.0, 1.0)


            # mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            # psnr = util.mse_to_psnr(mse)
            # ssim = util.compute_ssim(opt, ref)

            psnr_values.append(float(psnr))
            ssim_values.append(float(ssim))
            fps_values.append(float(fps))

            line = "%s, %1.8f, %1.8f, %.2f\n" % (filename[0].split('/')[3]+filename[0], psnr, ssim, fps)
            fout.write(str(line))
            df_list.append(dict(name=filename[0].split('/')[3], psnr=psnr, ssim=ssim, fps=fps))


        avg_psnr = np.mean(np.array(psnr_values))
        avg_ssim = np.mean(np.array(ssim_values))
        avg_fps = np.mean(np.array(fps_values))
        line = "AVERAGES: %2.6f, %2.6f, %.6f\n" % (avg_psnr, avg_ssim, avg_fps)
        fout.write(str(line))
        print("PSNR,    SSIM,    FPS")
        print("%2.6f, %2.6f, %.6f" % (avg_psnr, avg_ssim, avg_fps))

        df = pd.DataFrame(df_list)
        print(df.groupby("name").mean())
        print(df.groupby("name").mean().mean())

    return {
        # 'mse': avg_mse,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'fps': avg_fps
    }

if __name__ == '__main__':
    
    # dataset_type_list = ['h36m',                'h36m',                'h36m',                'h36m',                'zju_mocap',           'zju_mocap',           'zju_mocap',            'zju_mocap']
    # novel_type_list =   ['validate_novel_view', 'validate_novel_view', 'validate_novel_pose', 'validate_novel_pose', 'validate_novel_view', 'validate_novel_view', 'validate_novel_pose',  'validate_novel_pose']
    # time_list =         ['10',                  '120',                 '10',                  '120',                  '10',                 '120',                 '10',                   '120']
    
    dataset_type_list = ['zju_mocap']
    novel_type_list = ['validate_novel_view', 'validate_novel_pose']
    time_list = ['10']
    
    h36m_path = 'nvdiffrec-human/outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.h36m'
    # zju_path = 'outputs/base_config-1.zju_mocap.logl2.5min/zju_mocap'
    zju_path = 'outputs/base_config-all-smpl_init-meta_skin_net-smpl_surface_skinning_reg-full-loss-finetune_mat-full_subdiv.zju_mocap/zju_mocap'

    
    for dataset_type, novel_type, time in product(dataset_type_list, novel_type_list, time_list):
        if dataset_type == 'h36m':            path = h36m_path
        elif dataset_type == 'zju_mocap':
            path = zju_path
        print(dataset_type, novel_type, path, time)
        data = dataset_validate(dataset_type, novel_type, path, time)
        # print(data[0]['ref'].shape, data[0]['ref'].min(), data[0]['ref'].max())
        collect(data, f"./{dataset_type}/{novel_type}/{time}")
