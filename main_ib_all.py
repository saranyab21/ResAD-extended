"""
This script is used for test 2, 4, 6, and 8 shot settings simultaneously. 
You need to extract reference features with 8-shot samples in the ./data/8shot directory.
"""
import os
import warnings
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from train import train2
from validate import validate
from datasets.mvtec import MVTEC, MVTECANO
from datasets.visa import VISA, VISAANO
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS

from models.fc_flow import load_flow_model
from models.imagebind import ImageBindModel
from models.modules import MultiScaleOrthogonalProjector
from models.vq import MultiScaleVQ4
from utils import init_seeds, get_residual_features, get_mc_matched_ref_features, get_random_normal_images
from utils import BoundaryAverager
from losses.loss import calculate_log_barrier_bi_occ_loss, calculate_orthogonal_regularizer
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS

warnings.filterwarnings('ignore')

TOTAL_SHOT = 8  # total few-shot reference samples
FIRST_STAGE_EPOCH = 1
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS}


def main(args):
    if args.setting in SETTINGS.keys():
        CLASSES = SETTINGS[args.setting]
    else:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")
    if CLASSES['seen'][0] in MVTEC.CLASS_NAMES:  # from mvtec to other datasets
        train_dataset1 = MVTEC(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                           normalize="imagebind",
                           img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = MVTECANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                normalize='imagebind',
                                img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
    else:  # from visa to mvtec
        train_dataset1 = VISA(args.train_dataset_dir, class_name=CLASSES['seen'], train=True,
                               normalize="imagebind",
                               img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader1 = DataLoader(
            train_dataset1, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        train_dataset2 = VISAANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, 
                                 normalize="imagebind",
                                 img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
        train_loader2 = DataLoader(
            train_dataset2, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )

    encoder = ImageBindModel(device=args.device)
    encoder.to(args.device)
    feat_dims = [1280, 1280, 1280, 1280]
    
    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)
    vq_ops = MultiScaleVQ4(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
    optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(optimizer_vq, milestones=[70, 90], gamma=0.1)
    
    constraintor = MultiScaleOrthogonalProjector(feat_dims).to(args.device)
    # weight_decay is the l2 weight penalty lambda, weight_decay = lambda / 2
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[70, 90], gamma=0.1)
    
    # Normflow decoder
    estimators = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    estimators = [estimator.to(args.device) for estimator in estimators]
    params = list(estimators[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(estimators[l].parameters())
    optimizer1 = torch.optim.Adam(params, lr=args.lr, weight_decay=0.0005)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[70, 90], gamma=0.1)
    
    best_pro = 0
    N_batch = 16 * 16 * 32
    for epoch in range(args.epochs):
        vq_ops.train()
        constraintor.train()
        for estimator in estimators:
            estimator.train()
        if epoch < FIRST_STAGE_EPOCH:
            train_loader = train_loader1
        else:
            train_loader = train_loader2
        train_loss_total_vq, total_num_vq = 0, 0
        train_loss_total_occ, total_num_occ = 0, 0
        train_loss_total_occn, total_num_occn = 0, 0
        train_loss_total_occa, total_num_occa = 0, 0
        train_loss_total_ort, total_num_ort = 0, 0
        train_loss_total_flow, total_num_flow = 0, 0
        progress_bar = tqdm(total=len(train_loader))
        progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}]")
        for step, batch in enumerate(train_loader):
            progress_bar.update(1)
            images, _, masks, class_names = batch
            
            images = images.to(args.device)
            masks = masks.to(args.device)
            
            with torch.no_grad():
                features = encoder.encode_image_from_tensors(images)
                for i in range(len(features)):
                    b, l, c = features[i].shape
                    features[i] = features[i].permute(0, 2, 1).reshape(b, c, 16, 16)
            
            ref_features = get_mc_reference_features(encoder, args.train_dataset_dir, class_names, images.device, args.train_ref_shot)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures)
            
            lvl_masks = []
            for l in range(args.feature_levels):
                _, _, h, w = rfeatures[l].size()
                #m = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
                m = F.interpolate(masks, size=(h, w), mode='bilinear').squeeze(1)
                m = (m > 0.3).to(torch.float32)
                lvl_masks.append(m)
            rfeatures_t = [rfeature.detach().clone() for rfeature in rfeatures]
            
            loss_vq = vq_ops(rfeatures, lvl_masks, train=True)
            train_loss_total_vq += loss_vq.item()
            total_num_vq += 1
            optimizer_vq.zero_grad()
            loss_vq.backward()
            optimizer_vq.step()
            
            rfeatures = constraintor(*rfeatures)
            loss = 0
            for l in range(args.feature_levels):  # backward svdd loss
                e = rfeatures[l]  
                t = rfeatures_t[l]
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                t = t.permute(0, 2, 3, 1).reshape(-1, dim)
                m = lvl_masks[l]
                m = m.reshape(-1)
                
                loss_occ, loss_occn, loss_occa = calculate_log_barrier_bi_occ_loss(e, m, t)
                loss_ort = calculate_orthogonal_regularizer(e, m)
                loss_l = loss_occ + loss_ort
                loss += loss_l
                
                train_loss_total_occ += loss_occ.item()
                total_num_occ += 1
                train_loss_total_occn += loss_occn
                total_num_occn += 1
                train_loss_total_occa += loss_occa
                total_num_occa += 1
                train_loss_total_ort += loss_ort.item()
                total_num_ort += 1
            optimizer0.zero_grad()
            loss.backward()
            optimizer0.step()
            
            # detach the rfeatures for flow optimization
            rfeatures = [rfeature.detach().clone() for rfeature in rfeatures]
            # train flow corresponding to with neck
            loss, num = train2(args, rfeatures, estimators, optimizer1, lvl_masks, boundary_ops, epoch, N_batch=N_batch, FIRST_STAGE_EPOCH=FIRST_STAGE_EPOCH)
            train_loss_total_flow += loss
            total_num_flow += num
        
        scheduler_vq.step()
        scheduler0.step()
        scheduler1.step()
               
        progress_bar.close()
        print(f"Epoch[{epoch}/{args.epochs}]: VQ loss: {train_loss_total_vq / total_num_vq}, OCC loss: {train_loss_total_occ / total_num_occ} (n: {train_loss_total_occn / total_num_occn}, a: {train_loss_total_occa / total_num_occa}), " \
              f"Ort loss: {train_loss_total_ort / total_num_ort}, " \
              f"Flow loss: {train_loss_total_flow / total_num_flow}")
        
        if (epoch + 1) % args.eval_freq == 0:
            for num_shot in [2, 4, 6, 8]:
                s1_res, s2_res, s_res = [], [], []
                test_proto_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, num_shot)
                for class_name in CLASSES['unseen']:
                    if class_name in MVTEC.CLASS_NAMES:
                        test_dataset = MVTEC(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    elif class_name in VISA.CLASS_NAMES:
                        test_dataset = VISA(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    elif class_name in BTAD.CLASS_NAMES:
                        test_dataset = BTAD(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    elif class_name in MVTEC3D.CLASS_NAMES:
                        test_dataset = MVTEC3D(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    elif class_name in MPDD.CLASS_NAMES:
                        test_dataset = MPDD(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    elif class_name in MVTECLOCO.CLASS_NAMES:
                        test_dataset = MVTECLOCO(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    elif class_name in BRATS.CLASS_NAMES:
                        test_dataset = BRATS(args.test_dataset_dir, class_name=class_name, train=False,
                                            normalize='imagebind',
                                            img_size=224, crp_size=224, msk_size=224, msk_crp_size=224)
                    else:
                        raise ValueError('Unrecognized class name: {}'.format(class_name))
                    test_loader = DataLoader(
                        test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
                    )
                    metrics = validate(args, encoder, vq_ops, constraintor, estimators, test_loader, test_proto_features[class_name], args.device, class_name)
                    img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = metrics['scores']
                    
                    print("Epoch: {}, Class Name: {}, Image AUC | AP | F1_Score: {} | {} | {}, Pixel AUC | AP | F1_Score | AUPRO: {} | {} | {} | {}".format(
                        epoch, class_name, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
                    s1_res.append(metrics['scores1'])
                    s2_res.append(metrics['scores2'])
                    s_res.append(metrics['scores'])
            
                s1_res = np.array(s1_res)
                s2_res = np.array(s2_res)
                s_res = np.array(s_res)
                img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1 = np.mean(s1_res, axis=0)
                img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2 = np.mean(s2_res, axis=0)
                img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro = np.mean(s_res, axis=0)
                print('{}, (Logps) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                    num_shot, img_auc1, img_ap1, img_f1_score1, pix_auc1, pix_ap1, pix_f1_score1, pix_aupro1))
                print('{}, (BScores) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                    num_shot, img_auc2, img_ap2, img_f1_score2, pix_auc2, pix_ap2, pix_f1_score2, pix_aupro2))
                print('{}, (Merged) Average Image AUC | AP | F1_Score: {:.3f} | {:.3f} | {:.3f}, Average Pixel AUC | AP | F1_Score | AUPRO: {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(
                    num_shot, img_auc, img_ap, img_f1_score, pix_auc, pix_ap, pix_f1_score, pix_aupro))
            
            if pix_aupro > best_pro:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                best_pro = pix_aupro
                state_dict = {'vq_ops': vq_ops.state_dict(),
                              'constraintor': constraintor.state_dict(),
                              'estimators': [estimator.state_dict() for estimator in estimators]}
                torch.save(state_dict, os.path.join(args.checkpoint_path, f'{args.setting}_checkpoints.pth'))


def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))
        layer4_refs = np.load(os.path.join(root_dir, class_name, 'layer4.npy'))
        
        layer1_refs = torch.from_numpy(layer1_refs).to(device)
        layer2_refs = torch.from_numpy(layer2_refs).to(device)
        layer3_refs = torch.from_numpy(layer3_refs).to(device)
        layer4_refs = torch.from_numpy(layer4_refs).to(device)
        
        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer1_refs = layer1_refs[:K1, :]
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer2_refs = layer2_refs[:K2, :]
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer3_refs = layer3_refs[:K3, :]
        K4 = (layer4_refs.shape[0] // TOTAL_SHOT) * num_shot
        layer4_refs = layer4_refs[:K4, :]
        
        refs[class_name] = (layer1_refs, layer2_refs, layer3_refs, layer4_refs)
    
    return refs
    
    
def get_mc_reference_features(encoder, root, class_names, device, num_shot=4):
    reference_features = {}
    class_names = np.unique(class_names)
    for class_name in class_names:
        normal_paths = get_random_normal_images(root, class_name, num_shot)
        images = load_and_transform_vision_data(normal_paths, device)
        with torch.no_grad():
            features = encoder.encode_image_from_tensors(images.to(device))
            for l in range(len(features)):
                _, _, c = features[l].shape
                features[l] = features[l].reshape(-1, c)
            reference_features[class_name] = features
    return reference_features


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = T.Compose([
                T.Resize(224, T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)
                    
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="visa_to_mvtec")
    parser.add_argument('--train_dataset_dir', type=str, default="")
    parser.add_argument('--test_dataset_dir', type=str, default="")
    parser.add_argument('--test_ref_feature_dir', type=str, default="./ref_features/ib/mvtec_8shot")
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="imagebind")
    
    # flow parameters
    parser.add_argument('--flow_arch', type=str, default='flow_model')
    parser.add_argument('--feature_levels', default=4, type=int)
    parser.add_argument('--coupling_layers', type=int, default=4)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--pos_beta', type=float, default=0.05)
    parser.add_argument('--margin_tau', type=float, default=0.1)
    parser.add_argument('--bgspp_lambda', type=float, default=1)
    
    parser.add_argument('--fdm_alpha', type=float, default=0.4)  # low value, more training distribution
    parser.add_argument('--num_embeddings', type=int, default=1536)  # VQ embeddings
    parser.add_argument("--train_ref_shot", type=int, default=4)
    
    args = parser.parse_args()
    init_seeds(42)
    
    # enable tf32 for accelating
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main(args)
    

    
    
            
