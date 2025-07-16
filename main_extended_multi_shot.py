import os
import warnings
import argparse
from tqdm import tqdm
import numpy as np
import torch
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

from train import train
from validate import validate
from datasets.mvtec import MVTEC, MVTECANO
from datasets.visa import VISA, VISAANO
from datasets.btad import BTAD
from datasets.mvtec_3d import MVTEC3D
from datasets.mpdd import MPDD
from datasets.mvtec_loco import MVTECLOCO
from datasets.brats import BRATS

from models.fc_flow import load_flow_model
from models.modules import MultiScaleConv
from models.vq import MultiScaleVQ
from utils import init_seeds, get_residual_features, get_mc_matched_ref_features, get_mc_reference_features
from utils import BoundaryAverager
from losses.loss import calculate_log_barrier_bi_occ_loss
from classes import VISA_TO_MVTEC, MVTEC_TO_VISA, MVTEC_TO_BTAD, MVTEC_TO_MVTEC3D
from classes import MVTEC_TO_MPDD, MVTEC_TO_MVTECLOCO, MVTEC_TO_BRATS

warnings.filterwarnings('ignore')

TOTAL_SHOT = 8
FIRST_STAGE_EPOCH = 10
SETTINGS = {'visa_to_mvtec': VISA_TO_MVTEC, 'mvtec_to_visa': MVTEC_TO_VISA,
            'mvtec_to_btad': MVTEC_TO_BTAD, 'mvtec_to_mvtec3d': MVTEC_TO_MVTEC3D,
            'mvtec_to_mpdd': MVTEC_TO_MPDD, 'mvtec_to_mvtecloco': MVTEC_TO_MVTECLOCO,
            'mvtec_to_brats': MVTEC_TO_BRATS}

def save_metrics_csv_and_plot(metrics_list, save_dir, setting, epoch, shot):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(metrics_list)
    csv_path = os.path.join(save_dir, f"{setting}_{shot}shot_epoch{epoch}_metrics.csv")
    df.to_csv(csv_path, index=False)

    value_keys = [key for key in df.columns if key != "Class Name"]
    for key in value_keys:
        plt.figure(figsize=(10, 5))
        df_sorted = df.sort_values(by=key, ascending=False)
        plt.bar(df_sorted['Class Name'], df_sorted[key], color='royalblue')
        plt.title(f"{key} per Class - {shot}-Shot - Epoch {epoch}")
        plt.xlabel("Class Name")
        plt.ylabel(key)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend([key], loc='best')
        plot_path = os.path.join(save_dir, f"{setting}_{shot}shot_epoch{epoch}_{key.replace(' ', '_').lower()}.png")
        plt.savefig(plot_path)
        plt.close()

def load_mc_reference_features(root_dir: str, class_names, device: torch.device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = np.load(os.path.join(root_dir, class_name, 'layer1.npy'))
        layer2_refs = np.load(os.path.join(root_dir, class_name, 'layer2.npy'))
        layer3_refs = np.load(os.path.join(root_dir, class_name, 'layer3.npy'))

        layer1_refs = torch.from_numpy(layer1_refs).to(device)
        layer2_refs = torch.from_numpy(layer2_refs).to(device)
        layer3_refs = torch.from_numpy(layer3_refs).to(device)

        K1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        K2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        K3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot

        refs[class_name] = (
            layer1_refs[:K1, :],
            layer2_refs[:K2, :],
            layer3_refs[:K3, :]
        )
    return refs

def main(args):
    if args.setting not in SETTINGS:
        raise ValueError(f"Invalid setting. Choose from {list(SETTINGS.keys())}")
    CLASSES = SETTINGS[args.setting]

    if CLASSES['seen'][0] in MVTEC.CLASS_NAMES:
        train_loader1 = DataLoader(MVTEC(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224),
                                   batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        train_loader2 = DataLoader(MVTECANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize='w50', img_size=224, crp_size=224, msk_size=224, msk_crp_size=224),
                                   batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    else:
        train_loader1 = DataLoader(VISA(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224),
                                   batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        train_loader2 = DataLoader(VISAANO(args.train_dataset_dir, class_name=CLASSES['seen'], train=True, normalize="w50", img_size=224, crp_size=224, msk_size=224, msk_crp_size=224),
                                   batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    encoder = timm.create_model(args.backbone, features_only=True, out_indices=(1, 2, 3), pretrained=True).eval().to(args.device)
    feat_dims = encoder.feature_info.channels()
    vq_ops = MultiScaleVQ(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)
    constraintor = MultiScaleConv(feat_dims).to(args.device)
    estimators = [load_flow_model(args, d).to(args.device) for d in feat_dims]
    boundary_ops = BoundaryAverager(num_levels=args.feature_levels)

    optimizer_vq = torch.optim.Adam(vq_ops.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer0 = torch.optim.Adam(constraintor.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer1 = torch.optim.Adam([p for e in estimators for p in e.parameters()], lr=args.lr, weight_decay=0.0005)

    scheduler_vq = torch.optim.lr_scheduler.MultiStepLR(optimizer_vq, milestones=[70, 90], gamma=0.1)
    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, milestones=[70, 90], gamma=0.1)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[70, 90], gamma=0.1)

    for epoch in range(args.epochs):
        vq_ops.train(), constraintor.train(), [e.train() for e in estimators]
        train_loader = train_loader1 if epoch < FIRST_STAGE_EPOCH else train_loader2
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, _, masks, class_names = [x.to(args.device) if isinstance(x, torch.Tensor) else x for x in batch]
            with torch.no_grad():
                features = encoder(images)
            ref_features = get_mc_reference_features(encoder, args.train_dataset_dir, class_names, images.device, args.train_ref_shot)
            mfeatures = get_mc_matched_ref_features(features, class_names, ref_features)
            rfeatures = get_residual_features(features, mfeatures, pos_flag=True)

            lvl_masks = [F.interpolate(masks, size=f.shape[2:], mode='nearest').squeeze(1) for f in rfeatures]
            rfeatures_t = [f.detach().clone() for f in rfeatures]
            loss_vq = vq_ops(rfeatures, lvl_masks, train=True)
            optimizer_vq.zero_grad(); loss_vq.backward(); optimizer_vq.step()

            rfeatures = constraintor(*rfeatures)
            loss_total = 0
            for l in range(args.feature_levels):
                e, t = rfeatures[l], rfeatures_t[l]
                e = e.permute(0, 2, 3, 1).reshape(-1, e.size(1))
                t = t.permute(0, 2, 3, 1).reshape(-1, t.size(1))
                m = lvl_masks[l].reshape(-1)
                loss_i, _, _ = calculate_log_barrier_bi_occ_loss(e, m, t)
                loss_total += loss_i
            optimizer0.zero_grad(); loss_total.backward(); optimizer0.step()

            rfeatures = [f.detach().clone() for f in rfeatures]
            train(args, rfeatures, estimators, optimizer1, masks, boundary_ops, epoch)

        scheduler_vq.step(); scheduler0.step(); scheduler1.step()

        if (epoch + 1) % args.eval_freq == 0:
            for num_shot in [2, 4, 6, 8]:
                test_ref_features = load_mc_reference_features(args.test_ref_feature_dir, CLASSES['unseen'], args.device, num_shot)
                metrics_list = []
                for class_name in CLASSES['unseen']:
                    Dataset = MVTEC if class_name in MVTEC.CLASS_NAMES else VISA
                    dataset = Dataset(args.test_dataset_dir, class_name=class_name, train=False, normalize="w50", img_size=224)
                    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
                    metrics = validate(args, encoder, vq_ops, constraintor, estimators, loader, test_ref_features[class_name], args.device, class_name)
                    scores = metrics["scores"]
                    metrics_list.append({
                        "Class Name": class_name,
                        "Image AUC": scores[0], "Image F1": scores[2],
                        "Pixel AUC": scores[3], "Pixel F1": scores[5],
                        "Pixel AUPRO": scores[6]
                    })
                save_dir = os.path.join(args.checkpoint_path, f"{args.setting}_{num_shot}shot")
                save_metrics_csv_and_plot(metrics_list, save_dir, args.setting, epoch, shot=num_shot)
                # save latest .pth
                checkpoint = {
                    'vq_ops': vq_ops.state_dict(),
                    'constraintor': constraintor.state_dict(),
                    'estimators': [e.state_dict() for e in estimators]
                }
                torch.save(checkpoint, os.path.join(save_dir, f"{args.setting}_{num_shot}shot_latest.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="mvtec_to_visa")
    parser.add_argument('--train_dataset_dir', type=str, required=True)
    parser.add_argument('--test_dataset_dir', type=str, required=True)
    parser.add_argument('--test_ref_feature_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/")
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="wide_resnet50_2")
    parser.add_argument('--flow_arch', type=str, default='conditional_flow_model')
    parser.add_argument('--feature_levels', type=int, default=3)
    parser.add_argument('--coupling_layers', type=int, default=10)
    parser.add_argument('--clamp_alpha', type=float, default=1.9)
    parser.add_argument('--pos_embed_dim', type=int, default=256)
    parser.add_argument('--pos_beta', type=float, default=0.05)
    parser.add_argument('--margin_tau', type=float, default=0.1)
    parser.add_argument('--bgspp_lambda', type=float, default=1)
    parser.add_argument('--fdm_alpha', type=float, default=0.4)
    parser.add_argument('--num_embeddings', type=int, default=1536)
    parser.add_argument('--train_ref_shot', type=int, default=8)
    args = parser.parse_args()
    init_seeds(42)
    main(args)

