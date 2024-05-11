import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torchvision import models
import torch.nn.functional as F
from seed import torch_fix_seed
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse


# モデルの定義
def get_model(layer1_index=-1, layer2_index=-1, layer3_index=-1):
    device = torch.device('cuda:0')

    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    print(model)

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output.clone().detach().cpu().numpy())

    model.layer1[layer1_index].register_forward_hook(hook)
    model.layer2[layer2_index].register_forward_hook(hook)
    model.layer3[layer3_index].register_forward_hook(hook)
    return model, device, outputs


# トレーニングとテストのファイル名を取得
def get_files(path_train='screw/train/good', path_test_root='screw/test'):
    files_train = [os.path.join(path_train, f) for f in os.listdir(path_train) if f.endswith('png')]
    files_train = np.array(sorted(files_train))

    print('len(files_train) =', len(files_train))
    print('files_train[:5] =\n', files_train[:5])
    print()

    types_test = os.listdir(path_test_root)
    types_test = np.array(sorted(types_test))

    files_test = {}

    for type_test in types_test:
        path_test = os.path.join(path_test_root, type_test)
        files_test[type_test] = [os.path.join(path_test, f)
                                 for f in os.listdir(path_test)
                                 if (os.path.isfile(os.path.join(path_test, f)) &
                                     ('.png' in f))]
        files_test[type_test] = np.array(sorted(files_test[type_test]))

        print('len(files_test[%s]) =' % type_test, len(files_test[type_test]))
        print('files_test[%s][:5] =\n' % type_test, files_test[type_test][:5])
        print()
    return files_train, files_test, types_test


# ねじの向きをそろえるための特徴マッチングの準備
def init_feature_matching(files_train):
    # ORBディテクタの初期化
    orb = cv2.ORB_create()

    # ファイルリストから参照画像を読み込み
    ref_img = cv2.imread(files_train[0])[..., ::-1]  # BGR2RGB
    ref_img = cv2.resize(ref_img, (256, 256), interpolation=cv2.INTER_AREA)
    ref_img = ref_img[16:(256 - 16), 16:(256 - 16)]

    # 参照画像のキーポイントとディスクリプタを抽出
    kp1, des1 = orb.detectAndCompute(ref_img, None)

    # マッチャーの初期化
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return orb, kp1, des1, bf


# ねじの向きをそろえる
def align(orb, kp1, des1, bf, img_prep, white_padding):
    # 現在の画像のキーポイントとディスクリプタを抽出
    kp2, des2 = orb.detectAndCompute(img_prep, None)

    # マッチング
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 最良マッチのトップNを使用してホモグラフィを計算
    if len(matches) > 40:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

        # ホモグラフィ行列を計算し、画像を変換
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned_img = cv2.warpPerspective(img_prep, M, (img_prep.shape[1], img_prep.shape[0]),
                                          borderValue=(255, 255, 255) if white_padding else (0, 0, 0))

        return aligned_img
    else:
        # マッチングが不十分な場合、オリジナル画像を使用
        return img_prep


# 学習データから特徴収集
def get_features_train(files_train, device, model, outputs, use_matching=False, white_padding=False):
    outputs.clear()

    if use_matching:
        orb, kp1, des1, bf = init_feature_matching(files_train)
    else:
        orb, kp1, des1, bf = None, None, None, None

    img_train = []
    img_prep_train = []

    for file in tqdm(files_train):
        img = cv2.imread(file)[..., ::-1]  # BGR2RGB
        img_prep = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img_prep = img_prep[16:(256 - 16), 16:(256 - 16)]

        if use_matching:
            img_prep = align(orb, kp1, des1, bf, img_prep, white_padding)

        img_train.append(img)
        img_prep_train.append(img_prep)

        x = img_prep
        x = x / 255
        x = x - np.array([[[0.485, 0.456, 0.406]]])
        x = x / np.array([[[0.229, 0.224, 0.225]]])
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).permute(0, 3, 1, 2)
        x = x.to(device)

        with torch.no_grad():
            _ = model(x)

    img_prep_train = np.stack(img_prep_train)
    f1_train = np.vstack(outputs[0::3])
    f2_train = np.vstack(outputs[1::3])
    f3_train = np.vstack(outputs[2::3])

    print('len(img_train) =', len(img_train))
    print('img_prep_train.shape =', img_prep_train.shape)
    print('f1_train.shape =', f1_train.shape)
    print('f2_train.shape =', f2_train.shape)
    print('f3_train.shape =', f3_train.shape)
    return f1_train, f2_train, f3_train, orb, kp1, des1, bf


# テストデータから特徴収集
def get_features_test(files_test, device, model, types_test, outputs, orb, kp1, des1, bf, use_matching=False,
                      white_padding=False):
    img_test = {}
    img_prep_test = {}
    gt_test = {}

    f1_test = {}
    f2_test = {}
    f3_test = {}

    for type_test in types_test:

        outputs.clear()

        img_test[type_test] = []
        img_prep_test[type_test] = []
        gt_test[type_test] = []

        for file in tqdm(files_test[type_test]):
            img = cv2.imread(file)[..., ::-1]  # BGR2RGB
            img_prep = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img_prep = img_prep[16:(256 - 16), 16:(256 - 16)]

            if use_matching:
                img_prep = align(orb, kp1, des1, bf, img_prep, white_padding)

            img_test[type_test].append(img)
            img_prep_test[type_test].append(img_prep)

            if (type_test == 'good'):
                gt = np.zeros_like(img_prep[..., 0], dtype=np.uint8)
            else:
                file_gt = file.replace('/test/', '/ground_truth/')
                file_gt = file_gt.replace('.png', '_mask.png')
                gt = cv2.imread(file_gt, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
                gt = gt[16:(256 - 16), 16:(256 - 16)]
                gt = (gt / np.max(gt)).astype(np.uint8)
            gt_test[type_test].append(gt)

            x = img_prep
            x = x / 255
            x = x - np.array([[[0.485, 0.456, 0.406]]])
            x = x / np.array([[[0.229, 0.224, 0.225]]])
            x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).permute(0, 3, 1, 2)
            x = x.to(device)

            with torch.no_grad():
                _ = model(x)

        img_prep_test[type_test] = np.stack(img_prep_test[type_test])
        gt_test[type_test] = np.stack(gt_test[type_test])
        f1_test[type_test] = np.vstack(outputs[0::3])
        f2_test[type_test] = np.vstack(outputs[1::3])
        f3_test[type_test] = np.vstack(outputs[2::3])

        print('len(img_test[%s]) =' % type_test, len(img_test[type_test]))
        print('img_prep_test[%s].shape =' % type_test, img_prep_test[type_test].shape)
        print('gt_test[%s].shape =' % type_test, gt_test[type_test].shape)
        print('f1_test[%s].shape =' % type_test, f1_test[type_test].shape)
        print('f2_test[%s].shape =' % type_test, f2_test[type_test].shape)
        print('f3_test[%s].shape =' % type_test, f3_test[type_test].shape)
        print()
    return f1_test, f2_test, f3_test


# ランダムチョイスによる特徴次元削減
def get_features_random_choice(f1_train, f2_train, f3_train, f1_test, f2_test, f3_test, types_test):
    idx_tmp = np.sort(np.random.permutation(np.arange(256 + 512 + 1024))[:550])

    f1_train = f1_train[:, idx_tmp[idx_tmp < 256]]
    f2_train = f2_train[:, (idx_tmp[(256 <= idx_tmp) & (idx_tmp < (256 + 512))] - 256)]
    f3_train = f3_train[:, (idx_tmp[(256 + 512) <= idx_tmp] - (256 + 512))]

    print('f1_train.shape =', f1_train.shape)
    print('f2_train.shape =', f2_train.shape)
    print('f3_train.shape =', f3_train.shape)

    for type_test in types_test:
        f1_test[type_test] = f1_test[type_test][:, idx_tmp[idx_tmp < 256]]
        f2_test[type_test] = f2_test[type_test][:, (idx_tmp[(256 <= idx_tmp) & (idx_tmp < (256 + 512))] - 256)]
        f3_test[type_test] = f3_test[type_test][:, (idx_tmp[(256 + 512) <= idx_tmp] - (256 + 512))]

        print('f1_test[%s].shape =' % type_test, f1_test[type_test].shape)
        print('f2_test[%s].shape =' % type_test, f2_test[type_test].shape)
        print('f3_test[%s].shape =' % type_test, f3_test[type_test].shape)
        print()
    return f1_train, f2_train, f3_train, f1_test, f2_test, f3_test


# 各層のアクティベーションマップを縦連結
def get_features_concat(f1_train, f2_train, f3_train, f1_test, f2_test, f3_test, types_test):
    f1_train = F.interpolate(torch.from_numpy(f1_train), size=14,
                             mode='bilinear', align_corners=False).numpy()
    f2_train = F.interpolate(torch.from_numpy(f2_train), size=14,
                             mode='bilinear', align_corners=False).numpy()

    print('f1_train.shape =', f1_train.shape)
    print('f2_train.shape =', f2_train.shape)
    print('f3_train.shape =', f3_train.shape)

    f123_train = np.concatenate([f1_train, f2_train, f3_train], axis=1)

    print('f123_train.shape =', f123_train.shape)
    print()

    f123_test = {}

    for type_test in types_test:
        f1_test[type_test] = F.interpolate(torch.from_numpy(f1_test[type_test]),
                                           size=14, mode='bilinear',
                                           align_corners=False).numpy()
        f2_test[type_test] = F.interpolate(torch.from_numpy(f2_test[type_test]),
                                           size=14, mode='bilinear',
                                           align_corners=False).numpy()

        print('f2_test[%s].shape =' % type_test, f2_test[type_test].shape)
        print('f3_test[%s].shape =' % type_test, f3_test[type_test].shape)

        f123_test[type_test] = np.concatenate([f1_test[type_test], f2_test[type_test],
                                               f3_test[type_test]], axis=1)

        print('f123_test[%s].shape =' % type_test, f123_test[type_test].shape)
        print()
    return f123_train, f123_test


# マハラノビス距離の計算
def get_mahalanobis_distance(f123_train, f123_test, types_test):
    # 標本平均と標本共分散を算出
    cov_inv = np.zeros([f123_train.shape[2], f123_train.shape[3], 550, 550])
    mean = np.zeros([f123_train.shape[2], f123_train.shape[3], 550])

    for i_h in tqdm(range(f123_train.shape[2])):
        for i_w in range(f123_train.shape[3]):
            f = f123_train[:, :, i_h, i_w].copy()
            mean[i_h, i_w] = np.mean(f, axis=0)

            f = f - mean[i_h, i_w][None]
            f = ((f.T @ f) / (len(f) - 1)) + (0.01 * np.eye(f.shape[1]))

            cov_inv[i_h, i_w] = np.linalg.inv(f)

    score_test = {}

    for type_test in types_test:

        score_test[type_test] = np.zeros([len(f123_test[type_test]),
                                          f123_train.shape[2], f123_train.shape[3]])

        for i_h in tqdm(range(f123_train.shape[2])):
            for i_w in range(f123_train.shape[3]):
                f = f123_test[type_test][:, :, i_h, i_w]
                score_tmp = [mahalanobis(sample, mean[i_h, i_w], cov_inv[i_h, i_w])
                             for sample in f]
                score_test[type_test][:, i_h, i_w] = np.array(score_tmp)

        print('score_test[%s].shape =' % type_test, score_test[type_test].shape)
        print('np.mean(score_test[%s]) =' % type_test, np.mean(score_test[type_test]))
        print('np.mean(np.abs(score_test[%s])) =' % type_test, np.mean(np.abs(score_test[type_test])))
        print('np.std(score_test[%s]) =' % type_test, np.std(score_test[type_test]))
        print('np.max(score_test[%s]) =' % type_test, np.max(score_test[type_test]))
        print('np.min(score_test[%s]) =' % type_test, np.min(score_test[type_test]))
        print()
    return score_test


# テストデータについて、画像単位異常検知の予測分布可視化と精度算出
def get_anomaly_detection(score_test, types_test, save_dir):
    y_hat_list = []
    y_list = []
    N_test = 0

    type_test = 'good'

    plt.figure(figsize=(10, 8), dpi=100, facecolor='white')

    plt.subplot(2, 1, 1)
    plt.scatter((np.arange(len(score_test[type_test])) + N_test),
                np.max(np.max(score_test[type_test], axis=-1), axis=-1),
                alpha=0.5, label=type_test)
    plt.xlabel('Sample Index')
    plt.ylabel('Max Score')

    plt.subplot(2, 1, 2)
    plt.hist(np.max(np.max(score_test[type_test], axis=-1), axis=-1),
             alpha=0.5, bins=10, label=type_test)
    plt.xlabel('Max Score')
    plt.ylabel('Frequency')

    y_hat_list.append(np.max(np.max(score_test[type_test], axis=-1), axis=-1))
    y_list.append(np.zeros([len(score_test[type_test])], dtype=np.int16))
    N_test += len(score_test[type_test])

    for type_test in types_test[types_test != 'good']:
        plt.subplot(2, 1, 1)
        plt.scatter((np.arange(len(score_test[type_test])) + N_test),
                    np.max(np.max(score_test[type_test], axis=-1), axis=-1),
                    alpha=0.5, label=type_test)

        plt.subplot(2, 1, 2)
        plt.hist(np.max(np.max(score_test[type_test], axis=-1), axis=-1),
                 alpha=0.5, bins=10, label=type_test)

        y_hat_list.append(np.max(np.max(score_test[type_test], axis=-1), axis=-1))
        y_list.append(np.ones([len(score_test[type_test])], dtype=np.int16))
        N_test += len(score_test[type_test])

    y_list = np.hstack(y_list)
    y_hat_list = np.hstack(y_hat_list)

    plt.subplot(2, 1, 1)
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'scatter_hist.png'))
    plt.close()

    # Calculate per-image level ROCAUC
    fpr, tpr, _ = roc_curve(y_list, y_hat_list)
    per_image_rocauc = roc_auc_score(y_list, y_hat_list)

    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % ('screw', per_image_rocauc))
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'rocauc_curve.png'))
    plt.close()
    return per_image_rocauc


def main(use_matching, white_padding, save_dir):
    torch_fix_seed(0)

    model, device, outputs = get_model(layer1_index=2, layer2_index=3, layer3_index=3)

    files_train, files_test, types_test = get_files()

    f1_train, f2_train, f3_train, orb, kp1, des1, bf = get_features_train(files_train, device, model, outputs,
                                                                          use_matching, white_padding)

    f1_test, f2_test, f3_test = get_features_test(files_test, device, model, types_test, outputs, orb, kp1, des1, bf,
                                                  use_matching, white_padding)

    f1_train, f2_train, f3_train, f1_test, f2_test, f3_test = get_features_random_choice(f1_train, f2_train, f3_train,
                                                                                         f1_test, f2_test, f3_test,
                                                                                         types_test)

    f123_train, f123_test = get_features_concat(f1_train, f2_train, f3_train, f1_test, f2_test, f3_test, types_test)

    score_test = get_mahalanobis_distance(f123_train, f123_test, types_test)

    get_anomaly_detection(score_test, types_test, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('--use_matching', type=bool, default=False)
    parser.add_argument('--white_padding', type=bool, default=False)
    args = parser.parse_args()
    if args.use_matching and args.white_padding:
        save_dir = 'results/align_white'
    elif args.use_matching:
        save_dir = 'results/align'
    else:
        save_dir = 'results/original'

    main(args.use_matching, args.white_padding, save_dir)
