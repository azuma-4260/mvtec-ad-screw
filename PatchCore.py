import torch
from torchvision import models
from seed import torch_fix_seed
from PaDim import *
from torchinfo import summary

backbones = ['wide_resnet50_2', 'densenet201', 'effficientnet-b5']

# 特徴抽出器の準備
def prepare_extractor(first_layer_index=-1, second_layer_index=-1, backbone='wide_resnet50_2'):
    print('backbone =', backbone)
    assert backbone in backbones, 'invalid backbone'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    if backbone == backbones[0]:
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.layer2[first_layer_index].register_forward_hook(hook)
        model.layer3[second_layer_index].register_forward_hook(hook)

    elif backbone == backbones[1]:
        if first_layer_index == -1:
            first_layer_index = 12
        if second_layer_index == -1:
            second_layer_index = 48
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        block2_layer = getattr(model.features.denseblock2, f'denselayer{first_layer_index}')
        block3_layer = getattr(model.features.denseblock3, f'denselayer{second_layer_index}')
        block2_layer.register_forward_hook(hook)
        block3_layer.register_forward_hook(hook)

    elif backbone == backbones[2]:
        assert -1 <= first_layer_index <= 4, 'invalid first layer index'
        assert -1 <= second_layer_index <= 6, 'invalid second layer index'
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        model.features[3][first_layer_index].register_forward_hook(hook)
        model.features[5][second_layer_index].register_forward_hook(hook)

    summary(model, input_size=(1, 3, 224, 224))

    model.eval()
    model.to(device)

    print('model =', model)

    return model, outputs, device

# 学習データ画像読み込み
def load_train_imgs(files_train):
    img_prep_train = []

    for file in tqdm(files_train):
        img = cv2.imread(file)[..., ::-1]  # BGR2RGB
        img_prep = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img_prep = img_prep[16:(256-16), 16:(256-16)]

        img_prep_train.append(img_prep)

    img_prep_train = np.stack(img_prep_train)

    print('img_prep_train.shape =', img_prep_train.shape)
    return img_prep_train

# 学習データ画像から、特徴抽出
def extract_train_features(device, model, outputs, img_prep_train):
    # set param
    N_batch = 100
    patchsize = 3
    stride = 1
    padding = int((patchsize - 1) / 2)

    unfolder = torch.nn.Unfold(
        kernel_size=patchsize, stride=stride, padding=padding, dilation=1
    )

    outputs.clear()

    with torch.no_grad():
        _ = model(torch.randn(N_batch, 3, 224, 224).to(device))

    f1 = outputs[0].clone()  # (B, C, H, W)
    f2 = outputs[1].clone()  # (B, C, H, W)
    feat = [f1, f2]
    shapes = [f1.shape, f2.shape]

    patch_shapes = []
    for i in range(len(feat)):
        number_of_total_patches = []
        for s in shapes[i][-2:]:
            n_patches = (s + 2 * padding - 1 * (patchsize - 1) - 1) / stride + 1
            number_of_total_patches.append(int(n_patches))
        patch_shapes.append(number_of_total_patches)
    print('patch_shapes =', patch_shapes)

    ref_num_patches = patch_shapes[0]
    print('ref_num_patches =', ref_num_patches)

    pretrain_embed_dimension = 1024
    target_embed_dimension = 1024

    MEAN = torch.from_numpy(np.array([[[0.485, 0.456, 0.406]]]))
    MEAN = MEAN.to(torch.float).to(device)
    STD = torch.from_numpy(np.array([[[0.229, 0.224, 0.225]]]))
    STD = STD.to(torch.float).to(device)

    feat_train = []

    outputs.clear()

    for i_batch in tqdm(range(0, len(img_prep_train), N_batch)):

        img_batch = img_prep_train[i_batch:(i_batch + N_batch)]
        x = torch.from_numpy(img_batch).to(torch.float).to(device)
        x = x / 255
        x = x - MEAN
        x = x / STD
        x = x.permute(0, 3, 1, 2)

        with torch.no_grad():
            _ = model(x)

        f1 = outputs[0].clone()  # (B, C, H, W)
        f2 = outputs[1].clone()  # (B, C, H, W)
        feat = [f1, f2]
        shapes = [f1.shape, f2.shape]

        outputs.clear()

        # patchify
        for i in range(len(feat)):
            # (B, C, H, W) -> (B, C, H, W, PH, PW)
            with torch.no_grad():
                feat[i] = unfolder(feat[i])
            # (B, C, H, W, PH, PW) -> (B, C, PH, PW, HW)
            feat[i] = feat[i].reshape(*shapes[i][:2],
                                            patchsize, patchsize, -1)
            # (B, C, PH, PW, HW) -> (B, HW, C, PW, HW)
            feat[i] = feat[i].permute(0, 4, 1, 2, 3)

        for i in range(1, len(feat)):
            _feat = feat[i]
            patch_dims = patch_shapes[i]
            # (B, HW, C, PW, HW) -> (B, H, W, C, PH, PW)
            _feat = _feat.reshape(_feat.shape[0], patch_dims[0],
                                        patch_dims[1], *_feat.shape[2:])
            # (B, H, W, C, PH, PW) -> (B, C, PH, PW, H, W)
            _feat = _feat.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _feat.shape
            # (B, C, PH, PW, H, W) -> (BCPHPW, H, W)
            _feat = _feat.reshape(-1, *_feat.shape[-2:])
            # (BCPHPW, H, W) -> (BCPHPW, H_max, W_max)
            _feat = F.interpolate(_feat.unsqueeze(1),
                                    size=(ref_num_patches[0], ref_num_patches[1]),
                                    mode="bilinear", align_corners=False)
            _feat = _feat.squeeze(1)
            # (BCPHPW, H_max, W_max) -> (B, C, PH, PW, H_max, W_max)
            _feat = _feat.reshape(*perm_base_shape[:-2],
                                        ref_num_patches[0], ref_num_patches[1])
            # (B, C, PH, PW, H_max, W_max) -> (B, H_max, W_max, C, PH, PW)
            _feat = _feat.permute(0, -2, -1, 1, 2, 3)
            # (B, H_max, W_max, C, PH, PW) -> (B, H_maxW_max, C, PH, PW)
            _feat = _feat.reshape(len(_feat), -1, *_feat.shape[-3:])
            feat[i] = _feat

        # (B, H, W, C, PH, PW) -> (BHW, C, PH, PW)
        feat = [x.reshape(-1, *x.shape[-3:]) for x in feat]

        for i in range(len(feat)):
            _feat = feat[i]
            # (BHW, C, PH, PW) -> (BHW, 1, CPHPW)
            _feat = _feat.reshape(len(_feat), 1, -1)
            # (BHW, 1, CPHPW) -> (BHW, D_p)
            _feat = F.adaptive_avg_pool1d(_feat,
                                            pretrain_embed_dimension).squeeze(1)
            feat[i] = _feat

        # (BHW, D_p) -> (BHW, D_p*2)
        feat = torch.stack(feat, dim=1)
        """Returns reshaped and average pooled feat."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        # (BHW, D_p*2) -> (BHW, D_t)
        feat = feat.reshape(len(feat), 1, -1)
        feat = F.adaptive_avg_pool1d(feat, target_embed_dimension)
        feat = feat.reshape(len(feat), -1)

        feat_train.append(feat.cpu())

    feat_train = torch.vstack(feat_train)

    print('feat_train.shape =', feat_train.shape)
    return feat_train, MEAN, STD, patch_shapes, ref_num_patches, pretrain_embed_dimension, target_embed_dimension, unfolder

# コアセットサンプリング
def core_set_sampling(feat_train, device):
    percentage = 0.1
    dimension_to_project_features_to = 128
    number_of_starting_points = 10

    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    mapper = torch.nn.Linear(feat_train.shape[1], dimension_to_project_features_to,
                            bias=False).to(device)

    print('mapper =', mapper)

    feat_train = feat_train.to(device)

    with torch.no_grad():
        feat_train_proj = mapper(feat_train)

    print('feat_train.shape =', feat_train.shape)
    print('feat_train_proj.shape =', feat_train_proj.shape)

    number_of_starting_points = np.clip(number_of_starting_points,
                                        None, len(feat_train_proj))

    print('number_of_starting_points =', number_of_starting_points)

    np.random.seed(0)
    start_points = np.random.choice(len(feat_train_proj), number_of_starting_points,
                                    replace=False).tolist()

    print('len(start_points) =', len(start_points))
    print('start_points =', start_points)

    matrix_a = feat_train_proj
    matrix_b = feat_train_proj[start_points]

    print('matrix_a.shape =', matrix_a.shape)
    print('matrix_b.shape =', matrix_b.shape)
    print()

    print('matrix_a.unsqueeze(1).shape =', matrix_a.unsqueeze(1).shape)
    print('matrix_a.unsqueeze(2).shape =', matrix_a.unsqueeze(2).shape)
    print('matrix_b.unsqueeze(1).shape =', matrix_b.unsqueeze(1).shape)
    print('matrix_b.unsqueeze(2).shape =', matrix_b.unsqueeze(2).shape)
    print()

    """Computes batchwise Euclidean distances using PyTorch."""
    a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
    b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
    a_times_b = matrix_a.mm(matrix_b.T)

    print('a_times_a.shape =', a_times_a.shape)
    print('b_times_b.shape =', b_times_b.shape)
    print('a_times_b.shape =', a_times_b.shape)
    print()

    approximate_distance_matrix = (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None)  # .sqrt()

    print('approximate_distance_matrix.shape =', approximate_distance_matrix.shape)

    approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix,
                                                    axis=-1, keepdims=True)

    print('approximate_coreset_anchor_distances.shape =', approximate_coreset_anchor_distances.shape)

    coreset_indices = []
    num_coreset_samples = int(len(feat_train_proj) * percentage)

    with torch.no_grad():
        for _ in tqdm(range(num_coreset_samples), desc="Subsampling..."):
            select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            matrix_a = feat_train_proj
            matrix_b = feat_train_proj[[select_idx]]
            """Computes batchwise Euclidean distances using PyTorch."""
            a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
            b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
            a_times_b = matrix_a.mm(matrix_b.T)
            coreset_select_distance = (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None)  # .sqrt()

            approximate_coreset_anchor_distances = torch.cat(
                [approximate_coreset_anchor_distances, coreset_select_distance],
                dim=-1,
            )
            approximate_coreset_anchor_distances = torch.min(
                approximate_coreset_anchor_distances, dim=1
            ).values.reshape(-1, 1)

    coreset_indices = np.array(coreset_indices)

    print('len(coreset_indices) =', len(coreset_indices))
    print('coreset_indices[:10] =', coreset_indices[:10])

    feat_train_coreset = feat_train[coreset_indices]
    feat_train_coreset = feat_train_coreset.cpu().numpy()

    del feat_train
    torch.cuda.empty_cache()

    print('feat_train_coreset.shape =', feat_train_coreset.shape)
    return feat_train_coreset

# KNNインデックスの作成
def create_knn_index(feat_train_coreset, use_gpu=True):
    import faiss
    if use_gpu:
        try:
            # Attempt to create GPU-based index
            search_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                feat_train_coreset.shape[1],
                                                faiss.GpuIndexFlatConfig())
        except Exception as e:
            print("Failed to create GPU-based Faiss index:", e)
            print("Falling back to CPU-based Faiss index.")
            use_gpu = False

    if not use_gpu:
        # Create CPU-based index
        search_index = faiss.IndexFlatL2(feat_train_coreset.shape[1])

    search_index.add(feat_train_coreset)
    return search_index

# テストデータ画像読み込み
def load_test_imgs(files_test, types_test):
    img_prep_test = {}
    gt_test = {}

    for type_test in types_test:

        img_prep_test[type_test] = []
        gt_test[type_test] = []

        for file in tqdm(files_test[type_test]):
            img = cv2.imread(file)[..., ::-1]  # BGR2RGB
            img_prep = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img_prep = img_prep[16:(256-16), 16:(256-16)]
            img_prep_test[type_test].append(img_prep)

            if (type_test == 'good'):
                gt = np.zeros_like(img_prep[..., 0], dtype=np.uint8)
            else:
                file_gt = file.replace('/test/', '/ground_truth/')
                file_gt = file_gt.replace('.png', '_mask.png')
                gt = cv2.imread(file_gt, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
                gt = gt[16:(256-16), 16:(256-16)]
                gt = (gt / np.max(gt)).astype(np.uint8)
            gt_test[type_test].append(gt)

        img_prep_test[type_test] = np.stack(img_prep_test[type_test])
        gt_test[type_test] = np.stack(gt_test[type_test])

        print('img_prep_test[%s].shape =' % type_test, img_prep_test[type_test].shape)
        print('gt_test[%s].shape =' % type_test, gt_test[type_test].shape)
    return img_prep_test

# テストデータ画像からの特徴抽出
def extract_test_features(device, model, outputs, img_prep_test, types_test, MEAN, STD,
                          patch_shapes, ref_num_patches, pretrain_embed_dimension,
                          target_embed_dimension, unfolder, patchsize=3):
    # set param
    N_batch = 25

    feat_test = {}

    for type_test in types_test:

        outputs.clear()
        feat_test[type_test] = []

        for i_batch in tqdm(range(0, len(img_prep_test[type_test]), N_batch)):

            img_batch = img_prep_test[type_test][i_batch:(i_batch + N_batch)]
            x = torch.from_numpy(img_batch).to(torch.float).to(device)
            x = x / 255
            x = x - MEAN
            x = x / STD
            x = x.permute(0, 3, 1, 2)

            with torch.no_grad():
                _ = model(x)

            f1 = outputs[0].clone()  # (B, C, H, W)
            f2 = outputs[1].clone()  # (B, C, H, W)
            feat = [f1, f2]
            shapes = [f1.shape, f2.shape]

            outputs.clear()

            # patchify
            for i in range(len(feat)):
                # (B, C, H, W) -> (B, C, H, W, PH, PW)
                with torch.no_grad():
                    feat[i] = unfolder(feat[i])
                # (B, C, H, W, PH, PW) -> (B, C, PH, PW, HW)
                feat[i] = feat[i].reshape(*shapes[i][:2], patchsize, patchsize, -1)
                # (B, C, PH, PW, HW) -> (B, HW, C, PW, HW)
                feat[i] = feat[i].permute(0, 4, 1, 2, 3)

            for i in range(1, len(feat)):
                _feat = feat[i]
                patch_dims = patch_shapes[i]
                # (B, HW, C, PW, HW) -> (B, H, W, C, PH, PW)
                _feat = _feat.reshape(_feat.shape[0], patch_dims[0],
                                            patch_dims[1], *_feat.shape[2:])
                # (B, H, W, C, PH, PW) -> (B, C, PH, PW, H, W)
                _feat = _feat.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _feat.shape
                # (B, C, PH, PW, H, W) -> (BCPHPW, H, W)
                _feat = _feat.reshape(-1, *_feat.shape[-2:])
                # (BCPHPW, H, W) -> (BCPHPW, H_max, W_max)
                _feat = F.interpolate(_feat.unsqueeze(1),
                                        size=(ref_num_patches[0], ref_num_patches[1]),
                                        mode="bilinear", align_corners=False)
                _feat = _feat.squeeze(1)
                # (BCPHPW, H_max, W_max) -> (B, C, PH, PW, H_max, W_max)
                _feat = _feat.reshape(*perm_base_shape[:-2],
                                            ref_num_patches[0], ref_num_patches[1])
                # (B, C, PH, PW, H_max, W_max) -> (B, H_max, W_max, C, PH, PW)
                _feat = _feat.permute(0, -2, -1, 1, 2, 3)
                # (B, H_max, W_max, C, PH, PW) -> (B, H_maxW_max, C, PH, PW)
                _feat = _feat.reshape(len(_feat), -1, *_feat.shape[-3:])
                feat[i] = _feat

            # (B, H, W, C, PH, PW) -> (BHW, C, PH, PW)
            feat = [x.reshape(-1, *x.shape[-3:]) for x in feat]

            for i in range(len(feat)):
                _feat = feat[i]
                # (BHW, C, PH, PW) -> (BHW, 1, CPHPW)
                _feat = _feat.reshape(len(_feat), 1, -1)
                # (BHW, 1, CPHPW) -> (BHW, D_p)
                _feat = F.adaptive_avg_pool1d(_feat,
                                            pretrain_embed_dimension).squeeze(1)
                feat[i] = _feat

            # (BHW, D_p) -> (BHW, D_p*2)
            feat = torch.stack(feat, dim=1)
            """Returns reshaped and average pooled feat."""
            # batchsize x number_of_layers x input_dim -> batchsize x target_dim
            # (BHW, D_p*2) -> (BHW, D_t)
            feat = feat.reshape(len(feat), 1, -1)
            feat = F.adaptive_avg_pool1d(feat, target_embed_dimension)
            feat = feat.reshape(len(feat), -1)

            feat_test[type_test].append(feat.cpu())

        feat_test[type_test] = torch.vstack(feat_test[type_test]).numpy()

        print('feat_test.shape[%s] =' % type_test, feat_test[type_test].shape)
    return feat_test

# テスト画像特徴と、コアセットサンプルとのKNN実施をして、異常スコアマップ取得
def get_anomaly_score(feat_test, search_index, types_test):
    k = 1

    score_test = {}

    for type_test in types_test:

        score_test[type_test], _ = search_index.search(feat_test[type_test], k)
        score_test[type_test] = score_test[type_test].reshape(-1, 28, 28)

        print('score_test[%s].shape =' % type_test, score_test[type_test].shape)
        print('np.mean(score_test[%s]) =' % type_test, np.mean(score_test[type_test]))
        print('np.mean(np.abs(score_test[%s])) =' % type_test, np.mean(np.abs(score_test[type_test])))
        print('np.std(score_test[%s]) =' % type_test, np.std(score_test[type_test]))
        print('np.max(score_test[%s]) =' % type_test, np.max(score_test[type_test]))
        print('np.min(score_test[%s]) =' % type_test, np.min(score_test[type_test]))
    return score_test

# メイン処理
def main(save_dir, backbone):
    torch_fix_seed(0)

    model, outputs, device = prepare_extractor(first_layer_index=0, second_layer_index=6, backbone=backbone)

    # ファイル名を取得
    files_train, files_test, types_test = get_files()

    img_prep_train = load_train_imgs(files_train)

    feat_train, MEAN, STD, patch_shapes, ref_num_patches, pretrain_embed_dimension, target_embed_dimension, unfolder = extract_train_features(device, model, outputs, img_prep_train)

    feat_train_coreset = core_set_sampling(feat_train, device)

    # use_gpu=Trueだとエラーが出てしまう
    search_index = create_knn_index(feat_train_coreset, use_gpu=False)

    img_prep_test = load_test_imgs(files_test, types_test)

    feat_test = extract_test_features(device, model, outputs, img_prep_test, types_test, MEAN, STD, patch_shapes, ref_num_patches, pretrain_embed_dimension, target_embed_dimension, unfolder)

    score_test = get_anomaly_score(feat_test, search_index, types_test)

    # 結果保存
    get_anomaly_detection(score_test, types_test, save_dir)

if __name__ == '__main__':
    backbone = backbones[2]
    save_dir = os.path.join('results/PatchCore/base', backbone)
    os.makedirs(save_dir, exist_ok=True)
    main(save_dir, backbone=backbone)
