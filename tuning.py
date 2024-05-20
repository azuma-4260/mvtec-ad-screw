from PaDim import *
from seed import torch_fix_seed
from tqdm import trange
from PatchCore import *


def PaDim():
    results = []

    for layer1_index in trange(3):
        for layer2_index in trange(4):
            for layer3_index in trange(6):
                # 毎回seedをリセットしないと乱数が変わってしまう
                torch_fix_seed(0)

                model, device, outputs = get_model(layer1_index, layer2_index, layer3_index)

                files_train, files_test, types_test = get_files()

                f1_train, f2_train, f3_train, orb, kp1, des1, bf = get_features_train(files_train, device, model,
                                                                                      outputs, use_matching=True)

                f1_test, f2_test, f3_test = get_features_test(files_test, device, model, types_test, outputs, orb, kp1,
                                                              des1, bf, use_matching=True)

                f1_train, f2_train, f3_train, f1_test, f2_test, f3_test = get_features_random_choice(f1_train, f2_train,
                                                                                                     f3_train, f1_test,
                                                                                                     f2_test, f3_test,
                                                                                                     types_test)

                f123_train, f123_test = get_features_concat(f1_train, f2_train, f3_train, f1_test, f2_test, f3_test,
                                                            types_test)

                score_test = get_mahalanobis_distance(f123_train, f123_test, types_test)

                score = get_anomaly_detection(score_test, types_test, save_dir='results/PaDim/tuning')

                result = {'layer1_index': layer1_index,
                          'layer2_index': layer2_index,
                          'layer3_index': layer3_index,
                          'score': score}
                results.append(result)

                del model, device, outputs, files_train, files_test, types_test, f1_train, f2_train, f3_train, \
                    orb, kp1, des1, bf, f1_test, f2_test, f3_test, f123_train, f123_test, score_test, score, result

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    import json
    with open('results/PaDim/tuning/results.json', 'w') as f:
        json.dump(results, f, indent=2)

def PatchCore(save_dir, backbone):
    results = []
    if backbone == 'wide_resnet50_2':
        layer2_start = 0
        layer2_end = 3
        layer2_step = 1
        layer3_start = 0
        layer3_end = 6
        layer3_step = 1
    elif backbone == 'densenet201':
        layer2_start = 1
        layer2_end = 12
        layer2_step = 4
        layer3_start = 1
        layer3_end = 48
        layer3_step = 8
    elif backbone == 'effficientnet-b5':
        layer2_start = 0
        layer2_end = 4
        layer2_step = 1
        layer3_start = 0
        layer3_end = 6
        layer3_step = 1

    for i in range(layer2_start, layer2_end + 1, layer2_step):
        for j in range(layer3_start, layer3_end + 1, layer3_step):
            layer2_index = i
            layer3_index = j

            torch_fix_seed(0)

            model, outputs, device = prepare_extractor(layer2_index, layer3_index, backbone=backbone)

            # ファイル名を取得
            files_train, files_test, types_test = get_files()

            img_prep_train = load_train_imgs(files_train)

            feat_train, MEAN, STD, patch_shapes, ref_num_patches, pretrain_embed_dimension, target_embed_dimension, unfolder = extract_train_features(device, model, outputs, img_prep_train)

            feat_train_coreset = core_set_sampling(feat_train, device)

            search_index = create_knn_index(feat_train_coreset, use_gpu=False)

            img_prep_test = load_test_imgs(files_test, types_test)

            feat_test = extract_test_features(device, model, outputs, img_prep_test, types_test, MEAN, STD, patch_shapes, ref_num_patches, pretrain_embed_dimension, target_embed_dimension, unfolder)

            score_test = get_anomaly_score(feat_test, search_index, types_test)

            # 結果保存
            score = get_anomaly_detection(score_test, types_test, save_dir=save_dir)

            result = {'layer2_index': layer2_index,
                      'layer3_index': layer3_index,
                      'score': score}
            results.append(result)

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    import json
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    mode = 'PatchCore'
    backbones = ['wide_resnet50_2', 'densenet201', 'effficientnet-b5']

    if mode == 'PaDim':
        PaDim()
    else:
        backbone = backbones[2]
        save_dir = f'results/PatchCore/tuning/{backbone}'
        os.makedirs(save_dir, exist_ok=True)
        PatchCore(save_dir, backbone)

