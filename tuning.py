from main import *
from seed import torch_fix_seed
from tqdm import trange


def main():
    results = []

    for layer1_index in trange(3):
        for layer2_index in trange(4):
            for layer3_index in trange(6):
                # 毎回seedをリセットしないと乱数が変わってしまう
                torch_fix_seed(0)

                model, device, outputs = get_model(layer1_index, layer2_index, layer3_index)

                files_train, files_test, types_test = get_files()

                f1_train, f2_train, f3_train, orb, kp1, des1, bf = get_features_train(files_train, device, model,
                                                                                      outputs)

                f1_test, f2_test, f3_test = get_features_test(files_test, device, model, types_test, outputs, orb, kp1,
                                                              des1, bf)

                f1_train, f2_train, f3_train, f1_test, f2_test, f3_test = get_features_random_choice(f1_train, f2_train,
                                                                                                     f3_train, f1_test,
                                                                                                     f2_test, f3_test,
                                                                                                     types_test)

                f123_train, f123_test = get_features_concat(f1_train, f2_train, f3_train, f1_test, f2_test, f3_test,
                                                            types_test)

                score_test = get_mahalanobis_distance(f123_train, f123_test, types_test)

                score = get_anomaly_detection(score_test, types_test, save_dir='results/tuning')

                result = {'layer1_index': layer1_index,
                          'layer2_index': layer2_index,
                          'layer3_index': layer3_index,
                          'score': score}
                results.append(result)

                del model, device, outputs, files_train, files_test, types_test, f1_train, f2_train, f3_train, \
                    orb, kp1, des1, bf, f1_test, f2_test, f3_test, f123_train, f123_test, score_test, score, result

    results = sorted(results, key=lambda x: x['score'], reverse=True)
    import json
    with open('results/tuning/results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
