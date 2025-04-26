#plik do testowania modułów

from data_loader.data_loader import TimeWindowSegmenter
from data_loader.features_accelerometer import extract_acc_features
from data_loader.features_temporal import extract_temporal_features
from data_loader.features_cosine import extract_cosine_distances

data_processor = TimeWindowSegmenter(df_path="D:\MetaMotion\metamotion-ml\data_loader\wsidm.parquet", window_size=10, step_size=10)

for window in data_processor.segment():
    # acc_feats = extract_acc_features(window)
    # print(acc_feats)
    # for k, v in acc_feats.items():
    #     if 'jerk' in k:
    #         print(k, v)

    # feats = extract_temporal_features(window)
    # print(feats)

    feats = extract_cosine_distances(window)
    print("Cosine distances:")
    for k, v in feats.items():
        print(f"{k}: {v}")

    break
