python prepro.py -d talk2car -s kul
python scripts/extract_target_feats.py -d talk2car -s kul --batch_size 40 -g 0
python scripts/extract_image_feats.py -d talk2car -s kul --batch_size 40 -g 0
python scripts/extract_target_feature_map.py -d talk2car -s kul --batch_size 1 -g 3
python train.py -d talk2car -s kul -g 1 --id slr --id2 ver1
python eval_generation.py -d talk2car -s kul -g 1 --id slr --id2 ver1 -split test --batch_size 1