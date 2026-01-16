# Model-Merging

### Examples:

#### LayerSwap Merging
```bash
python merge_vlm.py --base_model google/paligemma-3b-pt-224 --model_a NYUAD-ComNets/FaceScanPaliGemma_Race --model_b NYUAD-ComNets/FaceScanPaliGemma_Age --output ./weights/merged_layerswap --mode layerswap --layerswap_layer 12 --alpha 0.5 --alpha2 0.5
```
#### TIES Merging (recommended)
```bash
python merge_vlm.py  --base_model google/paligemma-3b-pt-224  --model_a NYUAD-ComNets/FaceScanPaliGemma_Race --model_b NYUAD-ComNets/FaceScanPaliGemma_Age  --output ./weights/merged_ties --mode ties --alpha 1.0 --alpha2 1.0 --density 0.3
```
#### DARE-TIES Merging
```bash
python merge_vlm.py --base_model google/paligemma-3b-pt-224 --model_a NYUAD-ComNets/FaceScanPaliGemma_Race --model_b NYUAD-ComNets/FaceScanPaliGemma_Age  --output ./weights/merged_dare --mode dareties --alpha 1.2 --density 0.2 --device cpu
```
#### DARE-Linear Merging
```bash
python merge_vlm.py  --base_model google/paligemma-3b-pt-224 --model_a NYUAD-ComNets/FaceScanPaliGemma_Race --model_b NYUAD-ComNets/FaceScanPaliGemma_Age --output ./weights/merged_darelinear --mode darelinear --density 0.3 --alpha 0.5 --alpha2 0.5
```
#### Simple weighted average
```bash
python merge_vlm.py  --base_model google/paligemma-3b-pt-224  --model_a NYUAD-ComNets/FaceScanPaliGemma_Race --model_b NYUAD-ComNets/FaceScanPaliGemma_Age   --output ./weights/merged_base  --mode base --alpha 0.5
```


## Finetune

#### Docvqa
```bash
python finetuning/docvqa.py docvqa_root D:/VQA/docvqa --model_path google/paligemma-3b-pt-448 --output_dir ./models/paligemma-docvqa --num_epochs 3 --batch_size 4 --train_split_ratio 0.8
```

#### VQA2
```bash
python finetuning/vqav2.py --vqa_root D:/VQA/cocoqa --model_path google/paligemma-3b-pt-448 
  --output_dir ./models/paligemma-vqav2 --num_epochs 3 --batch_size 4  --train_split_ratio 0.8
```

#### GQA
```bash
python finetuning/gqa.py --gqa_root D:/GQA --model_path google/paligemma-3b-pt-448 --output_dir ./models/paligemma-gqa --num_epochs 3 --batch_size 4 --train_split_ratio 0.8
```

#### Captioning
```bash
python finetuning/coco_captioning.py --coco_root D:/coco2017 --model_path google/paligemma-3b-pt-448 --output_dir ./models/paligemma-coco --num_epochs 3 --batch_size 4 --train_split_ratio 0.8
```