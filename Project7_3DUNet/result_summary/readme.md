Predict result
`
python batch_predict.py ^
  --images_dir "D:\COMP3710\HipMRI_Study_open\semantic_MRs" ^
  --weights ".\runs\hipmri3d_unet_bin\best.pt" ^
  --out_dir ".\Predict_Batch" ^
  --label_root "D:\COMP3710\HipMRI_Study_open" ^
  --label_dir "semantic_labels_only" ^
  --limit 10

`