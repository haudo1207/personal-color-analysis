# Training Project (Personal Color AI)

Huong dan nay tach phan train thanh mot mini-project de de quan ly va mo rong trong thuc te.

## Cau truc

- `run_train.py`: entrypoint duy nhat de train.
- `src/train_ai/config.py`: config + labels map.
- `src/train_ai/dataset.py`: doc CSV, xu ly labels, sampler.
- `src/train_ai/model.py`: dinh nghia mo hinh.
- `src/train_ai/metrics.py`: metrics cho bai toan multi-task.
- `src/train_ai/trainer.py`: train loop, evaluate, early stop, luu artifacts.
- `train_split.csv`, `val_split.csv`: du lieu train/val.

## Cach chay

```bash
python run_train.py
```

Tham so hay dung:

```bash
python run_train.py --epochs 20 --batch-size 64 --lr 0.0005
python run_train.py --export-model-path "../personal-color-ai-analysis-main/personal-color-ai-analysis-main/models/best_personal_color_model.pth"
```

## Artifacts

Moi lan chay tao mot run moi tai:

`outputs/runs/<timestamp>/`

Bao gom:

- `best_personal_color_model_full.pth`
- `evaluation_metrics.json`
- `history.json`
- `confusion_matrices.png`

## Luu y du lieu

CSV can co cac cot:

- bat buoc: `fitzpatrick`, `undertone`, `personal_color`
- anh: uu tien cot `image_path`; neu khong co thi dung `file`

Neu `fitzpatrick` dang 1..6, pipeline se tu dong chuyen ve 0..5.

