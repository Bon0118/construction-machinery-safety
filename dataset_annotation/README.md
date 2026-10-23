# 告警数据集试标工具

本目录只读访问平台的 `alarm_outputs`，不会修改原始告警事件。所有抽样图片、标签和报告写入独立的 `dataset_outputs`。

安全帽 v1 规则：

- 类别顺序为 `person=0`、`head=1`、`helmet=2`。
- `head` 与 `helmet` 严格互斥。
- 普通帽子、摩托车头盔、焊接面罩及不规范佩戴安全帽均属于 `head`。
- `helmet` 框包含施工安全帽与对应完整人头，不含脖子和肩膀。
- 在保持比例缩放到 640×640 后，头部框短边小于20像素时不标头部类别。
- `person` 只标可见范围，不设固定最小尺寸。
- 画面边缘截断的框允许保留，并记录 `truncated=true`。
- 灰色异常帧和大面积白色过曝帧不进入数据集。
- 原始告警图保存在每次输出的 `images` 中用于溯源。
- 训练候选保存在 `crops/images`：根据人员框生成正方形头肩区域，再统一缩放到 640×640。
- 每个裁剪 JSON 都记录原告警目录、原图人员框、原图裁剪框和裁剪坐标中的人员框。
- 经人工或视觉质检确认的牛、车辆及设备误检保留为原图负样本，但不生成 `person` 标签或头肩裁剪。

当前输出属于模型预标注，不等于人工真值；`manual_review=true` 的样本必须人工确认后才能用于正式训练。

## 安全帽 v2 独立教师流水线

`run_safety_helmet_teacher_pilot.py` 不再使用待调优的安全帽 YOLOv5 模型生成标签。它依次使用独立 COCO 通用人员检测模型、Grounding DINO 候选定位、SAM2.1 边界细化和本地 Qwen3-VL-4B 语义选框。原始告警图继续保留用于溯源，640×640 人员头肩裁剪图作为主要训练候选。

使用独立 Python 3.10 环境运行：

```powershell
C:\Users\LENOVO\anaconda3\envs\helmet_annotation\python.exe `
  dataset_annotation\run_safety_helmet_teacher_pilot.py `
  --source-manifest dataset_outputs\safety_helmet_pilot_50_20260826_170521\manifest.jsonl
```

所有输出保持 `teacher_prelabeled_pending_manual_review` 状态，人工确认前不得作为训练真值。
