from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="new")
trainer.evaluateModel(model_path="new/models", json_path="new/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
