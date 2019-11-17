from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setGpuUsage(1)
trainer.setDataDirectory(data_directory="new")
trainer.setTrainConfig(object_names_array=["Car", "Delineator", "Wall", "SignBoard"],
                       batch_size=1, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()