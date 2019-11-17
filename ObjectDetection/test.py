from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-001--loss-0014.979.h5")
detector.setJsonPath("new/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="frame122.jpg", output_image_path="frame122_new-detected.jpg")
for detection in detections:
    if detection["percentage_probability"] > 80.0:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])