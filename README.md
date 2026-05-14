# project_recycle_mate
딥러닝 팀플 


## v1-> v2 
v1은 base model로 scratch인 yolov8n.yaml을 이용해 처음부터 학습하려 했지만 미리 학습된 모델을 통해서 초기학습을 안정화 하고 성능을 올리고자  v2에서는 pretrained를 이용해서 Fine-tuning 하는 것으로 바꾸게 됨     

Pretrained 모델에 초기 학습률을 바로 적용할 경우 gradient가 튈 수 있어 LambdaLR을 활용한 Warmup구간을 적용해 초반 gradient가 크게 튀지 않도록 함  

pre-resize를 통해서 학습 도중 이미지 원본을 읽는 것이 아닌 미리 640*640으로 줄여놓아 하드웨어 병목을 해결하여 gpu 사용을 극대화 함 

v1 에서는 yolov8의 손실함수인 v8DetectionLoss가 검증 데이터를 보고 평균과 분산을 저장했기 때문에 v2에서는 학습 데이터의 평균과 분산만 기억하도록 함 
```
with torch.no_grad():# 가중치 업데이트만 막아줌
    model.train() 
    preds = model(images) # 검증 데이터를 보고 메모리를 수정함 (오염 발생)
    model.eval() 
```

```
# v2 방식 (개선안)
model.train() 
with torch.no_grad(): 
    preds = model(images)
```