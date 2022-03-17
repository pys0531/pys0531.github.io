---
title:  <font size="5">Pytorch에서 Tensorboard 실행</font>
excerpt: "Pytorch에서 Tensorboard 실행"
toc: true
toc_sticky: true
categories:
  - Pytorch
tags:
  - Machine Learning
  - Tensorboard
  - Pytorch
last_modified_at: 2022-03-14T16:38:00-55:00
---

<font size="3">
Pytorch에서는 데이터를 시각화 하기위해 Tensorboard를 지원한다.
간단한 명령어로 Train Log를 찍어볼수 있어서 자주 애용한다.
</font> 

## 설치
```
$ pip install tensorboard
```

## 사용법
<font size="3">
아래는 내 기준으로 가장 애용하는 형식이다.
</font>
 
```python
from torch.utils.tensorboard import SummaryWriter
writer_dict = {
                    'writer': SummaryWriter(log_dir="../output/log/"),
                    'train_global_steps': 0,
                    'valid_global_steps': 0,
                 }

for epoch in range(start_epoch, end_epoch)
    ***
    학습코드
    ***

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    writer.add_scalar('train_losses', train_losses.avg, global_steps)
    writer.add_scalar('train_acc', train_avg_acc, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    ***
    평가코드
    ***
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_losses', valid_losses.avg, global_steps)
    writer.add_scalar('valid_acc', valid_avg_acc, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

writer_dict['writer'].close()
```

## 실행 
<font size="3">
터미널에 Log 파일이 저장된 폴더 명시
주소창에 <http://localhost:6006/>를 입력하면 확인가능
</font>
```
$ tensorboard logdir="../output/log/"
```


## Scalar 이외에 다른 데이터 추가

<font size="3">
Image
</font>
```
writer.add_image("image_name", img_grid)
```

<font size="3">
Model Graph
</font>
```
writer.add_graph(net,images)
```

<font size="3">
Scalar
</font>
```
writer.add_scalar('Train', Loss_val, global_step)
```


<font size="3">
Scalar
</font>
```
writer.add_scalar('Train', Loss_val, global_step)
```


<font size="3">
Figure
</font>
```
writer.add_figure('pred Vs target', fig, global_step)
```


<font size="3">
Projector
</font>
```python
def select_n_random(data, labels, n=100):
    '''
    데이터셋에서 n개의 임의의 데이터포인트(datapoint)와 그에 해당하는 라벨을 선택합니다
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# 임의의 이미지들과 정답(target) 인덱스를 선택합니다
images, labels = select_n_random(trainset.data, trainset.targets)

# 각 이미지의 분류 라벨(class label)을 가져옵니다
class_labels = [classes[lab] for lab in labels]

# 임베딩(embedding) 내역을 기록합니다
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()
```
![](/assets/images/2022-03-14-Pytorch에서 Tensorboard 실행/tensorboard_projector.png)
