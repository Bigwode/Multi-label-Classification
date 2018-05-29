5个epoch之后两个任务可以分别达到大概0.65和0.84的准确率
(1)、fit()没有使用Data_Augmentation方法，fit_generator()没有找到multi-task标签如何输入的方法。。。

```bash
 Epoch 5/5
4013/4013 [==============================] - 34s 8ms/step - loss: 0.9312 
- predict_class_loss: 0.5541 - predict_attri_loss: 0.3771 
- predict_class_acc: 0.7149 - predict_attri_acc: 0.8428 
- val_loss: 1.0013 - val_predict_class_loss: 0.6287 
- val_predict_attri_loss: 0.3726 
- val_predict_class_acc: 0.6537 
- val_predict_attri_acc: 0.8441
```
(2)、train_on_batch训练方法:
使用了Data_Augmentation方法，但是val数据需要单独evaluation.

```bash
('batches:', 3981, 'loss:', 0.99187338, 'loss_class:', 0.55250168, 'loss_attri:', 0.43937171, 'acc_class:', 0.625, 'acc_attr:', 0.796875)
('batches:', 4013, 'loss:', 1.0529957, 'loss_class:', 0.68102407, 'loss_attri:', 0.37197155, 'acc_class:', 0.53125, 'acc_attr:', 0.8359375)
Starting to test...
4022/4022 [==============================] - 13s 3ms/step
('test loss: ', [1.0471279424319275, 0.65188837181804549, 0.39523956855395015, 0.60790651423133235, 0.83419318735036374])
```



