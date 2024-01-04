# 数据挖掘大作业

将所有原始数据放置于`./data/`目录下。

## 任务1 路网匹配

### 1.1 异常点检测

原始的轨迹序列可能存在异常值。我们主要处理两类异常值：漂移点和缺失点。

- 漂移点：3个gps点 x1, x2, x3，其中x1和x3的距离在阈值之内，而x2和x1的距离、x2和x3的距离都大于阈值，则x2认为是异常点，对其坐标进行线性插值。
- 缺失点：两个gps点的距离大于某个阈值（该阈值远小于漂移阈值），则认为这两点之间存在缺失点，进行线性插值。

使用下面的命令完成异常点检测，并生成新的轨迹数据文件 `new_traj.csv`.

```
python fix_trajs.py
```

### 1.2 路网匹配

我们使用FMM进行路网匹配。FMM的输入包括路网的拓扑文件 `edges.shp` 和轨迹文件 `trips.shp`.

使用下面的命令生成FMM所需的输入文件。

```
python generate_fmm_input.py
```

然后基于生成的两个文件，通过FMM匹配，匹配的结果放置于`./data/fmmr.txt`

## 任务2 路段聚类

### 2.1 特征提取

路段的重要特征是其流量和流速，同时考虑到交通的周期性，我们设计了如下四种特征：

- flow_day: 路段在周一至周日某一天的流量。
- flow_hour: 路段在0点至23点某一个小时内的流量。
- speed_day: 路段在周一至周日某一天的平均速度。
- speed_hour: 路段在0点至23点某一个小时内的平均速度。

通过下面的命令来进行统计和插值填补，从而得到上述四种特征：

```
python generate_road_flow_speed.py
```

生成的四种特征存储在./data/目录下。

```
flow_day.npy: [38027, 7]
flow_hour.npy: [38027, 24]
speed_day.npy: [38027, 7]
speed_hour.npy: [38027, 24]
```

### 2.2 路段聚类

基于`road.csv`中的原始特征和新构建的四种特征进行聚类。进入`./task2/`目录后，通过如下命令进行聚类和可视化：
```
python cluster.py
```

## 任务3 ETA估计

我们基于Transformer模型完成eta估计。

### 3.1 数据集构建

通过下面的命令来构建数据集。我们将`traj.csv`中 80%的轨迹用于训练，20%的轨迹用于验证，`eta_task.csv`的轨迹用于测试。

```
python generate_eta_task_data.py
```

生成的数据集 `tain_x.npy, train_y.npy, val_x.npy, val_y.npy, test_x.npy`将放置于`./data/`目录下。

其中，x数据集的形状为`[N, L, D]`，其中`N`为轨迹个数，`L`为最大轨迹长度，`D=13`为特征维度，为了便于后续训练，长度不足`L`的轨迹都用`-1`进行补齐。输入的13个特征依次为：

```
'highway','lanes','tunnel','bridge','alley','roundabout','length','maxspeed','width','flow_day','flow_hout','speed_day','speed_hour'
```

其中，最后6个特征分别进行了标准化。

### 3.2 模型训练

基于Transformer的模型在`./task3/model.py`中。通过下面的命令来训练模型（参数可以在该文件中修改）

```
python train_eta.py
```

完成训练后，通过下面的命令来生成在测试集上的结果 `./data/eta_result.csv`。

```
python dump_eta_task_result.py
```

**模型在验证集上的平均绝对误差（MAE）为221.16秒，平均相对误差（MAPE）为21.56%。**

## 任务4 下一跳预测

我们基于LSTM模型完成下一跳预测任务。

### 4.1 模型训练
进入`./task4/`目录后，通过下面的命令来训练模型。

```
python model_train.py
```

### 4.2 结果导出
完成训练后，在`./task4/`目录下通过下面的命令来导出下一跳预测结果。
```
python jump_predict.py 
```


