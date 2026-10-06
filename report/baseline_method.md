## 4. Dataset and Data Preparation

For the Random Forest baseline, two sensor configurations were evaluated. The first being accelerometer, denoted as ACC, uses the three total-acceleration channels along the x, y, and z axes. The second configuration which is accelerometer + gyroscope (ACC+GYRO), combines the three total-acceleration channels with the three body-gyroscope channels, resulting in six sensor channels.

The preprocessed data are represented as windows of 128 samples. The official training and testing split is retained, while a subject-wise validation split is obtained from the training data. Data preprocessing, including windowing and normalization, is performed by the shared data-processing pipeline before the Random Forest model is trained. After preprocessing, the two inputs will have the shape of:
- ACC: (128, 3)
- ACC+GYRO: (128, 6)

## 5. Methods

### 5.1 Baseline Method: Random Forest

Random Forest (RF) is an ensemble learning method that combines the predictions of multiple decision trees, each tree is trained using a randomized subset of the training data and features, and the final prediction is determined by aggregating the predictions of the individual trees. Unlike the deep-learning models evaluated in the main experiment, the Random Forest classifier does not directly operate on the temporal sequence of sensor samples; therefore, handcrafted statistical features were extracted from each 128-sample sensor window before classification. Those features are:
1. Mean
2. Standard deviation (std)
3. Minimum (min)
4. Maximum (max)
5. Energy

| Sensor configuration | Channels | Features per channel | Total features |
| --- | --- | ---| --- |
| ACC | 3 | 5 | 15 |
| ACC+GYRO | 6 | 5 | 30 |

The extracted features are then used as input to a Random Forest classifier. The classifier is trained exclusively on the training split, the validation split is used for model evaluation and comparison during development, while the test split is reserved for the final evaluation. The train/test split are based on the official split and the val split contains 4 whole subjects carved from the official train set, seed 42 (as written in `shared.yaml`)

RF model configurations:
- Number of trees (n_estimators): 300
- Random state: 42
- n_jobs: -1
Random state is set at 42 t ensure reproducibility, as mentioned in `shared.yaml`

Two Random Forest configurations were evaluated under the same training procedure. The first configuration uses only total acceleration (ACC), while the second combines total acceleration with body gyroscope measurements (ACC+GYRO). Both configurations use the same five handcrafted features per sensor axis and the same Random Forest hyperparameters. This allows the effect of adding gyroscope information to be evaluated while keeping the classifier and feature-extraction method constant.

## 7. Results and Discussion
| Model | Sensors | Seed | Accuracy | Marco F1 | Epochs | Train time (sec) |
| --- | --- | --- | --- | --- | --- | --- |
| RF | ACC | 42 | 0.7648 | 0.7615 | 0 | 0.7945 |
| RF | ACC+GYRO | 42 | 0.8039 | 0.8016 | 0 | 0.7876 |

The ACC+GYRO configuration achieved higher accuracy and macro-F1 than the ACC configuration. This suggests that gyroscope information provides additional discriminative information for distinguishing activities that have similar acceleration patterns. In particular, activities involving changes in body orientation or rotational movement may benefit from the additional gyroscope measurements.