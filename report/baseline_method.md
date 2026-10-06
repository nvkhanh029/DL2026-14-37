## 4. Dataset and Data Preparation

For the Random Forest baseline, two sensor configurations were evaluated. The first being accelerometer, denoted as ACC, uses the three total-acceleration channels along the x, y, and z axes. The second configuration which is accelerometer + gyroscope (ACC+GYRO), combines the three total-acceleration channels with the three body-gyroscope channels, resulting in six sensor channels.

The preprocessed data are represented as windows of 128 samples. The official training and testing split is retained, while a subject-wise validation split is obtained from the training data. Data preprocessing, including windowing and normalization, is performed by the shared data-processing pipeline before the Random Forest model is trained. After preprocessing, the two inputs will have the shape of:
- ACC: (128, 3)
- ACC+GYRO: (128, 6)

## 5. Methods

### 5.1 Baseline Method: Random Forest

Random Forest (RF) is an ensemble learning method that combines the predictions of multiple decision trees, each tree is trained using a randomized subset of the training data and features, and the final prediction is determined by aggregating the predictions of the individual trees. Unlike the deep-learning models evaluated in the main experiment, the Random Forest classifier does not directly operate on the temporal sequence of sensor samples; therefore, handcrafted statistical features were extracted from each 128-sample sensor window before classification.

$$\mu=\frac{1}{T}\sum_{t=1}^{T}x_t$$