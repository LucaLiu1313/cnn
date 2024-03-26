import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets

train_dir = 'a/train'
test_dir = 'a/test'
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor(),           # 将图像转换为张量
])
# 加载数据集
tarinset = datasets.ImageFolder(root=train_dir, transform=transform)
testset = datasets.ImageFolder(root=test_dir, transform=transform)

def build_resnet(input_shape=(32, 32, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    # 第一层
    with tf.device('/GPU:0'):
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)

    # 第二层
    # 使用残差连接，将输入添加到第二层的输出中
        residual = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
   # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.add([x, residual])  # 残差连接
        x = layers.MaxPooling2D((2, 2))(x)

    # 第三层
    # 使用残差连接，将输入添加到第三层的输出中
        residual = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
   # x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.add([x, residual])  # 残差连接

    # 展平层
    x = layers.Flatten()(x)
    # 全连接层
    x = layers.Dense(64, activation='relu')(x)
    # 输出层
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = build_resnet()

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# 训练模型
history = model.fit(tarinset,epochs=10,
                    validation_data=(testset))



# 绘制训练过程中的损失和准确率曲线
plt.plot(history.history['accuracy'], label='accuracy')# 准确率
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')# 验证准确率
plt.xlabel('Epoch')# x轴设置为迭代轮数
plt.ylabel('Accuracy')# y轴设置为准确率
plt.ylim([0, 1])# y轴范围设置为0-1
plt.legend(loc='lower right')# 图例在右下角
plt.show()# 启动！

# 在测试集上评估模型
test_loss, test_acc = model.evaluate(testset, verbose=2)
print('\nTest accuracy:', test_acc)

tf.keras.backend.clear_session()