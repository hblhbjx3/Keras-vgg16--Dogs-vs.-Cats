from keras import optimizers
from keras import applications
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# 数据集
img_height, img_width = 256, 256  # 图片高宽
batch_size = 2  # 批量大小
epochs = 3# 迭代次数
train_data_dir = 'maogou/train'  # 训练集目录
validation_data_dir = 'maogou/validation'  # 测试集目录
OUT_CATEGORIES = 1  # 分类数
nb_train_samples = 2000  # 训练样本数
nb_validation_samples = 200  # 验证样本数

# 定义模型
# 预训练的VGG16网络，替换掉顶部网络
base_model = applications.VGG16(weights="imagenet", include_top=False,
                                input_shape=(img_width, img_height, 3))
print(base_model.summary())

# 冻结预训练网络前15层
for layer in base_model.layers[:15]: layer.trainable = False
# 自定义顶层网络
top_model = Sequential()
# 将预训练网络展平.
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# 全连接层，输入像素256
top_model.add(Dense(256, activation='relu'))
# Dropout概率0.5
top_model.add(Dropout(0.5))
# 输出层，二分类
top_model.add(Dense(OUT_CATEGORIES, activation='sigmoid'))

# top_model.load_weights("")  # 单独训练的自定义网络
# 新网络=预训练网络+自定义网络
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])  # 损失函数为二进制交叉熵，优化器为SGD
# 训练数据预处理器，随机水平翻转
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 测试数据预处理器
# 训练数据生成器
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size, class_mode='binary')
# 验证数据生成器
validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        shuffle=False)
# 保存最优模型
checkpointer = ModelCheckpoint(filepath='dogcatmodel_200.h5', verbose=1, save_best_only=True)
# 训练&评估
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    verbose=2, workers=12, callbacks=[checkpointer])# 每轮一行输出结果，最大进程12
