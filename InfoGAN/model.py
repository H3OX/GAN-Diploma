# example of training an infogan on mnist
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy import hstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from matplotlib import pyplot


# Определение модели дискриминатора
def define_discriminator(n_cat, in_shape=(28, 28, 1)):
    # Инициализация весов
    init = RandomNormal(stddev=0.02)
    # Входной слой изображения
    in_image = Input(shape=in_shape)
    # Даунсемплинг до 14x14
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.1)(d)
    # Даунсемплинг до 7x7
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = BatchNormalization()(d)
    # Нормализация
    d = Conv2D(256, (4, 4), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = BatchNormalization()(d)
    # Перевод в одномерное признаковое пространство
    d = Flatten()(d)
    # Выход фейк/настоящее
    out_classifier = Dense(1, activation='sigmoid')(d)
    # Определение модели дискриминатора
    d_model = Model(in_image, out_classifier)
    # Компиляция модели дискриминатора
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    # Определение структуры Q-модели
    q = Dense(128)(d)
    q = BatchNormalization()(q)
    q = LeakyReLU(alpha=0.1)(q)
    # Выход q-модели
    out_codes = Dense(n_cat, activation='softmax')(q)
    # Определение самой Q-модели
    q_model = Model(in_image, out_codes)
    return d_model, q_model


# Определение модели-генератора
def define_generator(gen_input_size):
    # Инициализация весов
    init = RandomNormal(stddev=0.02)
    # Вход изображений
    in_lat = Input(shape=(gen_input_size,))
    # Определение плейсхолдера для изображений 7х7
    n_nodes = 512 * 7 * 7
    gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    gen = Reshape((7, 7, 512))(gen)
    # Нормализация
    gen = Conv2D(128, (4, 4), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    # Апсемплинг до 14x14
    gen = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    # Апсемплинг до 28x28
    gen = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(gen)
    # Активация гиперболическим тангенсом
    out_layer = Activation('tanh')(gen)
    # Определение модели
    model = Model(in_lat, out_layer)
    return model


# Определение GAN-модели, комбинацией Q-модели, генератора и дискриминатора
def define_gan(g_model, d_model, q_model):
    # Закрепление необновляемых весов за моделью-дискриминатором
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # Создание связи между выходом генератора и входом дискриминатора
    d_output = d_model(g_model.output)
    # Создание связи между выходом дискриминатора и входом Q-модели
    q_output = q_model(g_model.output)
    # Определение GAN-модели
    model = Model(g_model.input, [d_output, q_output])
    # Компиляция GAN-модели
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
    return model


# Загрузка изображений
def load_real_samples():
    # Загрузка датасета
    (trainX, _), (_, _) = load_data()
    # Добавление каналов
    X = expand_dims(trainX, axis=-1)
    # Конвертация из целых в десятичные
    X = X.astype('float32')
    # Нормализация в интервале [-1, 1]
    X = (X - 127.5) / 127.5
    print(X.shape)
    return X


# Выбор реальных семплов
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y


# Генерация вектора шума для генератора
def generate_latent_points(latent_dim, n_cat, n_samples):
    z_latent = randn(latent_dim * n_samples)
    z_latent = z_latent.reshape(n_samples, latent_dim)
    cat_codes = randint(0, n_cat, n_samples)
    cat_codes = to_categorical(cat_codes, num_classes=n_cat)
    z_input = hstack((z_latent, cat_codes))
    return [z_input, cat_codes]


# Использование генератора для вывода новых изображений
def generate_fake_samples(generator, latent_dim, n_cat, n_samples):
    # generate points in latent space and control codes
    z_input, _ = generate_latent_points(latent_dim, n_cat, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y


# Генерация изображений в миде матрицы
def summarize_performance(step, g_model, gan_model, latent_dim, n_cat, n_samples=100):
    X, _ = generate_fake_samples(g_model, latent_dim, n_cat, n_samples)
    X = (X + 1) / 2.0
    for i in range(100):
        pyplot.subplot(10, 10, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    filename1 = f'generated_plot_{step + 1}.png'
    pyplot.savefig(filename1)
    pyplot.close()
    # Сохранение модели-генератора
    filename2 = f'generator_model_{step + 1}.h5'
    g_model.save(filename2)
    # Сохранение InfoGAN-модели
    filename3 = f'gan_model_{step + 1}.h5'
    gan_model.save(filename3)
    print(f'>Сохранено: {filename1}, {filename2}, и {filename3}')


# Тренировка моделей
def train(g_model, d_model, gan_model, dataset, latent_dim, n_cat, n_epochs=100, n_batch=64):
    # Вычисление количества батчей за эпоху
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # Вычисление количества итераций
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    for i in range(n_steps):
        # Получаем случайные n изображений
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # Обновляем веса дискриминатора и Q-модели
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        # Генерируем фейковые семплы
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_cat, half_batch)
        # Обновляем веса дискриминатора
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # Выводим вектор шума как вход для генератора
        z_input, cat_codes = generate_latent_points(latent_dim, n_cat, n_batch)
        # Вывод обратных классов для генерируемых объектов
        y_gan = ones((n_batch, 1))
        # Обновление моделей через обратное распространение ошибки
        _, g_1, g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
        # Суммируем ошибки
        print(f'>Эпоха: {i + 1}, D_loss[{d_loss1},{d_loss2}], G_loss[{g_1}], Q_loss[{g_2}]')
        # Тестирование моделй каждые 10 эпох
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, gan_model, latent_dim, n_cat)


n_cat = 10
latent_dim = 62
d_model, q_model = define_discriminator(n_cat)
gen_input_size = latent_dim + n_cat
g_model = define_generator(gen_input_size)
gan_model = define_gan(g_model, d_model, q_model)
dataset = load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim, n_cat)
