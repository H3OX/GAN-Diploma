from utils import generate_real_samples
from utils import generate_fake_samples
from utils import generate_latent_points
from numpy import ones


# Тренировка модели
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # Итерация по эпохам
    for i in range(n_epochs):
        # Итерация по батчам
        for j in range(bat_per_epo):
            # Рандомно выбираем настоящие объекты
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # Обновляем веса
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # Генерируем фейковые семплы
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # Обновляем веса дискриминатора
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # Объявляем вектор шума для последующей подачи в генератор
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # Создание обратных классов для фейковых объектов
            y_gan = ones((n_batch, 1))
            # Обновляем генератор с учетов ошибки дискриминатора
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # Выводим общую ошибку нейросети
            print(f'Эпоха: {i + 1}; Батч: {j + 1}/{bat_per_epo}; Ошибка дискриминатора: {d_loss1}; Ошибка дискриминатора на фейковых классах: {d_loss2}; Ошибка генератора: {g_loss}')
        # save the generator model
        if i % 10 == 0:
            g_model.save(f'gen_model_epoch{i}.h5')

