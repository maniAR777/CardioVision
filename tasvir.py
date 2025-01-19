import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# تنظیم مسیرهای داده
normal_path = r"C:\Users\Lenovo\Desktop\Data\Normal"
mi_path = r"C:\Users\Lenovo\Desktop\Data\MI"

# توابع کمکی


def load_image(path, grayscale=True):
    """خواندن تصویر به صورت سیاه و سفید"""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(path)


def prepare_data(normal_path, mi_path, target_size=(128, 128)):
    """آماده‌سازی داده‌ها"""
    normal_files = sorted(os.listdir(normal_path))
    mi_files = sorted(os.listdir(mi_path))

    normal_images = []
    mi_images = []

    for normal_file in normal_files:
        normal_img = load_image(os.path.join(normal_path, normal_file))
        normal_img = cv2.resize(normal_img, target_size)
        normal_img = normal_img / 255.0
        normal_images.append(normal_img)

    for mi_file in mi_files:
        mi_img = load_image(os.path.join(mi_path, mi_file))
        mi_img = cv2.resize(mi_img, target_size)
        mi_img = mi_img / 255.0
        mi_images.append(mi_img)

    normal_images = np.array(normal_images)
    mi_images = np.array(mi_images)

    return normal_images, mi_images


def build_autoencoder(input_shape=(128, 128, 1)):
    """ساخت Autoencoder برای تشخیص ناهنجاری"""
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


def create_overlay(image, mask, color=[255, 0, 0], alpha=0.5):
    """ایجاد تصویر با ماسک رنگی نیمه شفاف"""
    image_uint8 = (image * 255).astype(np.uint8)
    overlay = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    idx = mask.astype(bool)
    overlay[idx] = (np.array(color) * alpha + overlay[idx]
                    * (1 - alpha)).astype(np.uint8)
    return overlay


def display_results_grid(normal_imgs, mi_imgs, lesion_masks, anomaly_maps_norm, n_rows=3):
    """نمایش نتایج به صورت شبکه‌ای با ۵ ستون و ۳ ردیف"""
    num_samples = n_rows  # هر ردیف یک نمونه است

    fig, axes = plt.subplots(n_rows, 5, figsize=(25, 15))
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    columns = ["Normal MRI", "MI MRI", "MI with Lesion Overlay",
               "Lesion Heatmap 1", "Lesion Heatmap 2"]

    for i in range(num_samples):
        idx = i  # اندیس تصویر
        normal_img = normal_imgs[idx]  # بدون تبدیل به uint8
        mi_img = mi_imgs[idx]  # بدون تبدیل به uint8
        lesion_mask = lesion_masks[idx].reshape(128, 128)
        anomaly_map = anomaly_maps_norm[idx].reshape(128, 128)

        # ستون ۱: تصویر MRI نرمال
        axes[i, 0].imshow(normal_img, cmap='gray')
        axes[i, 0].set_title(columns[0])
        axes[i, 0].axis('off')

        # ستون ۲: تصویر MI
        axes[i, 1].imshow(mi_img, cmap='gray')
        axes[i, 1].set_title(columns[1])
        axes[i, 1].axis('off')

        # ستون ۳: تصویر MI با ناحیه آسیب‌دیده رنگ‌آمیزی شده
        mi_with_lesion = create_overlay(
            mi_img, lesion_mask, color=[0, 255, 0], alpha=0.5)
        axes[i, 2].imshow(mi_with_lesion)
        axes[i, 2].set_title(columns[2])
        axes[i, 2].axis('off')

        # ستون ۴: نقشه حرارتی ناحیه آسیب‌دیده (نقشه ناهنجاری ۱)
        im1 = axes[i, 3].imshow(anomaly_map, cmap='hot')
        axes[i, 3].set_title(columns[3])
        axes[i, 3].axis('off')
        fig.colorbar(im1, ax=axes[i, 3], fraction=0.046, pad=0.04)

        # ستون ۵: نقشه حرارتی ناحیه آسیب‌دیده (نقشه ناهنجاری ۲)
        im2 = axes[i, 4].imshow(anomaly_map, cmap='jet')
        axes[i, 4].set_title(columns[4])
        axes[i, 4].axis('off')
        fig.colorbar(im2, ax=axes[i, 4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def compute_lesion_masks(mi_imgs, reconstructed_imgs, threshold=0.5):
    """محاسبه ماسک‌های ناحیه آسیب‌دیده و نقشه ناهنجاری"""
    anomaly_maps = np.abs(mi_imgs - reconstructed_imgs)
    anomaly_maps_norm = (anomaly_maps - anomaly_maps.min()) / \
        (anomaly_maps.max() - anomaly_maps.min())
    lesion_masks = (anomaly_maps_norm > threshold).astype(np.uint8)
    return lesion_masks, anomaly_maps_norm


# آماده‌سازی داده‌ها
normal_images, mi_images = prepare_data(normal_path, mi_path)

# افزودن بعد کانال
X_normal = np.expand_dims(normal_images, axis=-1)
X_mi = np.expand_dims(mi_images, axis=-1)

# تقسیم داده‌های سالم به آموزشی و آزمایشی
X_train_normal, X_test_normal = train_test_split(
    X_normal, test_size=0.2, random_state=42)

# ساخت و آموزش Autoencoder
autoencoder = build_autoencoder(input_shape=(128, 128, 1))
autoencoder.summary()

checkpoint = ModelCheckpoint(
    "autoencoder_model.keras", monitor="val_loss", save_best_only=True)
early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True)

history = autoencoder.fit(X_train_normal, X_train_normal,
                          epochs=50, batch_size=16,
                          validation_data=(X_test_normal, X_test_normal),
                          callbacks=[checkpoint, early_stop])

# بازسازی تصاویر MI توسط Autoencoder
reconstructed_mi = autoencoder.predict(X_mi)

# محاسبه ماسک‌های ناحیه آسیب‌دیده و نقشه ناهنجاری
lesion_masks, anomaly_maps_norm = compute_lesion_masks(
    X_mi, reconstructed_mi, threshold=0.5)

# نمایش نتایج به صورت شبکه‌ای
display_results_grid(
    normal_images,  # بدون تبدیل به uint8
    mi_images,  # بدون تبدیل به uint8
    lesion_masks,
    anomaly_maps_norm,
    n_rows=3  # تعداد ردیف‌ها (نمونه‌ها)
)
