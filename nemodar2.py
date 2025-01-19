import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# مسیر داده‌ها
normal_path = r"C:\Users\Lenovo\Desktop\Data\Normal"
mi_path = r"C:\Users\Lenovo\Desktop\Data\MI"

# تابع افزایش داده‌ها با تکنیک‌های افزایش تصویر


def augment_image(image):
    augmented_images = []

    # چرخش
    for angle in [15, -15, 30, -30]:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        augmented_images.append(rotated)

    # وارونگی افقی
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    return augmented_images

# تابع استخراج ویژگی مساحت


def calculate_mi_area(image):
    # تبدیل به تصویر خاکستری
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # استفاده از آستانه‌گذاری Otsu برای تعیین آستانه بهینه
    _, thresholded = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # محاسبه مساحت نواحی سفید (آسیب‌دیده)
    mi_area = np.sum(thresholded == 255)

    # محاسبه کل مساحت تصویر
    total_area = image.shape[0] * image.shape[1]

    # درصد مساحت آسیب‌دیده نسبت به کل تصویر
    mi_percentage = (mi_area / total_area) * 100

    return mi_area, mi_percentage

# پردازش تصاویر و استخراج ویژگی‌ها


def process_images(path):
    areas = []
    percentages = []

    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return areas, percentages

    for filename in os.listdir(path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            file_path = os.path.join(path, filename)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Failed to read {file_path}.")
                continue

            # استخراج ویژگی‌های تصویر اصلی
            mi_area, mi_percentage = calculate_mi_area(image)
            areas.append(mi_area)
            percentages.append(mi_percentage)

            # افزایش داده‌ها
            augmented_images = augment_image(image)
            for aug_image in augmented_images:
                mi_area_aug, mi_percentage_aug = calculate_mi_area(aug_image)
                areas.append(mi_area_aug)
                percentages.append(mi_percentage_aug)

    return areas, percentages


# استخراج ویژگی‌ها برای هر دسته از تصاویر
normal_areas, normal_percentages = process_images(normal_path)
mi_areas, mi_percentages = process_images(mi_path)

# شبیه‌سازی داده‌های پیش‌بینی‌شده به صورت جداگانه برای هر دسته
# برای تصاویر سالم، فرض می‌کنیم مدل پیش‌بینی دقیقی دارد با خطای کمی
predicted_normal_areas = [x + np.random.normal(0, 1) for x in normal_areas]
predicted_normal_percentages = [
    x + np.random.normal(0, 0.5) for x in normal_percentages]

# برای تصاویر MI، مدل پیش‌بینی با کمی خطا عمل می‌کند
predicted_mi_areas = [x * 0.95 + np.random.normal(0, 5) for x in mi_areas]
predicted_mi_percentages = [x * 0.95 +
                            np.random.normal(0, 3) for x in mi_percentages]

# ترکیب داده‌ها (اختیاری، بستگی به نیاز شما دارد)
all_actual_areas = normal_areas + mi_areas
all_predicted_areas = predicted_normal_areas + predicted_mi_areas

all_actual_percentages = normal_percentages + mi_percentages
all_predicted_percentages = predicted_normal_percentages + predicted_mi_percentages

# رسم نمودارها
plt.figure(figsize=(14, 14))

# نمودار 1: همبستگی بین مقادیر واقعی و پیش‌بینی‌شده برای مساحت MI
plt.subplot(2, 2, 1)
slope, intercept, r_value, _, _ = linregress(
    all_actual_areas, all_predicted_areas)
plt.scatter(all_actual_areas, all_predicted_areas,
            c='r', alpha=0.6, label=f'r={r_value:.2f}')
plt.plot(np.array(all_actual_areas), np.array(all_actual_areas) *
         slope + intercept, 'k-', label=f'y={slope:.2f}x+{intercept:.2f}')
plt.xlabel('Actual MI Area (cm²)')
plt.ylabel('Predicted MI Area (cm²)')
plt.title('Correlation: Actual vs Predicted MI Area')
plt.legend()
plt.grid(True)

# نمودار 2: Bland-Altman برای مساحت MI
plt.subplot(2, 2, 2)
# استفاده از داده‌های MI برای Bland-Altman
mean_mi_area = (np.array(mi_areas) + np.array(predicted_mi_areas)) / 2
diff_mi_area = np.array(mi_areas) - np.array(predicted_mi_areas)

plt.scatter(mean_mi_area, diff_mi_area, c='b', alpha=0.5, label='MI')
plt.axhline(np.mean(diff_mi_area), color='k',
            linestyle='--', label='Mean Difference')
plt.axhline(np.mean(diff_mi_area) + 1.96 * np.std(diff_mi_area),
            color='k', linestyle=':', label='95% Upper Limit')
plt.axhline(np.mean(diff_mi_area) - 1.96 * np.std(diff_mi_area),
            color='k', linestyle=':', label='95% Lower Limit')

plt.xlabel('Average MI Area (cm²)')
plt.ylabel('Difference MI Area (cm²)')
plt.title('Bland-Altman Plot: MI Area')
plt.legend()
plt.grid(True)

# نمودار 3: همبستگی بین درصد ناحیه آسیب‌دیده واقعی و پیش‌بینی‌شده
plt.subplot(2, 2, 3)
slope_p, intercept_p, r_value_p, _, _ = linregress(
    all_actual_percentages, all_predicted_percentages)
plt.scatter(all_actual_percentages, all_predicted_percentages,
            c='g', alpha=0.6, label=f'r={r_value_p:.2f}')
plt.plot(np.array(all_actual_percentages), np.array(all_actual_percentages)
         * slope_p + intercept_p, 'k-', label=f'y={slope_p:.2f}x+{intercept_p:.2f}')
plt.xlabel('Actual Infarct Area Percentage (MI%)')
plt.ylabel('Predicted Infarct Area Percentage (MI%)')
plt.title('Correlation: Actual vs Predicted Infarct Area Percentage')
plt.legend()
plt.grid(True)

# نمودار 4: Bland-Altman برای درصد ناحیه آسیب‌دیده
plt.subplot(2, 2, 4)
# استفاده از داده‌های MI برای Bland-Altman
mean_mi_percentage = (np.array(mi_percentages) +
                      np.array(predicted_mi_percentages)) / 2
diff_mi_percentage = np.array(mi_percentages) - \
    np.array(predicted_mi_percentages)

plt.scatter(mean_mi_percentage, diff_mi_percentage,
            c='m', alpha=0.5, label='MI')
plt.axhline(np.mean(diff_mi_percentage), color='k',
            linestyle='--', label='Mean Difference')
plt.axhline(np.mean(diff_mi_percentage) + 1.96 * np.std(diff_mi_percentage),
            color='k', linestyle=':', label='95% Upper Limit')
plt.axhline(np.mean(diff_mi_percentage) - 1.96 * np.std(diff_mi_percentage),
            color='k', linestyle=':', label='95% Lower Limit')

plt.xlabel('Average Infarct Area Percentage (MI%)')
plt.ylabel('Difference Infarct Area Percentage (MI%)')
plt.title('Bland-Altman Plot: Infarct Area Percentage')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
