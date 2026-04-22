import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('poison_results.csv')

# 清理数据：处理由于浮点数精度导致的微小差异
df['x_alpha'] = df['x_alpha'].round(2)
df['y_alpha'] = df['y_alpha'].round(2)

# 设置风格
# sns.set_theme(style="whitegrid")
plt.style.use('ggplot')

# 获取不同的 nums
unique_nums = df['nums'].unique()

# 创建画布，两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 图1：基于 x_alpha (feature randomization)
# 筛选 y_alpha == 0 的数据来展示 x_alpha 的影响
df_x = df[df['y_alpha'] == 0].sort_values('x_alpha')
for num in unique_nums:
    data = df_x[df_x['nums'] == num]
    ax1.plot(data['x_alpha'], data['acc'], marker='o', label=f'nums={num}')

ax1.set_title('Feature Randomization')
ax1.set_xlabel('x_alpha')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
ax1.legend()

# 图2：基于 y_alpha (label flipping)
# 筛选 x_alpha == 0 的数据来展示 y_alpha 的影响
df_y = df[df['x_alpha'] == 0].sort_values('y_alpha')
for num in unique_nums:
    data = df_y[df_y['nums'] == num]
    ax2.plot(data['y_alpha'], data['acc'], marker='s', label=f'nums={num}')

ax2.set_title('Label Flipping')
ax2.set_xlabel('y_alpha')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.legend()

plt.tight_layout()
plt.savefig('poison_curves.png')
print("图像已保存为 poison_curves.png")
# plt.show() # 如果在支持显示的终端下可以打开
