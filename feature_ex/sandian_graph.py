import numpy as np
import matplotlib.pyplot as plt

# 假设有5个类别，每个类别对应一个权重
weights = np.random.randn(5)  # 随机生成权重

# 模拟分类结果，这里假设为随机值
results = np.random.rand(5)

# 绘制散点图
plt.scatter(weights, results)

# 添加标题和标签
plt.title('Scatter Plot of Classification Results by Weight Magnitude')
plt.xlabel('Weight')
plt.ylabel('Classification Result')

plt.show()