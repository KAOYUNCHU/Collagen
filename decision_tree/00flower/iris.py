from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 載入資料集
iris = load_iris()
X = iris.data      # 特徵
y = iris.target    # 標籤（花的種類）

# 2. 分成訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 建立決策樹模型（限制深度 = 3）
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. 預測並計算準確率
y_pred = model.predict(X_test)
print("測試集準確率:", accuracy_score(y_test, y_pred))

# 5. 視覺化樹
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
