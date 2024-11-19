# 导入所需库
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载训练数据集
training_set_file = r'C:\Users\mihao\Desktop\数学建模\B题\training_set_2ap_loc0_nav82.csv'
training_data = pd.read_csv(training_set_file)

# 选择特征
features = ['pkt_len', 'pd', 'ed', 'nav', 'sta_from_sta_0_rssi', 'sta_from_sta_1_rssi',
            'per', 'ppdu_dur', 'other_air_time', 'throughput']

# 填充缺失值
training_data['sta_from_sta_0_rssi'].fillna(training_data['sta_from_sta_0_rssi'].median(), inplace=True)
training_data['sta_from_sta_1_rssi'].fillna(training_data['sta_from_sta_1_rssi'].median(), inplace=True)

# 提取特征 (X) 和目标变量 (y)
X = training_data[features]
y_mcs = pd.to_numeric(training_data['mcs'], errors='coerce')  # MCS是调制编码方案
y_nss = pd.to_numeric(training_data['nss'], errors='coerce')  # NSS是空间流数

# 对特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据为训练集和测试集
X_train, X_test, y_train_mcs, y_test_mcs = train_test_split(X_scaled, y_mcs, test_size=0.2, random_state=42)
X_train_nss, X_test_nss, y_train_nss, y_test_nss = train_test_split(X_scaled, y_nss, test_size=0.2, random_state=42)

# 使用网格搜索调整超参数
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# 训练随机森林分类模型用于MCS预测
model_mcs = RandomForestClassifier(random_state=42)
grid_search_mcs = GridSearchCV(estimator=model_mcs, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_mcs.fit(X_train, y_train_mcs)

# 训练随机森林分类模型用于NSS预测
model_nss = RandomForestClassifier(random_state=42)
grid_search_nss = GridSearchCV(estimator=model_nss, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search_nss.fit(X_train_nss, y_train_nss)

# 获取最佳参数的模型
best_model_mcs = grid_search_mcs.best_estimator_
best_model_nss = grid_search_nss.best_estimator_

# 使用训练好的模型进行预测
y_pred_mcs = best_model_mcs.predict(X_test)
y_pred_nss = best_model_nss.predict(X_test_nss)

# 计算MCS和NSS的预测准确率
accuracy_mcs = accuracy_score(y_test_mcs, y_pred_mcs)
accuracy_nss = accuracy_score(y_test_nss, y_pred_nss)

print(f"MCS预测准确率: {accuracy_mcs}")
print(f"NSS预测准确率: {accuracy_nss}")

# 加载测试集1：test_set_2ap 和 test_set_3ap
test_set_2ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\test_set_2_2ap.csv'
test_set_2ap = pd.read_csv(test_set_2ap_file)

test_set_3ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\test_set_2_3ap.csv'
test_set_3ap = pd.read_csv(test_set_3ap_file)

# 确保测试集具有与训练集相同的特征列
X_test_2ap = scaler.transform(test_set_2ap[features])
X_test_3ap = scaler.transform(test_set_3ap[features])

# 预测2个AP和3个AP测试集上的MCS和NSS
y_pred_2ap_mcs = best_model_mcs.predict(X_test_2ap)
y_pred_2ap_nss = best_model_nss.predict(X_test_2ap)

y_pred_3ap_mcs = best_model_mcs.predict(X_test_3ap)
y_pred_3ap_nss = best_model_nss.predict(X_test_3ap)

# 将预测结果加入到测试集
test_set_2ap['predicted_mcs'] = y_pred_2ap_mcs
test_set_2ap['predicted_nss'] = y_pred_2ap_nss

test_set_3ap['predicted_mcs'] = y_pred_3ap_mcs
test_set_3ap['predicted_nss'] = y_pred_3ap_nss

# 输出预测结果
print("2个AP的测试集预测结果：")
print(test_set_2ap[['predicted_mcs', 'predicted_nss']].head())

print("\n3个AP的测试集预测结果：")
print(test_set_3ap[['predicted_mcs', 'predicted_nss']].head())

# 将预测结果保存为CSV文件
output_2ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\predicted_test_set_2_2ap.csv'
output_3ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\predicted_test_set_2_3ap.csv'

test_set_2ap.to_csv(output_2ap_file, index=False)
test_set_3ap.to_csv(output_3ap_file, index=False)

print(f"预测结果已保存至: {output_2ap_file} 和 {output_3ap_file}")
# 导入所需库
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载训练数据集
training_set_file = r'C:\Users\mihao\Desktop\数学建模\B题\training_set_2ap_loc0_nav82.csv'
training_data = pd.read_csv(training_set_file)

# 选择特征
features = ['pkt_len', 'pd', 'ed', 'nav', 'sta_from_sta_0_rssi', 'sta_from_sta_1_rssi',
            'per', 'ppdu_dur', 'other_air_time', 'throughput']

# 填充缺失值
training_data['sta_from_sta_0_rssi'].fillna(training_data['sta_from_sta_0_rssi'].median(), inplace=True)
training_data['sta_from_sta_1_rssi'].fillna(training_data['sta_from_sta_1_rssi'].median(), inplace=True)

# 提取特征 (X) 和目标变量 (y)
X = training_data[features]
y_mcs = training_data['mcs']  # MCS是调制编码方案
y_nss = training_data['nss']  # NSS是空间流数

# 将目标变量转换为数值类型（以确保分类任务）
y_mcs = pd.to_numeric(y_mcs, errors='coerce')
y_nss = pd.to_numeric(y_nss, errors='coerce')

# 分割数据为训练集和测试集
X_train, X_test, y_train_mcs, y_test_mcs = train_test_split(X, y_mcs, test_size=0.2, random_state=42)
X_train_nss, X_test_nss, y_train_nss, y_test_nss = train_test_split(X, y_nss, test_size=0.2, random_state=42)

# 训练随机森林分类模型用于MCS预测
model_mcs = RandomForestClassifier(n_estimators=100, random_state=42)
model_mcs.fit(X_train, y_train_mcs)

# 训练随机森林分类模型用于NSS预测
model_nss = RandomForestClassifier(n_estimators=100, random_state=42)
model_nss.fit(X_train_nss, y_train_nss)

# 预测测试集上的MCS
y_pred_mcs = model_mcs.predict(X_test)
accuracy_mcs = accuracy_score(y_test_mcs, y_pred_mcs)
print(f"MCS预测准确率: {accuracy_mcs}")

# 预测测试集上的NSS
y_pred_nss = model_nss.predict(X_test_nss)
accuracy_nss = accuracy_score(y_test_nss, y_pred_nss)
print(f"NSS预测准确率: {accuracy_nss}")

# 加载测试集1：test_set_2ap 和 test_set_3ap
test_set_2ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\test_set_2_2ap.csv'
test_set_2ap = pd.read_csv(test_set_2ap_file)

test_set_3ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\test_set_2_3ap.csv'
test_set_3ap = pd.read_csv(test_set_3ap_file)

# 确保测试集具有与训练集相同的特征列
X_test_2ap = test_set_2ap[features]
X_test_3ap = test_set_3ap[features]

# 预测2个AP和3个AP的测试集上的MCS和NSS
y_pred_2ap_mcs = model_mcs.predict(X_test_2ap)
y_pred_2ap_nss = model_nss.predict(X_test_2ap)

y_pred_3ap_mcs = model_mcs.predict(X_test_3ap)
y_pred_3ap_nss = model_nss.predict(X_test_3ap)

# 将预测结果加入到测试集
test_set_2ap['predicted_mcs'] = y_pred_2ap_mcs
test_set_2ap['predicted_nss'] = y_pred_2ap_nss

test_set_3ap['predicted_mcs'] = y_pred_3ap_mcs
test_set_3ap['predicted_nss'] = y_pred_3ap_nss

# 输出预测结果
print("2个AP的测试集预测结果：")
print(test_set_2ap[['predicted_mcs', 'predicted_nss']].head())

print("\n3个AP的测试集预测结果：")
print(test_set_3ap[['predicted_mcs', 'predicted_nss']].head())



# 保存预测结果到CSV文件
output_2ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\predicted_mcs_nss_test_set_2_2ap.csv'
output_3ap_file = r'C:\Users\mihao\Desktop\数学建模\B题\predicted_mcs_nss_test_set_2_3ap.csv'

test_set_2ap.to_csv(output_2ap_file, index=False)
test_set_3ap.to_csv(output_3ap_file, index=False)

print(f"预测MCS和NSS结果已保存至: {output_2ap_file} 和 {output_3ap_file}")
