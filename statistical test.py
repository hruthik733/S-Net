import numpy as np
import pandas as pd
from scipy import stats
import scipy.stats as st
from scipy.stats import wilcoxon

# 自定义格式化函数，使用 × 和上标数字表示科学计数法
def format_p_value(p):
    if p <= 0.001:
        # 将 p 值转换为科学计数法形式，如 3.00×10⁻³
        exp = int(np.floor(np.log10(p)))
        coeff = p / (10 ** exp)
        return f"{coeff:.1f}×10^{exp}"  # 使用 Unicode 上标字符
    else:
        return f"{p:.4f}"

# 读取Excel文件
data = pd.read_excel('/home/hp/Code/SQH/GeneralArch/Glas-IOU.xlsx')

# 提取第一列和第二列的数据
model_a_scores = data.iloc[:, 0]  # 第一列，模型A的数据
model_b_scores = data.iloc[:, 14]  # 第二列，模型B的数据
print(model_b_scores)

# 将数据转换为NumPy数组
model_a_scores = model_a_scores.to_numpy()
model_b_scores = model_b_scores.to_numpy()
# print(model_a_scores)

# 计算平均值和标准差
mean_a = np.mean(model_a_scores)
std_a = np.std(model_a_scores, ddof=1)
mean_b = np.mean(model_b_scores)
std_b = np.std(model_b_scores, ddof=1)

print(f"您的模型的Dice系数：平均值 = {mean_a:.4f}, 标准差 = {std_a:.4f}")
print(f"对比模型的Dice系数：平均值 = {mean_b:.4f}, 标准差 = {std_b:.4f}")
print("-"*50)

# 计算差值
difference = model_a_scores - model_b_scores

# 正态性检验
w_statistic, p_value_normality = stats.shapiro(difference)
p_norm_display = format_p_value(p_value_normality)
print(f"Shapiro-Wilk 正态性检验: W统计量 = {w_statistic:.4f}, p值 = {p_norm_display}")
print("-"*50)

if p_value_normality > 0.05:
    print(f"符合正态分布。")
    t_statistic, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
    p_display = format_p_value(p_value)
    print(f"配对 t 检验: t统计量 = {t_statistic:.4f}, p值 = {p_display}")
else:
    print(f"不符合正态分布")
    w_statistic, p_value = wilcoxon(model_a_scores, model_b_scores)
    p_display = format_p_value(p_value)
    print(f"Wilcoxon 符号秩检验: 统计量 = {w_statistic:.4f}, p值 = {p_display}")
print("-"*50)

# 计算均值和置信区间
def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    sem = st.sem(data)
    margin = sem * st.t.ppf((1 + confidence) / 2., len(data)-1)
    return mean, mean - margin, mean + margin

mean_a, ci_lower_a, ci_upper_a = mean_confidence_interval(model_a_scores)
mean_b, ci_lower_b, ci_upper_b = mean_confidence_interval(model_b_scores)

print(f"您的方法的Dice系数: 平均值 = {mean_a:.4f}, 95%置信区间 = [{ci_lower_a:.4f}, {ci_upper_a:.4f}]")
print(f"比较方法的Dice系数: 平均值 = {mean_b:.4f}, 95%置信区间 = [{ci_lower_b:.4f}, {ci_upper_b:.4f}]")