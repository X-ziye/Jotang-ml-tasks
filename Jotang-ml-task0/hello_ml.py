import numpy as np  # 导入NumPy库，用来做矩阵运算

# ================= 第一部分：字典 + 函数 =================

# 1. 用字典记录3个学生的姓名和成绩
scores = {
    "小明": 90,
    "小红": 85,
    "小刚": 95
}
print("成绩单:", scores)

# 2. 写一个函数计算平均分并找出最高分
def analyze_scores(score_dict):
    # 计算平均分：所有成绩相加，除以人数
    avg_score = sum(score_dict.values()) / len(score_dict)
    
    # 找最高分和对应的人
    max_score = 0
    top_student = ""
    for name, score in score_dict.items():
        if score > max_score:
            max_score = score
            top_student = name
            
    return avg_score, top_student, max_score

# 调用函数并接收结果
avg, top_name, top_score = analyze_scores(scores)
print("平均分:", avg)
print("最高分:", top_name, "成绩:", top_score)


# ================= 第二部分：NumPy 矩阵乘法 =================

# 3. 使用 NumPy 创建两个形状合适的矩阵
# 创建 3x4 的矩阵 A（3个样本，4个特征）
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# 创建 4x2 的矩阵 B（4个输入，2个输出）
B = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
])

# 4. 输出计算结果以及输入、输出矩阵的形状
print("\n--- NumPy 矩阵运算 ---")
print("矩阵A的形状:", A.shape)  # 应该是 (3, 4)
print("矩阵B的形状:", B.shape)  # 应该是 (4, 2)

# 完成矩阵乘法：A 乘 B
C = np.dot(A, B) # 或者写 C = A @ B
print("矩阵相乘结果C:")
print(C)
print("矩阵C的形状:", C.shape)  # 重点看这里，应该是 (3, 2)