import os

# 定义停用词文件路径
resources_dir = os.path.dirname(os.path.abspath(__file__))
stopwords_files = [
    os.path.join(resources_dir, "哈工大停用词表.txt"),
    os.path.join(resources_dir, "四川大学机器智能实验室停用词库.txt"),
    os.path.join(resources_dir, "中文停用词库.txt")
]
output_file = os.path.join(resources_dir, "stopwords.txt")

# 读取并合并停用词
all_stopwords = set()
for file_path in stopwords_files:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取每一行，去除空白字符
            words = [line.strip() for line in f if line.strip()]
            all_stopwords.update(words)
        print(f"已加载: {os.path.basename(file_path)}")
    else:
        print(f"文件不存在: {file_path}")

# 写入合并后的停用词
with open(output_file, 'w', encoding='utf-8') as f:
    # 排序后写入，确保结果一致
    for word in sorted(all_stopwords):
        f.write(word + '\n')

print(f"合并完成！共 {len(all_stopwords)} 个停用词，已写入: {os.path.basename(output_file)}")