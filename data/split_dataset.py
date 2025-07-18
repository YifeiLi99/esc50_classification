import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# 配置路径
# 当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
CSV_PATH = os.path.join(BASE_DIR, "ESC-50-master", "meta", "esc50.csv")
print(CSV_PATH)
AUDIO_DIR = os.path.join(BASE_DIR, "ESC-50-master", "audio")
print(AUDIO_DIR)

# 输出文件路径
TARGET_DIR = "./"
TRAIN_DIR = os.path.join(TARGET_DIR, "train")
VAL_DIR = os.path.join(TARGET_DIR, "val")
TEST_DIR = os.path.join(TARGET_DIR, "test")

TRAIN_LIST_PATH = os.path.join(TARGET_DIR, "train_files.txt")
VAL_LIST_PATH = os.path.join(TARGET_DIR, "val_files.txt")
TEST_LIST_PATH = os.path.join(TARGET_DIR, "test_files.txt")

# 确保输出文件夹存在
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 读取元数据
metadata = pd.read_csv(CSV_PATH)

# 按 fold == 1 做 test
print("\n按fold划分：fold==1为test，剩余做train/val")
test_df = metadata[metadata['fold'] == 1]
trainval_df = metadata[metadata['fold'] != 1]

# 从 trainval_df 划分 train/val（8:2）
train_df, val_df = train_test_split(
    trainval_df,
    test_size=0.2,
    stratify=trainval_df['category'],
    random_state=42
)

# 搬运文件的函数
def copy_files(file_list, target_folder):
    for fname in file_list:
        src = os.path.join(AUDIO_DIR, fname)
        dst = os.path.join(target_folder, fname)
        shutil.copy(src, dst)

# 搬运音频文件
copy_files(train_df['filename'], TRAIN_DIR)
copy_files(val_df['filename'], VAL_DIR)
copy_files(test_df['filename'], TEST_DIR)

print(f"音频文件已移动至 train({len(train_df)}), val({len(val_df)}), test({len(test_df)}) 文件夹")

# 生成完整路径列表
train_files = [os.path.abspath(os.path.join(TRAIN_DIR, fname)) for fname in train_df['filename']]
val_files = [os.path.abspath(os.path.join(VAL_DIR, fname)) for fname in val_df['filename']]
test_files = [os.path.abspath(os.path.join(TEST_DIR, fname)) for fname in test_df['filename']]

with open(TRAIN_LIST_PATH, 'w') as f:
    f.write('\n'.join(train_files))

with open(VAL_LIST_PATH, 'w') as f:
    f.write('\n'.join(val_files))

with open(TEST_LIST_PATH, 'w') as f:
    f.write('\n'.join(test_files))

print("文件路径索引已生成：train_files.txt, val_files.txt, test_files.txt")
