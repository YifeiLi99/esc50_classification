import os
import torch
from data.esc50_dataset import ESC50Dataset
from models.cnn_model import SimpleCNN
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 44100

# 加载类别索引
DATASET = ESC50Dataset("./data/train_files.txt")
idx2label = {idx: label for label, idx in DATASET.class2idx.items()}

# 英文类别 → 中文类别映射
category_zh = {
    "dog": "狗",
    "rooster": "公鸡",
    "pig": "猪",
    "cow": "奶牛",
    "frog": "青蛙",
    "cat": "猫",
    "hen": "母鸡",
    "insects": "昆虫",
    "sheep": "羊",
    "crow": "乌鸦",
    "rain": "雨声",
    "sea_waves": "海浪",
    "crackling_fire": "篝火",
    "crickets": "蟋蟀",
    "chirping_birds": "鸟鸣",
    "water_drops": "水滴声",
    "wind": "风声",
    "pouring_water": "倒水声",
    "toilet_flush": "冲厕所声",
    "thunderstorm": "雷暴",
    "crying_baby": "婴儿哭声",
    "sneezing": "打喷嚏",
    "clapping": "鼓掌",
    "breathing": "呼吸声",
    "coughing": "咳嗽",
    "footsteps": "脚步声",
    "laughing": "笑声",
    "brushing_teeth": "刷牙声",
    "snoring": "打鼾",
    "drinking_sipping": "喝水声",
    "door_wood_knock": "敲门声",
    "mouse_click": "鼠标点击",
    "keyboard_typing": "键盘打字",
    "door_wood_creaks": "门吱呀声",
    "can_opening": "开罐声",
    "washing_machine": "洗衣机声",
    "vacuum_cleaner": "吸尘器声",
    "clock_alarm": "闹钟",
    "clock_tick": "时钟滴答声",
    "glass_breaking": "玻璃破碎",
    "helicopter": "直升机",
    "chainsaw": "电锯",
    "siren": "警笛声",
    "car_horn": "汽车喇叭",
    "engine": "发动机声",
    "train": "火车",
    "church_bells": "教堂钟声",
    "airplane": "飞机",
    "fireworks": "烟花声",
    "hand_saw": "锯木头声"
}

# 加载模型
model = SimpleCNN(n_classes=len(DATASET.labels))
model.load_state_dict(torch.load("./weights/epoch_30.pth", map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
model.eval()

# torchaudio特征处理器
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

# 推理函数
def predict_single_file(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    mel_spec = mel_spectrogram(waveform)
    log_mel_spec = amplitude_to_db(mel_spec).squeeze(0)

    with torch.no_grad():
        input_tensor = log_mel_spec.unsqueeze(0).to(DEVICE)
        outputs = model(input_tensor)
        preds = outputs.argmax(dim=1).cpu().item()

    predicted_label = idx2label[preds]
    zh_label = category_zh.get(predicted_label, predicted_label)
    return predicted_label, zh_label

if __name__ == "__main__":
    test_audio = "./data/test/1-7974-A-49.wav"
    en_label, zh_label = predict_single_file(test_audio)
    print(f"文件: {test_audio} | 预测类别: {en_label} ({zh_label})")
