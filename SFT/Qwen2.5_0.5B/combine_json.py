import os
import json
import glob
import re

# 目标目录
DATA_DIR = "/home/rickc/Projects/LLMs/data/SFT"
OUTPUT_FILE = "merged_output.json"


def process_prompt_respond(file_path):
    """处理包含 'prompt' 和 'respond' 的 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    entry = {
                        "instruction": obj.get("prompt", "").strip(),
                        "input": "",
                        "output": obj.get("respond", "").strip(),
                        "system": "",
                        "history": obj.get("history", []),
                    }
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"解析 {file_path} 中的行时出错: {e}")
    return data


def process_text_chat(file_path):
    """处理包含 'text' 字段的 JSONL 文件，解析 system, user, assistant"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")

                    # 使用正则表达式提取内容
                    system_match = re.search(
                        r"<\|im_start\|>system\n(.*?)<\|im_end\|>", text, re.DOTALL
                    )
                    user_match = re.search(
                        r"<\|im_start\|>user\n\"?(.*?)\"?<\|im_end\|>", text, re.DOTALL
                    )
                    assistant_match = re.search(
                        r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL
                    )

                    system = system_match.group(1).strip() if system_match else ""
                    instruction = user_match.group(1).strip() if user_match else ""
                    output = assistant_match.group(1).strip() if assistant_match else ""

                    entry = {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                        "system": system,
                        "history": [],
                    }
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"解析 {file_path} 中的行时出错: {e}")
    return data


def process_data_json(file_path):
    """处理 data.json 文件，确保其为有效的 JSON 数组"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        # 检查是否以 [ 开头和 ] 结尾，如果不是，则添加
        if not content.startswith("["):
            content = "[" + content
        if not content.endswith("]"):
            content = content + "]"
        try:
            objs = json.loads(content)
            for obj in objs:
                entry = {
                    "instruction": obj.get("instruction", "").strip(),
                    "input": obj.get("input", "").strip(),
                    "output": obj.get("output", "").strip(),
                    "system": "",
                    "history": [],
                }
                data.append(entry)
        except json.JSONDecodeError as e:
            print(f"解析 {file_path} 时出错: {e}")
    return data


def process_ruozhiba_post(file_path):
    """处理 ruozhiba-post-annual.json 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            objs = json.load(f)
            for obj in objs:
                content = obj.get("content", "").strip()
                # 去除开头的序号，例如 "151、"
                content_clean = re.sub(r"^\d+、", "", content)
                entry = {
                    "instruction": "给我讲一个笑话",
                    "input": "",
                    "output": content_clean,
                    "system": "",
                    "history": [],
                }
                data.append(entry)
        except json.JSONDecodeError:
            # 如果不是 JSON 数组，逐行解析
            f.seek(0)
            for line in f:
                line = line.strip().rstrip(",")
                if line and line != "{" and line != "}":
                    try:
                        obj = json.loads(line)
                        content = obj.get("content", "").strip()
                        # 去除开头的序号，例如 "151、"
                        content_clean = re.sub(r"^\d+、", "", content)
                        entry = {
                            "instruction": "给我讲一个笑话",
                            "input": "",
                            "output": content_clean,
                            "system": "",
                            "history": [],
                        }
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"解析 {file_path} 中的行时出错: {e}")
    return data


def process_ruozhiba_title_good(file_path):
    """处理 ruozhiba-title-good.json 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            objs = json.load(f)
            for obj in objs:
                title = obj.get("title", "").strip()
                abs_text = obj.get("abs", "").strip() if obj.get("abs") else ""
                # 组合 title 和 abs，用逗号分隔，忽略空字段
                output_parts = []
                if title:
                    output_parts.append(title)
                if abs_text:
                    output_parts.append(abs_text)
                output_combined = ", ".join(output_parts)
                entry = {
                    "instruction": "给我讲一个笑话",
                    "input": "",
                    "output": output_combined,
                    "system": "",
                    "history": [],
                }
                data.append(entry)
        except json.JSONDecodeError as e:
            print(f"解析 {file_path} 时出错: {e}")
    return data


def process_ruozhiba_title_norm(file_path):
    """处理 ruozhiba-title-norm.json 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            objs = json.load(f)
            for obj in objs:
                title = obj.get("title", "").strip()
                abs_text = obj.get("abs", "").strip() if obj.get("abs") else ""
                # 组合 title 和 abs，用逗号分隔，忽略空字段
                output_parts = []
                if title:
                    output_parts.append(title)
                if abs_text:
                    output_parts.append(abs_text)
                output_combined = ", ".join(output_parts)
                entry = {
                    "instruction": "给我讲一个笑话",
                    "input": "",
                    "output": output_combined,
                    "system": "",
                    "history": [],
                }
                data.append(entry)
        except json.JSONDecodeError as e:
            print(f"解析 {file_path} 时出错: {e}")
    return data


def process_wikihow(file_path):
    """处理 wikihow.jsonl 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    entry = {
                        "instruction": obj.get("prompt", "").strip(),
                        "input": "",
                        "output": obj.get("output", "").strip(),
                        "system": "",
                        "history": [],
                    }
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"解析 {file_path} 中的行时出错: {e}")
    return data


def process_xuexue_rain(file_path):
    """处理 xuexue_rain.jsonl 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    entry = {
                        "instruction": obj.get("prompt", "").strip(),
                        "input": "",
                        "output": obj.get("respond", "").strip(),
                        "system": "",
                        "history": obj.get("history", []),
                    }
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"解析 {file_path} 中的行时出错: {e}")
    return data


def main():
    merged_data = []

    # 遍历所有文件
    for file_path in glob.glob(os.path.join(DATA_DIR, "*")):
        filename = os.path.basename(file_path)
        if filename.endswith(".jsonl"):
            if (
                "COIG-CQIA" in filename
                or "wikihow" in filename
                or "xuexue_rain" in filename
            ):
                data = process_prompt_respond(file_path)
                merged_data.extend(data)
            elif any(x in filename for x in ["test", "train", "valid"]):
                data = process_text_chat(file_path)
                merged_data.extend(data)
            else:
                print(f"未识别的 JSONL 文件格式: {filename}")
        elif filename == "data.json":
            data = process_data_json(file_path)
            merged_data.extend(data)
        elif "ruozhiba-post-annual" in filename:
            data = process_ruozhiba_post(file_path)
            merged_data.extend(data)
        elif "ruozhiba-title-good" in filename:
            data = process_ruozhiba_title_good(file_path)
            merged_data.extend(data)
        elif "ruozhiba-title-norm" in filename:
            data = process_ruozhiba_title_norm(file_path)
            merged_data.extend(data)
        else:
            print(f"未处理的文件: {filename}")

    # 检查每个条目是否存在 'instruction' 和 'output'
    valid_entries = []
    invalid_entries = []
    for idx, entry in enumerate(merged_data, 1):
        if entry.get("instruction") and entry.get("output"):
            valid_entries.append(entry)
        else:
            invalid_entries.append((idx, entry))

    if invalid_entries:
        print(
            f"\n共找到 {len(invalid_entries)} 个无效条目（缺少 'instruction' 或 'output'）："
        )
        for idx, entry in invalid_entries:
            print(f"条目 {idx}: {entry}")
    else:
        print("\n所有条目均包含 'instruction' 和 'output'。")

    # 仅保存有效条目
    merged_data = valid_entries

    # 保存合并后的数据
    with open(os.path.join(DATA_DIR, OUTPUT_FILE), "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成，保存至 {OUTPUT_FILE}。共 {len(merged_data)} 个有效条目。")


if __name__ == "__main__":
    main()
