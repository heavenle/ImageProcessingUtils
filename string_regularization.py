import re


def _clean_illegal_characters(text):
    """
        清理Unicode字符。
    Args:
        text:字符串文本

    Returns:

    """
    # 正则表达式匹配多种零宽度字符
    cleaned_text = re.sub('[\u200a-\u200f\u202a-\u202e\u2009]', '', text)
    # print(cleaned_text)
    return cleaned_text


def _clean_chemical_text(text):
    """
        ​​删除字符串末尾括号及其内部内容​​，仅处理字符串末尾的括号内容，中间的括号不受影响。
    Args:
        text:

    Returns:

    """
    # 处理生成物，反应物文本中后面的括号，及其里面的内容
    if len(text.rstrip().rsplit(' ', maxsplit=1)) > 1:
        text = re.sub(r"\s*\([^)]*\)$", "", text.rstrip())  # 小括号

    if text.rfind('[') != -1 and text[text.rfind('[') - 1] == " ":  # 中括号
        text = text[0:text.rfind('[') - 1]

    # 处理生成物，反应物文本中指代数字（未用括号、中括号括起来的情况）
    text_list = text.rstrip().rsplit(' ', maxsplit=1)
    if len(text_list) > 1:
        num = text_list[-1]
        try:
            num = int(num)
            text = text_list[0]
        except Exception as e:
            text = ' '.join(text_list)

        # 去除"complex"字符
        if "Complexes" in text:
            text = text.replace("Complexes", "")
        elif "complexes" in text:
            text = text.replace("complexes", "")
        elif "complex" in text:
            text = text.replace("complex", "")
        elif "Complex" in text:
            text = text.replace("Complex", "")

    return text.rstrip()

def _clean_yelid_text(text):
    """
    清理产率字符串中冗余的内容，只保留产率。如果有多个产率，则用';'连接。
    例如：['hello world 98%'] 处理后变成[98%]
         ['jack 6% and mark 18%'] 处理后变成['6%;18%']
    Args:
        text:产率字符串

    Returns:

    """
    if "and" in text:
        text = text.split('and')
    else:
        text = [text]
    result = []
    for i in text:
        # 定义正则表达式
        pattern = r'\d+\.?\d*%'
        # 查找所有匹配的百分数
        result += re.findall(pattern, i)

    return ";".join(result)