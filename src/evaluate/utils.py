import re


def extract_outer_braces(s):
    result = []
    count = 0
    start = None
    
    for i, char in enumerate(s):
        if char == '{':
            if count == 0:
                start = i
            count += 1
        elif char == '}':
            count -= 1
            if count == 0:
                result.append(s[start:i+1])
                start = None
            elif count < 0:
                count = 0
    
    return result

def fix_json(json_str):
    # 匹配键名或字符串值中的单引号
    pattern = r'(\'[^\']*?\')'
    # 替换单引号为双引号
    fixed_json = re.sub(pattern, lambda match: '"' + match.group(0)[1:-1] + '"', json_str)
    return fixed_json