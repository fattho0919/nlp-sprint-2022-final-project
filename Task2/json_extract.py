import json
import time

def extract_jsons(text):
    jsons = []
    start_index = 0
    repeat_count = 0

    while True:

        seg_start = text.find("[", start_index)
        seg_end = text.find("]", seg_start)

        json_start = text.find("{", start_index)
        json_end = text.find("}", seg_end) + 1

        if json_start == -1 or json_end == -1:
            break

        json_content = text[json_start:json_end]
        json_content = json.loads(json_content, strict=False)
        print(json_content)
        time.sleep(1)
        if json_content not in jsons:
            jsons.append(json_content)

        start_index = json_end

    return jsons

# Example text with multiple JSON segments
filename = 'Task_2_MED _chatGPT_output.md'

with open(filename, 'r', encoding='utf-8') as file:
    contents = file.read()

json_segments = extract_jsons(contents)
print(len(json_segments))

with open ('output.json', 'w', encoding='utf-8-sig') as file:
    json.dump({'output':json_segments}, file, indent = 4, ensure_ascii = False, )