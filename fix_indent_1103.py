from pathlib import Path

test_file = Path('src/kisansathi/orchestration/graph.py')
content = test_file.read_text(encoding='utf-8')
lines = content.split('\n')

# Fix line 1103 (index 1102): graph.add_node("speech_to_text", speech_to_text_node)
# Should be indented by 4 spaces
if len(lines) > 1102 and lines[1102].startswith('graph.add_node("speech_to_text"'):
    lines[1102] = '    ' + lines[1102].lstrip(' ')

with open('src/kisansathi/orchestration/graph.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Fixed indentation')