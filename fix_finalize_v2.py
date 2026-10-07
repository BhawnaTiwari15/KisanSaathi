from pathlib import Path

test_file = Path('src/kisansathi/orchestration/graph.py')
content = test_file.read_text(encoding='utf-8')
lines = content.split('\n')

# Fix the finalize_response function body (lines 1024-1110)
# The function body should be indented by 8 spaces (4 for function + 4 for body)
# Lines 1024-1110 (0-indexed: 1023-1109)

# First, let's check the current state
for i in range(1023, 1120):
    if i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip(' '))
        print(f'{i+1}: indent={len(line) - len(line.lstrip(" "))}, {line[:80]!r}')

# Fix: lines 1046-1110 (indices 1045-1109) should be indented by 8 spaces
# The function body starts at line 1025 (index 1024) with indent 8
# All body lines should have indent >= 8

for i in range(1024, 1110):
    if i < len(lines):
        line = lines[i]
        stripped = line.lstrip(' ')
        current_indent = len(line) - len(stripped)
        
        # Skip empty lines
        if not stripped:
            continue
        if line.strip().startswith('#'):
            # Comments should be at indent 8
            if current_indent < 8:
                lines[i] = '        ' + stripped
            continue
        
        # Non-empty, non-comment lines should be at indent 8
        if current_indent < 8:
            lines[i] = '        ' + stripped
        elif current_indent > 8:
            # Remove excess indentation
            lines[i] = '        ' + stripped

with open('src/kisansathi/orchestration/graph.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Fixed finalize_response indentation')