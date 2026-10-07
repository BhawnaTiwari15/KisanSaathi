from pathlib import Path

test_file = Path('src/kisansathi/orchestration/graph.py')
content = test_file.read_text(encoding='utf-8')
lines = content.split('\n')

# Fix the finalize_response function body (lines 1024-1110)
# The function body should be indented by 8 spaces (4 for function + 4 for body)
# Currently lines 1046-1110 have wrong indentation

# Lines 1046-1110 (1-indexed) = indices 1045-1109 (0-indexed)
for i in range(1045, 1110):
    if i < len(lines):
        line = lines[i]
        stripped = line.lstrip(' ')
        current_indent = len(line) - len(stripped)
        
        # Skip empty lines
        if not stripped:
            continue
        if stripped.startswith('#'):
            # Comments should be indented by 8 spaces
            if not line.startswith('        '):
                lines[i] = '        ' + stripped
            continue
        
        # Non-empty, non-comment lines should be indented by 8 spaces
        if current_indent < 8:
            lines[i] = '        ' + stripped
        elif current_indent > 8:
            # Remove excess indentation
            lines[i] = '        ' + stripped

with open('src/kisansathi/orchestration/graph.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Fixed finalize_response indentation')