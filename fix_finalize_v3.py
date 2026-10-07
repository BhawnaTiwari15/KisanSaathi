from pathlib import Path

test_file = Path('src/kisansathi/orchestration/graph.py')
content = test_file.read_text(encoding='utf-8')
lines = content.split('\n')

# Rebuild the finalize_response function properly (lines 1024-1110)
# The function starts at line 1024 (index 1023) with indent=4
# Function body should be at indent 8 (4 for function + 4 for body)
# Nested if statements should be at indent 12, 16, etc.

# Let's rebuild the finalize_response function properly
new_lines = []
for i, line in enumerate(lines):
    if i < 1023:  # Before finalize_response function
        new_lines.append(line)
    elif i >= 1023 and i < 1115:  # Lines 1024-1114 (0-indexed: 1023-1109)
        # This is the finalize_response function body
        # Should be indented by 8 spaces (inside function)
        stripped = line.lstrip(' ')
        current_indent = len(line) - len(stripped)
        
        if not line.strip():
            new_lines.append(line)  # Keep empty lines as-is
        elif line.strip().startswith('#'):
            # Comments at indent 8
            new_lines.append('        ' + line.lstrip())
        else:
            # Code should be at indent 8
            new_lines.append('        ' + line.lstrip(' '))
    else:
        new_lines.append(line)

with open('src/kisansathi/orchestration/graph.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))
print('Rewrote finalize_response function')