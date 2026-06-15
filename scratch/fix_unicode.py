path = 'api/main.py'
content = open(path, 'rb').read()

replacements = [
    # 1. replace \\u093\\u0940 with \\u0939\\u0940
    (b'\\u093\\u0940', b'\\u0939\\u0940'),
    # 2. replace \\u093hash with \\u0939
    (b'\\u093hash', b'\\u0939'),
    # 3. replace \\u090\\u0924 with \\u0909\\u0924
    (b'\\u090\\u0924', b'\\u0909\\u0924'),
]

modified = False
for broken, fixed in replacements:
    if broken in content:
        count = content.count(broken)
        print(f"Replacing {count} occurrences of {broken} -> {fixed}")
        content = content.replace(broken, fixed)
        modified = True

if modified:
    open(path, 'wb').write(content)
    print("All replacements done!")
else:
    print("No replacements matched.")
