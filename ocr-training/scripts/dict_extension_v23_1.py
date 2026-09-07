import os
import json
import hashlib

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    old_dict_path = 'data/cham_dict_v23.txt'
    new_dict_path = 'data/cham_dict_v23_1.txt'
    report_path = 'output/v23_1_training/dict_extension_report.json'
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # 1. Read old dictionary
    with open(old_dict_path, 'r', encoding='utf-8') as f:
        old_lines = f.read().splitlines()
    
    # Remove empty lines
    old_chars = [line for line in old_lines if line]
    old_dict_size = len(old_chars)
    old_dict_sha256 = sha256(old_dict_path)
    
    # 2. Define required extensions
    required_new_chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '[', ']', '(', ')', '.', ',', ':', ';', '-', '/'
    ]
    
    # We find which ones are already in old_chars
    added_chars = []
    for c in required_new_chars:
        if c not in old_chars:
            added_chars.append(c)
            
    # Preserve the exact list from USER request:
    # 0 1 2 3 4 5 6 7 8 9 [] () . , : ; - /
    # ':' is already present in V23 dictionary at line 1.
    
    # 3. Create new dictionary
    new_chars = old_chars + added_chars
    new_dict_size = len(new_chars)
    
    with open(new_dict_path, 'w', encoding='utf-8', newline='\n') as f_out:
        for c in new_chars:
            f_out.write(c + '\n')
            
    new_dict_sha256 = sha256(new_dict_path)
    
    # 4. Verify all required characters are present in new_chars
    missing_required = [c for c in required_new_chars if c not in new_chars]
    
    report = {
        "old_dict_size": old_dict_size,
        "new_dict_size": new_dict_size,
        "added_chars": added_chars,
        "missing_required_chars": missing_required,
        "old_dict_sha256": old_dict_sha256,
        "new_dict_sha256": new_dict_sha256
    }
    
    with open(report_path, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=2)
        
    print("✅ Extended dictionary generated at:", new_dict_path)
    print("   Old size:", old_dict_size)
    print("   New size:", new_dict_size)
    print("   Added:", added_chars)
    print("   Missing:", missing_required)
    print("Report saved to:", report_path)

if __name__ == '__main__':
    main()
