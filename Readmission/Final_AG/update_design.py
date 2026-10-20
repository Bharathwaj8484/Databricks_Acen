import os

enh_path = r"c:\Users\bhara\Downloads\Final_AG\Enhanced_UI_main.py"
orig_path = r"c:\Users\bhara\Downloads\Final_AG\Readmission_HF.py"

with open(enh_path, "r", encoding="utf-8") as f:
    enh_content = f.read()

with open(orig_path, "r", encoding="utf-8") as f:
    orig_content = f.read()

start_marker = "def get_patient_data(patient_id):"
end_marker = "</html>\n'''"

start_idx_enh = enh_content.find(start_marker)
end_idx_enh = enh_content.find(end_marker, start_idx_enh)
if start_idx_enh == -1 or end_idx_enh == -1:
    print("Could not find boundaries in Enhanced_UI_main.py")
    exit(1)

design_chunk = enh_content[start_idx_enh:end_idx_enh + len(end_marker)]

start_idx_orig = orig_content.find(start_marker)
end_idx_orig = orig_content.find(end_marker, start_idx_orig)
if start_idx_orig == -1 or end_idx_orig == -1:
    print("Could not find boundaries in Readmission_HF.py")
    exit(1)

new_content = orig_content[:start_idx_orig] + design_chunk + orig_content[end_idx_orig + len(end_marker):]

with open(orig_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully applied design changes to Readmission_HF.py!")
