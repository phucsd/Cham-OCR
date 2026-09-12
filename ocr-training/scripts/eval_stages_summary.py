import re
import os

def analyze_losses(path):
    epochs = {}
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r'epoch: \[(\d+)/\d+\], global_step: (\d+), lr: ([0-9.e-]+), CTCLoss: ([0-9.]+), NRTRLoss: ([0-9.]+), loss: ([0-9.]+)', line)
            if m:
                ep = int(m.group(1))
                ctc = float(m.group(4))
                nrtr = float(m.group(5))
                loss = float(m.group(6))
                if ep not in epochs:
                    epochs[ep] = {'ctc': [], 'nrtr': [], 'loss': []}
                epochs[ep]['ctc'].append(ctc)
                epochs[ep]['nrtr'].append(nrtr)
                epochs[ep]['loss'].append(loss)
    for ep in sorted(epochs.keys()):
        c = epochs[ep]
        avg_loss = sum(c['loss']) / len(c['loss'])
        avg_ctc = sum(c['ctc']) / len(c['ctc'])
        avg_nrtr = sum(c['nrtr']) / len(c['nrtr'])
        min_loss = min(c['loss'])
        max_loss = max(c['loss'])
        print(f"Epoch {ep:02d}: loss={avg_loss:.4f} [min: {min_loss:.3f}, max: {max_loss:.3f}], CTCLoss={avg_ctc:.4f}, NRTRLoss={avg_nrtr:.4f} ({len(c['loss'])} log points)")

print("=== STAGE 1 LOSSES ===")
analyze_losses(r"ocr-training/output/v25_stage1/stage1_final_checkpoint/output/rec_cham_v25/train.log")
print("\n=== STAGE 2 LOSSES ===")
analyze_losses(r"ocr-training/output/v25_stage2/stage2_final_checkpoint/train.log")
