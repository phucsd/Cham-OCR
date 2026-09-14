import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

log_path = r'C:\Users\admin\.gemini\antigravity\brain\4840561f-5c42-4e77-b2e3-145dae823fb9\.system_generated\tasks\task-4158.log'
epochs = {}
evals = []

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = re.search(r'epoch: \[(\d+)/40\], global_step: (\d+), lr: ([0-9.e-]+), CTCLoss: ([0-9.]+), NRTRLoss: ([0-9.]+), loss: ([0-9.]+)', line)
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
        
        m_eval = re.search(r'\[([^\]]+)\] ppocr INFO: cur metric, acc: ([0-9.]+), norm_edit_dis: ([0-9.]+)', line)
        if m_eval:
            evals.append({
                'time': m_eval.group(1),
                'acc': float(m_eval.group(2)),
                'ned': float(m_eval.group(3))
            })

print("=== TIẾN TRÌNH LOSS TỪNG EPOCH (CHẶNG 3) ===")
for ep in sorted(epochs.keys()):
    c = epochs[ep]
    avg_loss = sum(c['loss']) / len(c['loss'])
    avg_ctc = sum(c['ctc']) / len(c['ctc'])
    avg_nrtr = sum(c['nrtr']) / len(c['nrtr'])
    min_loss = min(c['loss'])
    min_ctc = min(c['ctc'])
    print(f"Epoch {ep:02d}: Avg Loss = {avg_loss:.4f} [Min: {min_loss:.3f}], Avg CTCLoss = {avg_ctc:.4f} [Min: {min_ctc:.3f}], NRTRLoss = {avg_nrtr:.4f} ({len(c['loss'])} log points)")

print("\n=== CÁC MỐC ĐÁNH GIÁ (EVAL) NỔI BẬT CHẶNG 3 ===")
# Sort by acc descending
sorted_evals = sorted(evals, key=lambda x: x['acc'], reverse=True)
for i, ev in enumerate(sorted_evals[:8]):
    print(f"Top {i+1}: Thời điểm {ev['time']} | Acc = {ev['acc']*100:.2f}% | Norm Edit Dis = {ev['ned']*100:.2f}%")

print(f"\nTổng số lần đánh giá định kỳ trong Chặng 3: {len(evals)}")
