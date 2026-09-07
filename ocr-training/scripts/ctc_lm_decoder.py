import numpy as np
import re

def greedy_decode(logits, vocab, blank_idx):
    """
    Giải mã Greedy CTC từ ma trận xác suất logits [T, C].
    """
    preds = np.argmax(logits, axis=-1)
    prev = -1
    decoded = []
    for p in preds:
        if p != prev:
            if p != blank_idx and p < len(vocab):
                decoded.append(vocab[p])
            prev = p
    return "".join(decoded)

def log_add(x, y):
    """
    Log-add exp helper: returns log(exp(x) + exp(y))
    """
    if x == -float('inf'):
        return y
    if y == -float('inf'):
        return x
    return max(x, y) + np.log1p(np.exp(-abs(x - y)))

def prefix_beam_search_decode(logits, vocab, blank_idx, lm_model=None, beam_size=10, alpha=0.5, beta=0.1):
    """
    Giải mã CTC Prefix Beam Search trong log-domain tránh tràn số dưới (underflow) trên chuỗi dài.
    logits: ma trận xác suất [T, C] hoặc log-probs.
    """
    T, C = logits.shape
    # Đảm bảo logits là log-probabilities
    if np.max(logits) <= 1.0 + 1e-5:
        log_probs = np.log(np.clip(logits, 1e-20, 1.0))
    else:
        log_probs = logits

    # Khởi tạo path: {text: (log_p_blank, log_p_non_blank)}
    paths = {"": (0.0, -float('inf'))}

    for t in range(T):
        new_paths = {}
        step_log_probs = log_probs[t]

        for text, (log_pb, log_pnb) in paths.items():
            for c_idx in range(C):
                log_p_char = float(step_log_probs[c_idx])
                
                # Xử lý blank token
                if c_idx == blank_idx:
                    log_pb_new, log_pnb_new = new_paths.get(text, (-float('inf'), -float('inf')))
                    log_pb_new = log_add(log_pb_new, log_p_char + log_add(log_pb, log_pnb))
                    new_paths[text] = (log_pb_new, log_pnb_new)
                else:
                    char = vocab[c_idx] if c_idx < len(vocab) else ""
                    # Xử lý lặp ký tự ở biên
                    if len(text) > 0 and text[-1] == char:
                        # 1. Kéo dài chuỗi (nếu trước đó có blank)
                        log_pb_new_ext, log_pnb_new_ext = new_paths.get(text + char, (-float('inf'), -float('inf')))
                        log_pnb_new_ext = log_add(log_pnb_new_ext, log_p_char + log_pb)
                        new_paths[text + char] = (log_pb_new_ext, log_pnb_new_ext)
                        # 2. Giữ nguyên chuỗi (nếu trước đó không có blank)
                        log_pb_new, log_pnb_new = new_paths.get(text, (-float('inf'), -float('inf')))
                        log_pnb_new = log_add(log_pnb_new, log_p_char + log_pnb)
                        new_paths[text] = (log_pb_new, log_pnb_new)
                    else:
                        # Ký tự mới: luôn kéo dài chuỗi
                        log_pb_new_ext, log_pnb_new_ext = new_paths.get(text + char, (-float('inf'), -float('inf')))
                        log_pnb_new_ext = log_add(log_pnb_new_ext, log_p_char + log_add(log_pb, log_pnb))
                        new_paths[text + char] = (log_pb_new_ext, log_pnb_new_ext)

        # Thu hẹp chùm (Keep top beam_size * 2)
        sorted_paths = sorted(new_paths.items(), key=lambda x: log_add(x[1][0], x[1][1]), reverse=True)
        paths = dict(sorted_paths[:beam_size * 2])

    # Tính điểm rescoring kết hợp LM
    final_results = []
    for text, (log_pb, log_pnb) in paths.items():
        log_ctc_prob = log_add(log_pb, log_pnb)
        if log_ctc_prob == -float('inf'):
            continue

        lm_score = 0.0
        if lm_model is not None and len(text) > 0:
            tokens = [('_' if c == ' ' else c) for c in text]
            lm_query = " ".join(tokens)
            try:
                lm_score = lm_model.score(lm_query)
            except Exception:
                lm_score = -10.0

        # Score = log_ctc_prob + alpha * lm_score + beta * len(text)
        score = log_ctc_prob + alpha * lm_score + beta * len(text)
        final_results.append((text, score))

    final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
    return final_results[:beam_size]

def calculate_oracle_accuracy(predictions_top_n, ground_truth):
    """
    Kiểm tra xem ground_truth có nằm trong danh sách Top-N hay không.
    predictions_top_n: list của text giải mã.
    """
    gt_clean = re.sub(r'\s+', ' ', ground_truth.strip())
    for pred in predictions_top_n:
        pred_clean = re.sub(r'\s+', ' ', pred.strip())
        if pred_clean == gt_clean:
            return 1.0
    return 0.0
