import os
import json
import paddle
import numpy as np

def main():
    old_ckpt_path = 'data/output_v23/rec_cham_best_model/iter_epoch_200.pdparams'
    new_ckpt_dir = 'data/output_v23_1_temp/rec_cham_best_model'
    new_ckpt_path = os.path.join(new_ckpt_dir, 'iter_epoch_200.pdparams')
    report_path = 'output/v23_1_training/head_extension_report.json'
    
    os.makedirs(new_ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Load old state dict
    old_state = paddle.load(old_ckpt_path)
    new_state = {}
    
    # Weights we need to surgery:
    # 1. head.ctc_head.fc.weight: [120, 84] -> [120, 103]
    # 2. head.ctc_head.fc.bias: [84] -> [103]
    # 3. head.gtc_head.embedding.embedding.weight: [88, 384] -> [107, 384]
    # 4. head.gtc_head.tgt_word_prj.weight: [384, 88] -> [384, 107]
    
    for k, v in old_state.items():
        if k == 'head.ctc_head.fc.weight':
            # shape: [in_features, out_features]
            in_features, old_classes = v.shape # 120, 84
            new_classes = 103
            
            # Create new tensor
            std = v.std().item()
            mean = v.mean().item()
            # Random initialization for new classes
            new_v_np = np.random.normal(mean, std, (in_features, new_classes)).astype('float32')
            
            # Copy old weights
            # idx 0: blank (index 0)
            # idx 1 to 82: old chars (indices 1 to 82)
            new_v_np[:, 0:83] = v.numpy()[:, 0:83]
            # idx 83 (old space) -> idx 102 (new space)
            new_v_np[:, 102] = v.numpy()[:, 83]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"Surgery on {k}: {v.shape} -> {new_state[k].shape}")
            
        elif k == 'head.ctc_head.fc.bias':
            # shape: [out_features]
            old_classes = v.shape[0] # 84
            new_classes = 103
            
            new_v_np = np.zeros(new_classes, dtype='float32')
            # Copy old biases
            new_v_np[0:83] = v.numpy()[0:83]
            new_v_np[102] = v.numpy()[83]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"Surgery on {k}: {v.shape} -> {new_state[k].shape}")
            
        elif k == 'head.gtc_head.embedding.embedding.weight':
            # shape: [vocab, embed_dim]
            old_vocab, embed_dim = v.shape # 88, 384
            new_vocab = 107
            
            std = v.std().item()
            mean = v.mean().item()
            new_v_np = np.random.normal(mean, std, (new_vocab, embed_dim)).astype('float32')
            
            # Copy old weights
            # idx 0 to 85: blank, <unk>, <s>, </s> + 82 old characters
            new_v_np[0:86, :] = v.numpy()[0:86, :]
            # idx 86 (old space) -> idx 105 (new space)
            new_v_np[105, :] = v.numpy()[86, :]
            # idx 87 (old padding/other) -> idx 106 (new padding/other)
            new_v_np[106, :] = v.numpy()[87, :]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"Surgery on {k}: {v.shape} -> {new_state[k].shape}")
            
        elif k == 'head.gtc_head.tgt_word_prj.weight':
            # shape: [embed_dim, vocab]
            embed_dim, old_vocab = v.shape # 384, 88
            new_vocab = 107
            
            std = v.std().item()
            mean = v.mean().item()
            new_v_np = np.random.normal(mean, std, (embed_dim, new_vocab)).astype('float32')
            
            # Copy old weights
            new_v_np[:, 0:86] = v.numpy()[:, 0:86]
            new_v_np[:, 105] = v.numpy()[:, 86]
            new_v_np[:, 106] = v.numpy()[:, 87]
            
            new_state[k] = paddle.to_tensor(new_v_np)
            print(f"Surgery on {k}: {v.shape} -> {new_state[k].shape}")
            
        else:
            new_state[k] = v
            
    # Save new state dict
    paddle.save(new_state, new_ckpt_path)
    
    # Write report
    report = {
        "old_num_classes": 84,
        "new_num_classes": 103,
        "old_chars_preserved": True,
        "new_chars_initialized": True,
        "blank_index_old": 0,
        "blank_index_new": 0,
        "strategy": "head_weight_extension"
    }
    
    with open(report_path, 'w', encoding='utf-8') as f_rep:
        json.dump(report, f_rep, indent=2)
        
    print("✅ Head extension report saved to:", report_path)
    print("✅ Extended trainable checkpoint saved to:", new_ckpt_path)

if __name__ == '__main__':
    main()
