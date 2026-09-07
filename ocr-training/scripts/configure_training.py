import os
import sys
import urllib.request
import tarfile
import yaml

if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

MODEL_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"
PRETRAIN_DIR_NAME = "ch_PP-OCRv4_rec_train"

def download_progress_hook(block_num, block_size, total_size):
    percent = int(block_num * block_size * 100 / total_size)
    percent = min(100, percent)
    sys.stdout.write(f"\r📥 Đang tải base model: {percent}% [{block_num * block_size}/{total_size} bytes]")
    sys.stdout.flush()

def download_and_extract_base_model(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join(target_dir, "ch_PP-OCRv4_rec_train.tar")
    extracted_model_path = os.path.join(target_dir, PRETRAIN_DIR_NAME)
    
    if os.path.exists(tar_path) and os.path.getsize(tar_path) == 0:
        print("⚠️  Phát hiện tệp tar tải dở dang (0 bytes). Đang xóa để tải lại...")
        os.remove(tar_path)

    if not os.path.exists(tar_path) and not os.path.exists(extracted_model_path):
        print(f"📥 Bắt đầu tải base model PP-OCRv4 từ: {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, tar_path, reporthook=download_progress_hook)
            print("\n✅ Tải hoàn tất.")
        except Exception as e:
            print(f"\n❌ Lỗi khi tải mô hình: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            return None
    else:
        print("ℹ️  Tệp mô hình base đã tồn tại hoặc đã được giải nén. Bỏ qua tải xuống.")

    if os.path.exists(tar_path) and not os.path.exists(extracted_model_path):
        print(f"📦 Bắt đầu giải nén tệp {tar_path}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=target_dir)
            print("✅ Giải nén hoàn tất.")
            os.remove(tar_path)
            print("🗑️  Đã dọn dẹp tệp nén .tar.")
        except Exception as e:
            print(f"❌ Lỗi khi giải nén mô hình: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            return None
            
    student_model_prefix = os.path.join(extracted_model_path, "best_accuracy")
    print(f"🎯 Đường dẫn base model học sinh (student model): {student_model_prefix}")
    return student_model_prefix

def configure_training(template_path, output_config_path, data_dir, dict_path, pretrained_model_prefix, 
                       is_kaggle=None, max_text_length=80, image_shape=[3, 48, 1024], batch_size=64, 
                       model_head_type='MultiHead', epoch_num=200, save_model_dir=None, train_label_name='train_label.txt'):
    """
    Cấu hình YAML cho V25 với các tham số độ dài tối đa, kích thước ảnh, batch size, loại model head.
    """
    if is_kaggle is None:
        is_kaggle = os.path.exists('/kaggle/working')

    if not os.path.exists(template_path):
        print(f"❌ Không tìm thấy tệp cấu hình mẫu tại: {template_path}")
        return False
        
    print(f"🛠️  Đang đọc tệp cấu hình mẫu từ: {template_path}...")
    with open(template_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. Global config
    config['Global'] = config.get('Global', {})
    
    # Auto GPU check
    has_gpu = False
    try:
        import paddle
        has_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except ImportError:
        has_gpu = True
            
    config['Global']['use_gpu'] = has_gpu
    config['Global']['pretrained_model'] = pretrained_model_prefix.replace('\\', '/')

    if save_model_dir is None:
        output_dir = os.path.join(data_dir, 'output/rec_cham').replace('\\', '/')
    else:
        output_dir = save_model_dir.replace('\\', '/')
    
    config['Global']['save_res_path'] = output_dir
    config['Global']['save_model_dir'] = output_dir
    config['Global']['character_dict_path'] = dict_path.replace('\\', '/')
    config['Global']['epoch_num'] = epoch_num
    config['Global']['save_epoch_step'] = min(10, epoch_num)
    config['Global']['eval_batch_step'] = [0, 1000] if not is_kaggle else [0, 2000]
    config['Global']['use_space_char'] = True
    config['Global']['max_text_length'] = max_text_length
    config['Global']['d2s_train_image_shape'] = image_shape

    # 2. Config datasets and loaders
    for split in ['Train', 'Eval']:
        if split in config:
            dataset = config[split].get('dataset', {})
            dataset['data_dir'] = os.path.join(data_dir, 'cham_synthetic_images').replace('\\', '/')
            
            # Default label list
            label_file = train_label_name if split == 'Train' else 'val_clean_short_label.txt'
            dataset['label_file_list'] = [os.path.join(data_dir, f'cham_synthetic_images/{label_file}').replace('\\', '/')]
            
            config[split]['dataset'] = dataset
            
            # Loader
            loader = config[split].get('loader', {})
            loader['batch_size_per_card'] = batch_size
            loader['num_workers'] = 2
            config[split]['loader'] = loader

            # Train sampler
            if split == 'Train' and 'sampler' in config[split]:
                sampler = config[split]['sampler']
                if 'first_bs' in sampler:
                    sampler['first_bs'] = batch_size
                # Match scales to the target image size
                img_c, img_h, img_w = image_shape
                sampler['scales'] = [[img_w, img_h - 16], [img_w, img_h], [img_w, img_h + 16]]

    # 3. Model Head & Loss modification
    img_c, img_h, img_w = image_shape
    
    if model_head_type == 'CTCHead':
        # CTC-only setup
        config['Architecture']['Head'] = {
            'name': 'CTCHead',
            'Neck': {
                'name': 'svtr',
                'dims': 120,
                'depth': 2,
                'hidden_dims': 120,
                'kernel_size': [1, 3],
                'use_guide': True
            },
            'Head': {
                'fc_decay': 0.00001
            }
        }
        config['Loss'] = {
            'name': 'CTCLoss'
        }
        config['PostProcess'] = {
            'name': 'CTCLabelDecode'
        }
        
        # Modify transforms for Train
        train_transforms = []
        for t in config['Train']['dataset'].get('transforms', []):
            t_name = list(t.keys())[0]
            if t_name == 'RecConAug':
                t['RecConAug']['image_shape'] = [img_h, img_w, img_c]
                t['RecConAug']['max_text_length'] = max_text_length
                train_transforms.append(t)
            elif t_name == 'MultiLabelEncode':
                # Replace with CTCLabelEncode
                train_transforms.append({'CTCLabelEncode': None})
            elif t_name == 'KeepKeys':
                train_transforms.append({'KeepKeys': {'keep_keys': ['image', 'label', 'length']}})
            else:
                train_transforms.append(t)
        config['Train']['dataset']['transforms'] = train_transforms
        
        # Modify transforms for Eval
        eval_transforms = []
        for t in config['Eval']['dataset'].get('transforms', []):
            t_name = list(t.keys())[0]
            if t_name == 'MultiLabelEncode':
                eval_transforms.append({'CTCLabelEncode': None})
            elif t_name == 'RecResizeImg':
                t['RecResizeImg']['image_shape'] = [img_c, img_h, img_w]
                eval_transforms.append(t)
            elif t_name == 'KeepKeys':
                eval_transforms.append({'KeepKeys': {'keep_keys': ['image', 'label', 'length']}})
            else:
                eval_transforms.append(t)
        config['Eval']['dataset']['transforms'] = eval_transforms

    else:
        # MultiHead (CTC + NRTR) setup
        config['Architecture']['Head'] = {
            'name': 'MultiHead',
            'head_list': [
                {
                    'CTCHead': {
                        'Neck': {
                            'name': 'svtr',
                            'dims': 120,
                            'depth': 2,
                            'hidden_dims': 120,
                            'kernel_size': [1, 3],
                            'use_guide': True
                        },
                        'Head': {
                            'fc_decay': 0.00001
                        }
                    }
                },
                {
                    'NRTRHead': {
                        'nrtr_dim': 384,
                        'max_text_length': max_text_length
                    }
                }
            ]
        }
        config['Loss'] = {
            'name': 'MultiLoss',
            'loss_config_list': [
                {'CTCLoss': None},
                {'NRTRLoss': None}
            ]
        }
        config['PostProcess'] = {
            'name': 'CTCLabelDecode'
        }
        
        # Modify transforms for Train
        train_transforms = []
        for t in config['Train']['dataset'].get('transforms', []):
            t_name = list(t.keys())[0]
            if t_name == 'RecConAug':
                t['RecConAug']['image_shape'] = [img_h, img_w, img_c]
                t['RecConAug']['max_text_length'] = max_text_length
                train_transforms.append(t)
            elif t_name == 'MultiLabelEncode':
                t['MultiLabelEncode']['gtc_encode'] = 'NRTRLabelEncode'
                train_transforms.append(t)
            elif t_name == 'KeepKeys':
                train_transforms.append({'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']}})
            else:
                train_transforms.append(t)
        config['Train']['dataset']['transforms'] = train_transforms
        
        # Modify transforms for Eval
        eval_transforms = []
        for t in config['Eval']['dataset'].get('transforms', []):
            t_name = list(t.keys())[0]
            if t_name == 'MultiLabelEncode':
                t['MultiLabelEncode']['gtc_encode'] = 'NRTRLabelEncode'
                eval_transforms.append(t)
            elif t_name == 'RecResizeImg':
                t['RecResizeImg']['image_shape'] = [img_c, img_h, img_w]
                eval_transforms.append(t)
            elif t_name == 'KeepKeys':
                eval_transforms.append({'KeepKeys': {'keep_keys': ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']}})
            else:
                eval_transforms.append(t)
        config['Eval']['dataset']['transforms'] = eval_transforms

    # Save out
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
    with open(output_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
    print(f"✅ Đã tạo cấu hình V25 ({model_head_type}, {image_shape}, BS={batch_size}) tại: {output_config_path}")
    return True

if __name__ == '__main__':
    is_kaggle = os.path.exists('/kaggle/working')
    
    if is_kaggle:
        PROJECT_ROOT = "/kaggle/working/paddleocr_cham_finetune"
        paddleocr_root = "/kaggle/working/PaddleOCR"
    else:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paddleocr_root = os.path.join(PROJECT_ROOT, "PaddleOCR")
    
    config_template = os.path.join(paddleocr_root, "configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml")
    output_config = os.path.join(PROJECT_ROOT, "configs", "rec_cham_config.yml")
    data_root = os.path.join(PROJECT_ROOT, "data")
    dict_file = os.path.join(data_root, "cham_dict.txt")
    
    pretrain_target_dir = os.path.join(data_root, "pretrain_models")
    model_prefix = download_and_extract_base_model(pretrain_target_dir)
    
    if model_prefix and os.path.exists(config_template):
        configure_training(config_template, output_config, data_root, dict_file, model_prefix, is_kaggle)
