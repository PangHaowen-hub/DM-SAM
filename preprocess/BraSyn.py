import SimpleITK as sitk
import os
import numpy as np
from tqdm import tqdm


def percentile_clip(input_tensor, p_min=0.5, p_max=99.5, strictlyPositive=True):
    v_min, v_max = np.percentile(input_tensor, [p_min, p_max])
    if v_min < 0 and strictlyPositive:
        v_min = 0
    if v_max <= v_min:
        return np.zeros_like(input_tensor, dtype=np.float32)
    output_tensor = np.clip(input_tensor, v_min, v_max)
    output_tensor = (output_tensor - v_min) / (v_max - v_min)
    return output_tensor.astype(np.float32)


def save_patient_slices_npz(img_id, img_root, save_root):
    image_path_t1n = os.path.join(img_root, img_id, img_id + '-t1n.nii.gz')
    image_path_t2w = os.path.join(img_root, img_id, img_id + '-t2w.nii.gz')
    image_path_t2f = os.path.join(img_root, img_id, img_id + '-t2f.nii.gz')
    image_path_t1c = os.path.join(img_root, img_id, img_id + '-t1c.nii.gz')
    image_path_seg = os.path.join(img_root, img_id, img_id + '-seg.nii.gz')

    for p in (image_path_t1n, image_path_t2w, image_path_t2f, image_path_t1c, image_path_seg):
        if not os.path.exists(p):
            print(f"[WARN] Missing file for {img_id}: {p}")
            return

    t1n = sitk.GetArrayFromImage(sitk.ReadImage(image_path_t1n))
    t2w = sitk.GetArrayFromImage(sitk.ReadImage(image_path_t2w))
    t2f = sitk.GetArrayFromImage(sitk.ReadImage(image_path_t2f))
    t1c = sitk.GetArrayFromImage(sitk.ReadImage(image_path_t1c))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(image_path_seg))

    if not (t1n.shape == t2w.shape == t2f.shape == t1c.shape == seg.shape):
        print(f"[WARN] Volume shape mismatch for {img_id}:",
              t1n.shape, t2w.shape, t2f.shape, t1c.shape, seg.shape)

    t1n_norm = percentile_clip(t1n, p_min=0.5, p_max=99.5, strictlyPositive=True)
    t2w_norm = percentile_clip(t2w, p_min=0.5, p_max=99.5, strictlyPositive=True)
    t2f_norm = percentile_clip(t2f, p_min=0.5, p_max=99.5, strictlyPositive=True)
    t1c_norm = percentile_clip(t1c, p_min=0.5, p_max=99.5, strictlyPositive=True)

    seg_bin = (seg != 0).astype(np.uint8)

    patient_dir = os.path.join(save_root, img_id)
    os.makedirs(patient_dir, exist_ok=True)

    nz = seg_bin.shape[0]
    saved_count = 0
    for i in range(nz):
        if np.sum(seg_bin[i, :, :]) == 0:
            continue
        slice_t1n = t1n_norm[i, :, :].astype(np.float32)
        slice_t2w = t2w_norm[i, :, :].astype(np.float32)
        slice_t2f = t2f_norm[i, :, :].astype(np.float32)
        slice_t1c = t1c_norm[i, :, :].astype(np.float32)

        npz_name = f"{img_id}_{str(i).rjust(5, '0')}.npz"
        npz_path = os.path.join(patient_dir, npz_name)

        np.savez_compressed(
            npz_path,
            T1N=slice_t1n,
            T2W=slice_t2w,
            T2F=slice_t2f,
            T1C=slice_t1c
        )
        saved_count += 1

    if saved_count == 0:
        print(f"[INFO] No non-empty slices for {img_id}, no npz saved.")
    else:
        print(f"[INFO] Saved {saved_count} npz files for {img_id} -> {patient_dir}")


if __name__ == "__main__":
    img_root = r'data'
    save_root = r'data_npz'

    os.makedirs(save_root, exist_ok=True)
    img_list = sorted(os.listdir(img_root))

    for img_id in tqdm(img_list):
        save_patient_slices_npz(img_id, img_root, save_root)
