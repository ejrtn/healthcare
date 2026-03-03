import pickle
import glob
import os

base_dir = r"d:\healthcare_img\healthcare\assets"
pkl_files = glob.glob(os.path.join(base_dir, "monai_ct_convnext_v*.pkl"))

results = []
results.append("CT Model Validation Results Summary")
results.append("="*50)

for p in sorted(pkl_files):
    filename = os.path.basename(p)
    version = filename.replace("monai_ct_convnext_", "").replace(".pkl", "")
    try:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        
        auc = data.get('auc_avg_loss', [])
        val_loss = data.get('val_loss', [])
        
        if auc:
            best_auc = max(auc)
            best_epoch = auc.index(best_auc) + 1
            corr_loss = val_loss[best_epoch-1] if val_loss and len(val_loss) >= best_epoch else "N/A"
            if isinstance(best_auc, float):
                best_auc = f"{best_auc:.4f}"
            if isinstance(corr_loss, float):
                corr_loss = f"{corr_loss:.4f}"
            
            results.append(f"Version: {version:5s} | Best Epoch: {best_epoch:2d} | Best AUC: {best_auc} | Val Loss: {corr_loss}")
        else:
            results.append(f"Version: {version:5s} | Best Epoch: N/A | Best AUC: N/A | Val Loss: N/A")
            
    except Exception as e:
        results.append(f"Version: {version:5s} | Error loading file: {e}")

output_path = r"d:\healthcare_img\healthcare\ct_results_summary.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("Extraction complete. Results saved to:", output_path)
