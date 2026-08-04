# Workspace Cleanup - Consolidation Plan

## Python Scripts to Archive (11 files → 3 files)

### Keep (Core Workflows):
- train_do_it_2.py (primary training)
- do_it.py (dataset conversion)
- live_inference_windows.py (visualization)

### Archive to archive/scripts/:
- additional.py
- buildfpga.py (superseded by WSL workflow)
- convert_onnx_patched.py (superseded)
- download_onnx.py
- export_fpga.py
- finalize.py
- fix.py
- fix_litert.py
- model_train.py
- new_patch.py
- patch_model.py
- reconstruction.py
- train_do_it.py (old version)
- trial.py
- visualize.py
- yolov8fpga.py
- export_clean_vnnx.py (used once, done)
- export_tflite_nometadata.py (used once, done)
- reconstruct_test_images.py (used once, done)
- verify_model.py (used once, done)

## Markdown Files to Consolidate (8 files → 2 files)

### Keep:
- README.md (project overview + quick start)
- DEPLOYMENT_GUIDE.md (consolidate all deployment docs)

### Archive to archive/docs/:
- claude_context.md
- readme.md (duplicate of README.md?)
- TECHNICAL_OVERVIEW.md
- VNNX_COMPILATION_SUCCESS.md
- MODEL_VERIFICATION_REPORT.md
- WINDOWS_INFERENCE_GUIDE.md
- README_DEPLOYMENT.md
- technical_overview.md (duplicate)

## Log Files to Archive (3 files):
- onnx2tf_clean.log (413 KB)
- vnnx_compile_clean.log (75 KB)
- training_log.txt (if exists)

## Directories to Clean:
- checkpoints/ (if empty or superseded)
- test_images_reconstructed/ (already verified, archive)
- test_labels_reconstructed/ (already verified, archive)

## Estimated Token Savings:
- ~30-40% reduction in context size
- Faster file navigation
- Clearer project structure
