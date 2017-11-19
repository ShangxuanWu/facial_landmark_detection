rm -r ~/test_results/*

# images_dir, save_dir, resize_factor, gpu_id, start_frame, end_frame, rot_correct, gopro, vis_resize_factor, camera_id

python chopping_pipeline.py ~/test_images ~/test_results 4 1 0 1000000 1 0 1.0 330030 all
