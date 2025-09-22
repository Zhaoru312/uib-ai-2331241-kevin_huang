# Configure TensorFlow to use GPU with mixed precision if available
physical_devices = tf.config.list_physical_devices('GPU')
gpu_available = bool(physical_devices)

if gpu_available:
    try:
        # Enable memory growth for all GPUs
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set mixed precision policy for GPUs
        # Using 'mixed_float16' for computation, 'float32' for variables
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        logger.info(f"Using GPU with mixed precision: {physical_devices}")
        logger.info(f"Compute dtype: {policy.compute_dtype}")
        logger.info(f"Variable dtype: {policy.variable_dtype}")
        
    except RuntimeError as e:
        logger.error(f"Error configuring GPU: {e}")
        gpu_available = False

if not gpu_available:
    # Fallback to CPU with float32 policy
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    logger.info("Using CPU with float32 precision")
