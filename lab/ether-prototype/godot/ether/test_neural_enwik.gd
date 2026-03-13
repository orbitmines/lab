extends Node

## Test GPU neural compression on FULL enwik8 (96MB).
## Does it actually compress?

func _ready() -> void:
	print("=== GPU Neural Compression on FULL enwik8 ===\n")

	var est := CompressionEstimator.new()
	var info := est.get_backend_info()
	print("Device: %s (%s)" % [info.get("device_name", "?"), info.get("backend", "?")])
	print()

	# Find enwik8
	var project_dir := ProjectSettings.globalize_path("res://")
	var abs_path := project_dir.get_base_dir().get_base_dir().get_base_dir().path_join("enwik8")
	if not FileAccess.file_exists(abs_path):
		print("ERROR: enwik8 not found at: %s" % abs_path)
		get_tree().quit(1)
		return

	# Load FULL enwik8
	print("Loading full enwik8 from: %s" % abs_path)
	var err := est.load_file(abs_path)
	if err != OK:
		print("ERROR: load_file failed: %d" % err)
		get_tree().quit(1)
		return

	# Also read raw bytes for roundtrip verification
	var f := FileAccess.open(abs_path, FileAccess.READ)
	var full_data := f.get_buffer(f.get_length())
	f.close()
	print("Loaded %d bytes (%.1f MB)\n" % [full_data.size(), full_data.size() / 1048576.0])

	# --- Train GPU for 120s on full enwik8 ---
	print("--- Training: full enwik8, 120s GPU training ---")
	est.neural_create(8, 8, 64)

	var t0 := Time.get_ticks_msec()
	var bpb := est.neural_train_gpu(120)
	var t1 := Time.get_ticks_msec()
	print("  Trained: %.4f bpb in %d ms (%.1f min)" % [bpb, t1 - t0, (t1 - t0) / 60000.0])

	# --- Compress ---
	print("  Compressing...")
	var result := est.neural_compress()
	if result.is_empty():
		print("  COMPRESS FAILED")
	else:
		var orig: int = result["original_size"]
		var comp: int = result["compressed_size"]
		var ratio: float = result["ratio"]
		var params: int = result["param_count"]
		print("  Original:   %d bytes (%.1f MB)" % [orig, orig / 1048576.0])
		print("  Compressed: %d bytes (%.1f MB) (%.1f%%)" % [comp, comp / 1048576.0, ratio * 100.0])
		print("  Parameters: %d (%d bytes as 8-bit quantized)" % [params, params])
		if comp < orig:
			var saved := orig - comp
			print("  ACTUALLY COMPRESSES! Saved %d bytes (%.1f MB)" % [saved, saved / 1048576.0])
		else:
			print("  Does NOT compress yet (ratio %.1f%%)" % (ratio * 100.0))

		# Verify roundtrip
		print("  Decompressing for roundtrip check...")
		var comp_data: PackedByteArray = result["compressed"]
		var decompressed := CompressionEstimator.neural_decompress(comp_data)
		if decompressed.is_empty():
			print("  ROUNDTRIP FAILED: decompress returned empty")
		elif decompressed == full_data:
			print("  Roundtrip: VERIFIED (lossless!)")
		else:
			print("  ROUNDTRIP MISMATCH (got %d bytes, expected %d)" % [decompressed.size(), full_data.size()])

	est.neural_destroy()
	print()

	# --- Compare with zstd on same full data ---
	print("--- Comparison: zstd on full enwik8 ---")
	var zstd_result := est.estimate("zstd")
	print("  zstd estimate: %d bytes (%.1f MB, %.4f bpb)" % [
		zstd_result.get("bytes", 0),
		zstd_result.get("bytes", 0) / 1048576.0,
		zstd_result.get("bits_per_byte", 0.0)])
	print()

	print("=== Done ===")
	get_tree().quit(0)
