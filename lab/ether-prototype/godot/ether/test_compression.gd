extends Node

## Quick test for the CompressionEstimator GDExtension.
## Attach to any node, or set this scene as main to run it.

func _print_estimates(est: CompressionEstimator, label: String) -> void:
	var all := est.estimate_all()
	var input_size: int = all.get("input_size", 0)
	print("%s (%d bytes):" % [label, input_size])
	print("  %-20s %12s %10s %8s" % ["algorithm", "bytes", "bpb", "ratio"])
	print("  %-20s %12s %10s %8s" % ["─────────", "─────", "───", "─────"])
	var results: Dictionary = all.get("results", {})
	for algo_name: String in results:
		var r: Dictionary = results[algo_name]
		print("  %-20s %12d %10.4f %7.2f%%" % [
			algo_name,
			r.get("bytes", 0),
			r.get("bits_per_byte", 0.0),
			r.get("ratio", 0.0) * 100.0,
		])
	print()


func _roundtrip_tests(est: CompressionEstimator, original: PackedByteArray) -> void:
	var algos := ["huffman", "arithmetic", "ans", "gzip", "zstd", "lz4", "deflate"]
	print("  Round-trip tests:")
	for algo_name: String in algos:
		var compressed := est.compress(algo_name)
		if compressed.is_empty():
			print("    %-12s  compress FAILED" % algo_name)
			continue
		var decompressed := CompressionEstimator.decompress(algo_name, compressed, original.size())
		if decompressed.is_empty():
			print("    %-12s  decompress FAILED (compressed %d bytes)" % [algo_name, compressed.size()])
			continue
		var ok := decompressed == original
		print("    %-12s  %d -> %d bytes (%.1f%%)  %s" % [
			algo_name,
			original.size(),
			compressed.size(),
			float(compressed.size()) / float(original.size()) * 100.0,
			"OK" if ok else "MISMATCH",
		])
	print()


func _test_gpu_session(est: CompressionEstimator) -> void:
	print("=== GPU Session (transform + scoring pipeline) ===\n")

	var err := est.create_session()
	if err != OK:
		print("  create_session FAILED: ", err)
		return

	var slots := est.get_session_slots()
	print("  Session created: %d parallel slots" % slots)

	# Test all 5 transform types
	var transforms: Array = [
		{"type": "delta"},
		{"type": "xor"},
		{"type": "rotate", "param0": 0},  # identity
		{"type": "rotate", "param0": 3},
		{"type": "sub_mean", "param0": 8},
		{"type": "byte_swap"},
	]

	print("  Evaluating %d transforms..." % transforms.size())
	var results := est.evaluate_batch(transforms)

	if results.is_empty():
		print("  evaluate_batch FAILED")
		est.destroy_session()
		return

	print("  %-15s %10s %16s" % ["transform", "bpb", "total bits"])
	print("  %-15s %10s %16s" % ["─────────", "───", "──────────"])
	for r: Dictionary in results:
		var name: String = r.get("transform", "?")
		var p: Dictionary = r.get("params", {})
		var label := name
		if p.get("param0", 0) != 0:
			label = "%s(%d)" % [name, p["param0"]]
		print("  %-15s %10.4f %16.0f" % [
			label,
			r.get("entropy_o0_bpb", 0.0),
			r.get("entropy_o0_total", 0.0),
		])
	print()

	# Batch performance test: 200 transforms
	var perf_transforms: Array = []
	for i in 200:
		var types := ["delta", "xor", "rotate", "sub_mean", "byte_swap"]
		perf_transforms.append({
			"type": types[i % 5],
			"param0": i % 8,
		})

	var t0 := Time.get_ticks_msec()
	var perf_results := est.evaluate_batch(perf_transforms)
	var t1 := Time.get_ticks_msec()
	var elapsed := t1 - t0
	if not perf_results.is_empty():
		print("  Performance: %d transforms in %d ms" % [perf_transforms.size(), elapsed])
		if elapsed > 0:
			print("  Throughput:  %.0f transforms/sec" % (perf_transforms.size() * 1000.0 / elapsed))

		# Find best and worst
		var best_bpb := 99.0
		var worst_bpb := 0.0
		var best_name := ""
		var worst_name := ""
		for r: Dictionary in perf_results:
			var bpb: float = r.get("entropy_o0_bpb", 99.0)
			var name: String = r.get("transform", "?")
			var p: Dictionary = r.get("params", {})
			var label := "%s(%d)" % [name, p.get("param0", 0)]
			if bpb < best_bpb:
				best_bpb = bpb
				best_name = label
			if bpb > worst_bpb:
				worst_bpb = bpb
				worst_name = label
		print("  Best:  %s at %.4f bpb" % [best_name, best_bpb])
		print("  Worst: %s at %.4f bpb" % [worst_name, worst_bpb])
	print()

	est.destroy_session()


func _test_neural_compression(est: CompressionEstimator) -> void:
	print("=== Neural Compression (model IS the algorithm) ===\n")

	# Use 64KB of loaded data (or synthetic if no enwik8)
	var test_data := PackedByteArray()
	test_data.resize(8192)  # 8 KB for quick Godot test
	for i in test_data.size():
		test_data[i] = (i * 7 + i / 100) % 256
	est.load_data(test_data)

	# Create neural compressor
	var err := est.neural_create(8, 8, 64)
	if err != OK:
		print("  neural_create FAILED: ", err)
		return

	# Train on CPU first for baseline
	print("  Training CPU for 5 seconds on %d bytes..." % test_data.size())
	var t0 := Time.get_ticks_msec()
	var bpb_cpu := est.neural_train(5)
	var t1 := Time.get_ticks_msec()
	print("  CPU bpb: %.4f (%d ms)" % [bpb_cpu, t1 - t0])

	# Recreate and train on GPU
	est.neural_destroy()
	est.neural_create(8, 8, 64)
	print("  Training GPU for 5 seconds on %d bytes..." % test_data.size())
	t0 = Time.get_ticks_msec()
	var bpb := est.neural_train_gpu(5)
	t1 = Time.get_ticks_msec()
	print("  GPU bpb: %.4f (%d ms)" % [bpb, t1 - t0])
	print("  GPU speedup: GPU reached %.4f bpb vs CPU %.4f bpb in same time" % [bpb, bpb_cpu])

	# Compress
	var result := est.neural_compress()
	if result.is_empty():
		print("  neural_compress FAILED")
		est.neural_destroy()
		return

	var comp_data: PackedByteArray = result["compressed"]
	var orig_size: int = result["original_size"]
	var comp_size: int = result["compressed_size"]
	print("  Compressed: %d -> %d bytes (%.1f%%)" % [orig_size, comp_size, result["ratio"] * 100.0])
	print("  Parameters: %d" % result["param_count"])

	# Decompress and verify roundtrip
	var decompressed := CompressionEstimator.neural_decompress(comp_data)
	if decompressed.is_empty():
		print("  neural_decompress FAILED")
	elif decompressed == test_data:
		print("  Roundtrip: OK")
	else:
		print("  Roundtrip: MISMATCH (got %d bytes)" % decompressed.size())

	est.neural_destroy()
	print()


func _ready() -> void:
	print("=== CompressionEstimator GDExtension Test ===\n")

	var est := CompressionEstimator.new()

	# --- Backend info ---
	var info := est.get_backend_info()
	print("Backend:  ", info.get("backend", "?"))
	print("Device:   ", info.get("device_name", "?"))
	print("Vendor:   ", info.get("vendor", "?"))
	print("Max buf:  ", info.get("max_buffer_size", 0), " bytes")
	print("Slots:    ", info.get("max_slots", 0))
	print()

	# --- Synthetic test data (64 KB, near-uniform) ---
	var synth := PackedByteArray()
	synth.resize(65536)
	for i in synth.size():
		synth[i] = (i * 7 + i / 100) % 256
	est.load_data(synth)
	_print_estimates(est, "Synthetic data")
	_roundtrip_tests(est, synth)

	# --- Easy data (all same byte) ---
	var easy := PackedByteArray()
	easy.resize(4096)
	easy.fill(65)  # all 'A'
	est.load_data(easy)
	print("Easy data (4096 x 'A'):")
	var easy_est := est.estimate("huffman")
	print("  Huffman estimate: %d bytes (%.1f%%)" % [
		easy_est.get("bytes", 0),
		easy_est.get("ratio", 0.0) * 100.0,
	])
	var easy_comp := est.compress("huffman")
	if not easy_comp.is_empty():
		var easy_decomp := CompressionEstimator.decompress("huffman", easy_comp, easy.size())
		print("  Huffman roundtrip: %d -> %d bytes  %s" % [
			easy.size(), easy_comp.size(),
			"OK" if easy_decomp == easy else "MISMATCH",
		])
	print()

	# --- enwik8 (real-world data) ---
	var project_dir := ProjectSettings.globalize_path("res://")
	var abs_path := project_dir.get_base_dir().get_base_dir().get_base_dir().path_join("enwik8")
	if FileAccess.file_exists(abs_path):
		print("Loading enwik8 from: ", abs_path)
		# Load 10MB chunk for session tests (more parallel slots than full 100MB)
		var f := FileAccess.open(abs_path, FileAccess.READ)
		var chunk_10mb := f.get_buffer(10 * 1048576)  # 10 MB
		f.close()
		est.load_data(chunk_10mb)
		print("  Loaded 10MB chunk, parallel slots: ", est.get_parallel_slots())
		_print_estimates(est, "enwik8 first 10MB")

		# GPU session test on 10MB chunk
		_test_gpu_session(est)

		# Round-trip a 1MB chunk
		f = FileAccess.open(abs_path, FileAccess.READ)
		var chunk_1mb := f.get_buffer(1048576)
		f.close()
		est.load_data(chunk_1mb)
		print("enwik8 first 1MB chunk:")
		_roundtrip_tests(est, chunk_1mb)

		# Full enwik8 estimates
		var err := est.load_file(abs_path)
		if err == OK:
			_print_estimates(est, "enwik8 full (100MB)")
		else:
			print("  load_file FAILED: ", err)
	else:
		print("enwik8 not found at: ", abs_path, " (skipping)")

	# --- Neural compression ---
	_test_neural_compression(est)

	print("=== All tests complete ===")
