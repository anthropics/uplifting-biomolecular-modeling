"""CPU test: the image key of a shipped driver cache is the LIST of images it was captured on (captures[].cuda_libs_sha16 + the capture image's own):
a host whose CUDA-libs fingerprint is in the list reads witnessed_on_this_image True; a foreign fingerprint reads False and is STATED (never a refusal);
a sidecar without any fingerprint reads unknown."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chrombpnet_fastkit as kit
A = {"libcudnn.so.8": "aaaa", "libcublas.so.11": "bbbb", "libcudart.so.11.0": "cccc"}; C = {"libcudnn.so.8": "dddd", "libcublas.so.11": "eeee", "libcudart.so.11.0": "ffff"}; X = {"libcudnn.so.8": "9999", "libcublas.so.11": "8888", "libcudart.so.11.0": "7777"}
want = {"cuda_libs_sha16": A, "captures": [{"image": "im-A", "cuda_libs_sha16": A}, {"image": "im-C", "cuda_libs_sha16": C}]}
def test_witnessed_images():
    s = kit.image_key_status(want, A); assert s["witnessed_on_this_image"] is True and s["known_images"] == 2, s
    s = kit.image_key_status(want, C); assert s["witnessed_on_this_image"] is True and s["matched_image"] == "im-C", s
    s = kit.image_key_status(want, X); assert s["witnessed_on_this_image"] is False and "foreign" in s["note"] and "never a refusal" in s["note"], s
    s = kit.image_key_status({"gpu_name": "x"}, {}); assert s["witnessed_on_this_image"] is None and "unknown" in s["note"], s
    s = kit.image_key_status(want, {}); assert s["witnessed_on_this_image"] is None, s
if __name__ == "__main__":
    test_witnessed_images(); print("IMAGE-KEY CPU TEST OK")
