"""cache_content_witness() of the shipped driver cache, on a fake dir (no GPU, no TF): the kit's own install-marker files are never 'new'
(the marker alone must not read 'MISS (0 rewritten, 1 new …)'); the shipped-member list comes from the
MEMBERS.sha256 sidecar on every path (the marker-HIT path included); a rewritten member -> MISS by content; a foreign file -> 'new 1', named."""
import os, sys, tempfile, hashlib, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import chrombpnet_fastkit as kit
def _mk(d, files):
    members = {}
    for rel, data in files.items():
        p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data); members[rel] = hashlib.sha256(data).hexdigest()
    return members
def test_marker_is_not_a_new_member():
    d = tempfile.mkdtemp(); members = _mk(d, {"a/k1.bin": b"kernel-1", "b/k2.bin": b"kernel-2"})
    open(os.path.join(d, ".chrombpnet_fastkit_cache_installed_deadbeef00000000.json"), "w").write(json.dumps({"utc": "x", "members": 2}))
    w = kit.cache_content_witness(d, members); assert w["verdict"] == "HIT", w; assert w["markers"] == 1, w
def test_rewritten_member_is_a_miss_by_content():
    d = tempfile.mkdtemp(); members = _mk(d, {"a/k1.bin": b"kernel-1", "b/k2.bin": b"kernel-2"}); open(os.path.join(d, "a/k1.bin"), "wb").write(b"kernel-1-rewritten")
    w = kit.cache_content_witness(d, members); assert w["verdict"].startswith("MISS (1 rewritten, 0 new, 0 missing of 2 shipped)"), w; assert w["rewritten"] == ["a/k1.bin"], w
def test_foreign_file_is_new_and_named():
    d = tempfile.mkdtemp(); members = _mk(d, {"a/k1.bin": b"kernel-1"}); open(os.path.join(d, "c/k9.bin") if os.makedirs(os.path.join(d, "c"), exist_ok=True) is None else "", "wb").write(b"jit-written")
    w = kit.cache_content_witness(d, members); assert w["verdict"].startswith("MISS (0 rewritten, 1 new, 0 missing of 1 shipped)"), w; assert w["new"] == ["c/k9.bin"], w
def test_shipped_members_sidecar_read_on_every_path():
    d = tempfile.mkdtemp(); tar = os.path.join(d, "nv_compute_cache_H100.tar"); open(tar, "wb").write(b"not-a-real-tar")
    open(tar + ".MEMBERS.sha256", "w").write("aa" * 32 + "  a/k1.bin\n" + "bb" * 32 + "  b/k2.bin\n")
    m = kit._shipped_members(tar); assert m == {"a/k1.bin": "aa" * 32, "b/k2.bin": "bb" * 32}, m
    assert kit._shipped_members(os.path.join(d, "absent.tar")) == {}
def test_no_list_wording():
    d = tempfile.mkdtemp(); _mk(d, {"a/k1.bin": b"kernel-1"}); w = kit.cache_content_witness(d, {}); assert w["verdict"].startswith("MISS (no shipped-member list"), w
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("CACHE-WITNESS CPU TEST OK")
