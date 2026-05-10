import tarfile
import urllib.request
import email
import re
import json
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR  = DATA_DIR / "raw"
OUT_DIR  = DATA_DIR / "processed"

HAM_URLS = [
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
]

NAZARIO_URL = (
    "https://raw.githubusercontent.com/shaeferd/Phishing-Detection"
    "/master/phishing3.mbox"
)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def extract_text_from_email(raw: bytes) -> str:
    try:
        msg = email.message_from_bytes(raw)
    except Exception:
        return ""
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body_parts.append(
                        part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    )
                except Exception:
                    pass
    else:
        try:
            body_parts.append(
                msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            )
        except Exception:
            pass
    return " ".join(body_parts).strip()


def clean(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_spamassassin(tarball: Path) -> list[str]:
    texts = []
    with tarfile.open(tarball, "r:bz2") as tf:
        for member in tf.getmembers():
            if member.isfile():
                f = tf.extractfile(member)
                if f:
                    raw = f.read()
                    text = clean(extract_text_from_email(raw))
                    if len(text) > 50:
                        texts.append(text)
    return texts


def load_nazario(mbox_path: Path) -> list[str]:
    # parsed line-by-line because non-ASCII bytes in phishing headers break mailbox.mbox
    texts = []
    current_lines = []

    def _process(raw_lines):
        raw = b"\n".join(raw_lines)
        try:
            text = clean(extract_text_from_email(raw))
            if len(text) > 50:
                texts.append(text)
        except Exception:
            pass

    try:
        with open(mbox_path, "rb") as fh:
            for line in fh:
                if line.startswith(b"From ") and current_lines:
                    _process(current_lines)
                    current_lines = [line.rstrip(b"\n")]
                else:
                    current_lines.append(line.rstrip(b"\n"))
            if current_lines:
                _process(current_lines)
    except Exception as e:
        print(f"  Warning loading Nazario mbox: {e}")

    return texts


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] SpamAssassin legitimate email corpus (easy_ham)")
    ham_texts = []
    for url in HAM_URLS:
        dest = RAW_DIR / url.split("/")[-1]
        try:
            download(url, dest)
            ham_texts.extend(load_spamassassin(dest))
        except Exception as e:
            print(f"  Could not process {dest.name}: {e}")

    print(f"  Loaded {len(ham_texts)} legitimate emails")

    print("\n[2/2] Nazario phishing corpus")
    naz_dest = RAW_DIR / "phishing3.mbox"
    try:
        download(NAZARIO_URL, naz_dest)
        phish_texts = load_nazario(naz_dest)
    except Exception as e:
        print(f"  Could not load Nazario corpus: {e}")
        phish_texts = []

    print(f"  Loaded {len(phish_texts)} phishing emails")

    with open(OUT_DIR / "ham.json", "w", encoding="utf-8") as f:
        json.dump(ham_texts, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "phishing.json", "w", encoding="utf-8") as f:
        json.dump(phish_texts, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Files saved to {OUT_DIR}/")
    print(f"  ham.json      : {len(ham_texts)} emails")
    print(f"  phishing.json : {len(phish_texts)} emails")


if __name__ == "__main__":
    main()
