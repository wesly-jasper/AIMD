"""
Content Credentials verification, against files signed here at test time.

The signing chain is generated per session (a throwaway root and an ES256 end
entity meeting the C2PA certificate profile), so the signature verifies while
the signer is, correctly, not on any trust list. ``cryptography`` comes in with
python-jose[cryptography].
"""
import datetime
import json

import pytest
from PIL import Image

c2pa = pytest.importorskip("c2pa")
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID  # noqa: E402

from app.services.preprocessing.metadata_extractor import MetadataExtractor  # noqa: E402
from app.services.provenance.c2pa_verifier import verify_c2pa  # noqa: E402
from app.services.provenance.timeline import build_timeline  # noqa: E402

CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
GENERATED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def _usage(**on):
    flags = dict(digital_signature=False, content_commitment=False, key_encipherment=False,
                 data_encipherment=False, key_agreement=False, key_cert_sign=False,
                 crl_sign=False, encipher_only=False, decipher_only=False)
    return x509.KeyUsage(**{**flags, **on})


@pytest.fixture(scope="module")
def signer():
    now = datetime.datetime.now(datetime.timezone.utc)
    valid = dict(not_valid_before=now - datetime.timedelta(days=1),
                 not_valid_after=now + datetime.timedelta(days=30))
    ca_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AIMD Test Root"),
                         x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIMD tests")])
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AIMD Test Signer"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AIMD tests")])
    ca = (x509.CertificateBuilder(subject_name=ca_name, issuer_name=ca_name,
                                  public_key=ca_key.public_key(),
                                  serial_number=x509.random_serial_number(), **valid)
          .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
          .add_extension(_usage(key_cert_sign=True, crl_sign=True), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    cert = (x509.CertificateBuilder(subject_name=name, issuer_name=ca_name,
                                    public_key=key.public_key(),
                                    serial_number=x509.random_serial_number(), **valid)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(_usage(digital_signature=True), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                           critical=False)
            .sign(ca_key, hashes.SHA256()))
    chain = cert.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM)
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    return c2pa.Signer.from_info(c2pa.C2paSignerInfo(
        alg=b"es256", sign_cert=chain, private_key=private, ta_url=None))


def sign(signer, tmp_path, source_type=CAPTURE, name="signed.jpg"):
    plain = tmp_path / f"plain-{name}"
    Image.new("RGB", (96, 64), (120, 80, 40)).save(plain, "JPEG")
    manifest = {
        "claim_generator_info": [{"name": "AIMD tests", "version": "1"}],
        "title": name,
        "assertions": [{"label": "c2pa.actions", "data": {"actions": [
            {"action": "c2pa.created", "digitalSourceType": source_type}]}}],
    }
    out = tmp_path / name
    with open(plain, "rb") as src, open(out, "w+b") as dst:
        c2pa.Builder(json.dumps(manifest)).sign(signer, "image/jpeg", src, dst)
    return out


def test_a_file_without_credentials_says_so(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (32, 32)).save(path, "JPEG")
    result = verify_c2pa(path)
    assert result["status"] == "NO_MANIFEST"
    assert result["sdk_version"]


def test_an_intact_manifest_from_an_unknown_signer_is_valid_not_trusted(signer, tmp_path):
    result = verify_c2pa(sign(signer, tmp_path))
    assert result["status"] == "VALID"
    assert result["trusted_signer"] is False
    assert "signingCredential.untrusted" in {f["code"] for f in result["failures"]}
    assert result["signer"]["common_name"] == "AIMD Test Signer"
    assert result["signer"]["issuer"] == "AIMD tests"
    assert result["claim_generator"] == "AIMD tests"
    assert result["actions"][0]["action"] == "c2pa.created"
    assert result["digital_source_types"] == ["digitalCapture"]
    assert result["ai_generation_declared"] is False


def test_a_file_changed_after_signing_is_invalid(signer, tmp_path):
    path = sign(signer, tmp_path)
    data = bytearray(path.read_bytes())
    data[-200] ^= 0xFF  # inside the image data, after the manifest
    path.write_bytes(bytes(data))
    result = verify_c2pa(path)
    assert result["status"] == "INVALID"
    assert result["failures"]


def test_declared_ai_generation_is_reported_as_the_signers_declaration(signer, tmp_path):
    result = verify_c2pa(sign(signer, tmp_path, GENERATED, "generated.jpg"))
    assert result["status"] == "VALID"
    assert result["digital_source_types"] == ["trainedAlgorithmicMedia"]
    assert result["ai_generation_declared"] is True


def test_a_broken_manifest_declares_nothing(signer, tmp_path):
    path = sign(signer, tmp_path, GENERATED, "generated-tampered.jpg")
    data = bytearray(path.read_bytes())
    data[-200] ^= 0xFF
    path.write_bytes(bytes(data))
    result = verify_c2pa(path)
    assert result["status"] == "INVALID" and result["ai_generation_declared"] is False


def test_every_extracted_image_carries_its_c2pa_result(signer, tmp_path):
    meta = MetadataExtractor().extract(sign(signer, tmp_path))
    assert meta["c2pa"]["status"] == "VALID"


def test_an_intact_signature_time_is_a_metadata_claim_on_the_timeline():
    credentials = {"status": "VALID", "trusted_signer": False, "manifest_label": "urn:c2pa:1",
                   "signer": {"common_name": "Some Camera", "signed_at": "2024-03-01T10:00:00Z"}}
    [event] = build_timeline([], c2pa=credentials)["events"]
    assert event["type"] == "METADATA" and event["basis"] == "c2pa_manifest"
    assert event["timestamp"] == "2024-03-01T10:00:00+00:00"
    assert "untrusted" in event["description"]
    assert build_timeline([], c2pa={**credentials, "status": "INVALID"})["events"] == []
