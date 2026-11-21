"""Self-signed HTTPS certificate: created on first start, renewed when it nears expiry or this machine's
names/addresses change. Browsers warn once (it isn't signed by a public authority); accept it, or import
cert.pem into the browser/OS trust store to silence the warning."""
from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

log = logging.getLogger(__name__)
VALID_DAYS = 3650
RENEW_DAYS = 30


def local_names() -> tuple[set[str], set[str]]:
    """DNS names and IP addresses this machine is likely reached by."""
    names = {"localhost"}
    for n in (socket.gethostname(), socket.getfqdn()):
        if n:
            names.update({n, f"{n.split('.')[0]}.local"})
    ips = {"127.0.0.1", "::1"}
    try:                                       # IPv4 only: IPv6 privacy addresses rotate daily (renewal churn)
        for *_, sa in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(sa[0])
    except OSError:
        pass
    try:                                       # the address of the default route (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return names, ips


def _san(cert: x509.Certificate) -> tuple[set[str], set[str]]:
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return set(), set()
    return set(san.get_values_for_type(x509.DNSName)), {str(i) for i in san.get_values_for_type(x509.IPAddress)}


def ensure_cert(folder: Path) -> tuple[Path, Path]:
    """Return (cert, key) paths in `folder`, (re)creating a self-signed pair when needed."""
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    cert_p, key_p = folder / "cert.pem", folder / "key.pem"
    names, ips = local_names()
    if cert_p.exists() and key_p.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_p.read_bytes())
            end = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(tzinfo=dt.timezone.utc)
            left = end - dt.datetime.now(dt.timezone.utc)
            have_n, have_i = _san(cert)
            if left > dt.timedelta(days=RENEW_DAYS) and names <= have_n and ips <= have_i:
                return cert_p, key_p
            # keep the old names/addresses: at boot the network may not be up yet, and dropping the LAN
            # address would break (and re-warn) every browser that reaches TheEye by it
            names |= have_n
            ips |= have_i
            log.info("renewing HTTPS certificate (%s)", "expiring" if left <= dt.timedelta(days=RENEW_DAYS)
                     else "this machine's names/addresses changed")
        except ValueError:
            log.warning("unreadable HTTPS certificate in %s; creating a new one", folder)

    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname() or "localhost"
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"TheEye ({host})"[:64]),
                         x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TheEye self-signed")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
            .add_extension(x509.SubjectAlternativeName(
                [x509.DNSName(n) for n in sorted(names)] +
                [x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(ips)]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=False,
                                         content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))

    fd = os.open(key_p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    os.chmod(key_p, 0o600)
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    log.info("created a self-signed HTTPS certificate in %s (valid %d years)", folder, VALID_DAYS // 365)
    return cert_p, key_p
