import ipaddress

import psutil

_PRIVATE = ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network(
    "172.16.0.0/12"
), ipaddress.ip_network("192.168.0.0/16")


def _is_private(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_link_local or any(addr in net for net in _PRIVATE)


def snapshot():
    """Return set of (dst_ip, dst_port) for all established OUTBOUND conns."""
    return {
        (c.raddr.ip, c.raddr.port)
        for c in psutil.net_connections()
        if c.raddr and c.status == "ESTABLISHED"
    }


def audit(callable_, *args, **kwargs):
    """Run callable_; assert zero NEW outbound destinations appeared while it ran.

    Loopback and private/LAN destinations are allowed. Raises AssertionError
    with the offending destinations on violation.
    """
    before = snapshot()
    result = callable_(*args, **kwargs)
    after = snapshot()
    new = after - before
    offending = {(ip, port) for ip, port in new if not _is_private(ip)}
    assert not offending, f"new outbound connection(s) during inference: {offending}"
    return result
