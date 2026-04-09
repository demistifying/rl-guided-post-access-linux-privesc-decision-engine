FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        openssh-server sudo python3 coreutils procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd

# Low-privilege user with SSH access
RUN useradd -m -s /bin/bash testuser && \
    echo "testuser:testpass" | chpasswd

# ── Planted vulnerability ──────────────────────────────────
# python3 has the SUID bit set → can call os.setuid(0)
RUN chmod u+s /usr/bin/python3*

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
