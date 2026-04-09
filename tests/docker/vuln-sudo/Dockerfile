FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        openssh-server sudo findutils coreutils procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd

# Low-privilege user with SSH access
RUN useradd -m -s /bin/bash testuser && \
    echo "testuser:testpass" | chpasswd

# ── Planted vulnerability ──────────────────────────────────
# testuser can run /usr/bin/find as root without a password
RUN echo "testuser ALL=(ALL) NOPASSWD: /usr/bin/find" >> /etc/sudoers

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
