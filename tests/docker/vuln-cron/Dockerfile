FROM ubuntu:22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        openssh-server sudo cron coreutils procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/run/sshd /opt/scripts

# Low-privilege user with SSH access
RUN useradd -m -s /bin/bash testuser && \
    echo "testuser:testpass" | chpasswd

# ── Planted vulnerability ──────────────────────────────────
# Root cron job runs a world-writable script every minute
RUN echo '#!/bin/bash\necho "cleanup" > /dev/null' > /opt/scripts/cleanup.sh && \
    chmod 777 /opt/scripts/cleanup.sh && \
    echo "* * * * * root /opt/scripts/cleanup.sh" >> /etc/crontab

EXPOSE 22
# Start both cron and sshd
CMD service cron start && /usr/sbin/sshd -D
